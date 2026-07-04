#!/usr/bin/env bash
#SBATCH --job-name=baldro_drnpo_l2
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-4%1
#SBATCH --output=logs/baldro_fq_audit/%x-%A_%a.out
#SBATCH --error=logs/baldro_fq_audit/%x-%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/home/zkzhang/unlearning/BalDRO}"
CONDA_ENV="${CONDA_ENV:-unlearning-new}"
MODEL_ROOT="${MODEL_ROOT:-/home/zkzhang/models}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/tofu_Llama-2-7b-chat-hf_full}"
MODEL_CONFIG="${MODEL_CONFIG:-Llama-2-7b-chat-hf}"
LR="${LR:-1e-5}"
TRAIN_BSZ="${TRAIN_BSZ:-1}"
GRAD_ACC="${GRAD_ACC:-32}"
EPOCHS="${EPOCHS:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
DO_SAVE="${DO_SAVE:-True}"
NPO_BETA="${NPO_BETA:-0.1}"
NPO_ALPHA="${NPO_ALPHA:-1.0}"
NPO_GAMMA="${NPO_GAMMA:-1.0}"

cd "${ROOT}"
mkdir -p logs/baldro_fq_audit results/baldro_fq_audit/train

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  . "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/home/zkzhang/unlearning/HF_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-BalDRO-FQ-Audit}"

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Missing original model path: ${MODEL_PATH}" >&2
  exit 2
fi

BETA_DV_GRID=(0.5 1.0 2.0 5.0 10.0)
beta_dv_forget="${BETA_DV_GRID[${SLURM_ARRAY_TASK_ID:-0}]}"

forget_split="forget01"
holdout_split="holdout01"
retain_split="retain99"
retain_log="saves/eval/tofu_${MODEL_CONFIG}_${retain_split}/TOFU_EVAL.json"
if [ ! -f "${retain_log}" ]; then
  echo "Missing bundled retain log: ${retain_log}" >&2
  exit 3
fi

echo "===== TOFU CACHE PREFLIGHT ====="
python - <<PY
from datasets import load_dataset

configs = [
    "${forget_split}",
    "${retain_split}",
    "${forget_split}_perturbed",
    "${holdout_split}",
    "retain_perturbed",
    "real_authors_perturbed",
    "world_facts_perturbed",
]
for name in configs:
    ds = load_dataset("locuslab/TOFU", name)
    print(f"{name}: {ds}")
PY

suffix="lr${LR}_b${TRAIN_BSZ}_ga${GRAD_ACC}_beta${NPO_BETA}_bdv${beta_dv_forget}_e${EPOCHS}"
task_name="fq_audit_unlearn_tofu_${MODEL_CONFIG}_${forget_split}_DrNPO_${suffix}"
output_dir="results/baldro_fq_audit/train/${task_name}"

echo "Training DrNPO ${forget_split}, beta_dv_forget=${beta_dv_forget}"
echo "  hf_home=${HF_HOME}"
echo "  retain_log=${retain_log}"

python src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  trainer=DrNPO \
  model="${MODEL_CONFIG}" \
  model.model_args.pretrained_model_name_or_path="${MODEL_PATH}" \
  model.tokenizer_args.pretrained_model_name_or_path="${MODEL_PATH}" \
  forget_split="${forget_split}" \
  holdout_split="${holdout_split}" \
  retain_split="${retain_split}" \
  task_name="${task_name}" \
  paths.output_dir="${output_dir}" \
  do_save="${DO_SAVE}" \
  retain_logs_path="${retain_log}" \
  eval.tofu.batch_size="${EVAL_BATCH_SIZE}" \
  trainer.args.ddp_find_unused_parameters=true \
  trainer.args.gradient_checkpointing=true \
  trainer.args.report_to=wandb \
  trainer.args.run_name="${task_name}" \
  trainer.args.logging_steps=1 \
  trainer.args.learning_rate="${LR}" \
  trainer.args.per_device_train_batch_size="${TRAIN_BSZ}" \
  trainer.args.gradient_accumulation_steps="${GRAD_ACC}" \
  trainer.args.num_train_epochs="${EPOCHS}" \
  trainer.args.eval_strategy=epoch \
  trainer.args.eval_on_start=True \
  trainer.method_args.beta="${NPO_BETA}" \
  trainer.method_args.alpha="${NPO_ALPHA}" \
  trainer.method_args.gamma="${NPO_GAMMA}" \
  trainer.method_args.retain_loss_type=NLL \
  trainer.method_args.beta_dv_forget="${beta_dv_forget}" \
  trainer.method_args.beta_dv_retain=1.0 \
  trainer.method_args.forget_dro=True \
  trainer.method_args.retain_dro=False

echo "Done: ${output_dir}"
