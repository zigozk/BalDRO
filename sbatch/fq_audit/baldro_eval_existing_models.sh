#!/usr/bin/env bash
#SBATCH --job-name=baldro_eval_audit
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 4:00:00
#SBATCH --output=logs/baldro_fq_audit/%x-%A_%a.out
#SBATCH --error=logs/baldro_fq_audit/%x-%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/home/zkzhang/unlearning/BalDRO}"
CONDA_ENV="${CONDA_ENV:-unlearning}"
MODEL_ROOT="${MODEL_ROOT:-/home/zkzhang/models}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

cd "${ROOT}"
mkdir -p logs/baldro_fq_audit results/baldro_fq_audit/eval

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

# label|model_config|model_path|forget_split|holdout_split|retain_log
CASES=(
  "l2_full_f01|Llama-2-7b-chat-hf|${MODEL_ROOT}/tofu_Llama-2-7b-chat-hf_full|forget01|holdout01|saves/eval/tofu_Llama-2-7b-chat-hf_retain99/TOFU_EVAL.json"
  "l2_full_f05|Llama-2-7b-chat-hf|${MODEL_ROOT}/tofu_Llama-2-7b-chat-hf_full|forget05|holdout05|saves/eval/tofu_Llama-2-7b-chat-hf_retain95/TOFU_EVAL.json"
  "l2_full_f10|Llama-2-7b-chat-hf|${MODEL_ROOT}/tofu_Llama-2-7b-chat-hf_full|forget10|holdout10|saves/eval/tofu_Llama-2-7b-chat-hf_retain90/TOFU_EVAL.json"
  "l3_1b_full_f10|Llama-3.2-1B-Instruct|${MODEL_ROOT}/tofu_Llama-3.2-1B-Instruct_full|forget10|holdout10|saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"
  "l3_1b_official_npo_f10|Llama-3.2-1B-Instruct|${MODEL_ROOT}/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.5_alpha1_epoch10|forget10|holdout10|saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"
)

case_spec="${CASES[${SLURM_ARRAY_TASK_ID:-0}]}"
IFS="|" read -r label model_config model_path forget_split holdout_split retain_log <<< "${case_spec}"

if [ ! -d "${model_path}" ]; then
  echo "Missing model path: ${model_path}" >&2
  exit 2
fi
if [ ! -f "${retain_log}" ]; then
  echo "Missing retain reference log: ${retain_log}" >&2
  exit 3
fi

task_name="fq_audit_${label}"
output_dir="results/baldro_fq_audit/eval/${task_name}"

echo "Evaluating ${label}"
echo "  model_config=${model_config}"
echo "  model_path=${model_path}"
echo "  forget_split=${forget_split}"
echo "  retain_log=${retain_log}"
echo "  hf_home=${HF_HOME}"

python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model="${model_config}" \
  task_name="${task_name}" \
  paths.output_dir="${output_dir}" \
  model.model_args.pretrained_model_name_or_path="${model_path}" \
  model.tokenizer_args.pretrained_model_name_or_path="${model_path}" \
  forget_split="${forget_split}" \
  holdout_split="${holdout_split}" \
  retain_logs_path="${retain_log}" \
  eval.tofu.batch_size="${EVAL_BATCH_SIZE}" \
  eval.tofu.overwrite=true

echo "Done: ${output_dir}/TOFU_SUMMARY.json"
