#!/usr/bin/env bash
#SBATCH --job-name=baldro_ref_audit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%1
#SBATCH --output=logs/baldro_fq_audit/%x-%A_%a.out
#SBATCH --error=logs/baldro_fq_audit/%x-%A_%a.err

set -euo pipefail

ROOT="${ROOT:-/home/zkzhang/unlearning/BalDRO}"
CONDA_ENV="${CONDA_ENV:-baldro}"
MODEL_ROOT="${MODEL_ROOT:-/home/zkzhang/models}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

cd "${ROOT}"
mkdir -p logs/baldro_fq_audit results/baldro_fq_audit/retain_refs

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  . "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# This evaluates retain models on the corresponding forget split. The resulting
# TOFU_EVAL.json contains `forget_truth_ratio`, which FQ uses as the retain
# reference distribution.
#
# label|model_config|retain_model_path|forget_split|holdout_split|retain_split
CASES=(
  "l2_retain99_for_f01|Llama-2-7b-chat-hf|${MODEL_ROOT}/tofu_Llama-2-7b-chat-hf_retain99|forget01|holdout01|retain99"
  "l2_retain95_for_f05|Llama-2-7b-chat-hf|${MODEL_ROOT}/tofu_Llama-2-7b-chat-hf_retain95|forget05|holdout05|retain95"
  "l2_retain90_for_f10|Llama-2-7b-chat-hf|${MODEL_ROOT}/tofu_Llama-2-7b-chat-hf_retain90|forget10|holdout10|retain90"
  "l3_1b_retain99_for_f01|Llama-3.2-1B-Instruct|${MODEL_ROOT}/tofu_Llama-3.2-1B-Instruct_retain99|forget01|holdout01|retain99"
  "l3_1b_retain95_for_f05|Llama-3.2-1B-Instruct|${MODEL_ROOT}/tofu_Llama-3.2-1B-Instruct_retain95|forget05|holdout05|retain95"
  "l3_1b_retain90_for_f10|Llama-3.2-1B-Instruct|${MODEL_ROOT}/tofu_Llama-3.2-1B-Instruct_retain90|forget10|holdout10|retain90"
)

case_spec="${CASES[${SLURM_ARRAY_TASK_ID:-0}]}"
IFS="|" read -r label model_config model_path forget_split holdout_split retain_split <<< "${case_spec}"

if [ ! -d "${model_path}" ]; then
  echo "Missing retain model path: ${model_path}" >&2
  echo "Either download this retain model or submit only the array ids that exist." >&2
  echo "Skipping ${label}."
  exit 0
fi

task_name="fq_audit_ref_${label}"
output_dir="results/baldro_fq_audit/retain_refs/${task_name}"

echo "Generating retain reference ${label}"
echo "  model_path=${model_path}"
echo "  forget_split=${forget_split}"

python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model="${model_config}" \
  task_name="${task_name}" \
  paths.output_dir="${output_dir}" \
  model.model_args.pretrained_model_name_or_path="${model_path}" \
  model.tokenizer_args.pretrained_model_name_or_path="${model_path}" \
  forget_split="${forget_split}" \
  holdout_split="${holdout_split}" \
  retain_logs_path=null \
  eval.tofu.batch_size="${EVAL_BATCH_SIZE}" \
  eval.tofu.overwrite=true

echo "Done: ${output_dir}/TOFU_EVAL.json"
