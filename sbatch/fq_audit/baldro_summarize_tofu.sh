#!/usr/bin/env bash
#SBATCH --job-name=baldro_sum
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/baldro_fq_audit/%x-%j.out
#SBATCH --error=logs/baldro_fq_audit/%x-%j.err

set -euo pipefail

ROOT="${ROOT:-/home/zkzhang/unlearning/BalDRO}"
CONDA_ENV="${CONDA_ENV:-baldro}"
SEARCH_ROOT="${SEARCH_ROOT:-results/baldro_fq_audit}"
OUT_CSV="${OUT_CSV:-results/baldro_fq_audit/tofu_summary.csv}"

cd "${ROOT}"
mkdir -p logs/baldro_fq_audit "$(dirname "${OUT_CSV}")"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  . "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

python tools/fq_audit/summarize_tofu.py \
  --root "${SEARCH_ROOT}" \
  --out "${OUT_CSV}"

echo "Wrote ${OUT_CSV}"
