#!/bin/bash
# Regenerate ShareGPT V4.3 answers with gpt-oss-20b (harmony: analysis + final kept).
#   smoke: sbatch -J oss_regen_smoke --time=01:00:00 --export=ALL,MAX_SAMPLES=200,OUTPUT=/scratch/yf3005/gto_data/gptoss20b/smoke200_raw.jsonl sdpo/slurm/gptoss_regen.slurm
#   full:  sbatch -J oss_regen sdpo/slurm/gptoss_regen.slurm
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yf3005/gto_data/gptoss20b/logs/%x_%j.log

set -uo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
ENV=${ENV:-/scratch/yf3005/envs/cerebras-sd}
MODEL=${MODEL:-/scratch/yf3005/.huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee}
DATA=${DATA:-/scratch/yf3005/.huggingface/hub/datasets--Aeala--ShareGPT_Vicuna_unfiltered/snapshots/8b0048ad6ae8c22f46a78c15559dec98feef5539/ShareGPT_V4.3_unfiltered_cleaned_split.json}
OUTPUT=${OUTPUT:-/scratch/yf3005/gto_data/gptoss20b/sharegpt_gptoss20b_regen.jsonl}
REASONING=${REASONING:-medium}

source "$ENV/bin/activate"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# ~/.cache/vllm is a dangling symlink; keep compile caches on scratch
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/scratch/yf3005/.cache/vllm} TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/scratch/yf3005/.cache/triton}
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"
mkdir -p "$(dirname "$OUTPUT")"
cd "$REPO"
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) head=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
echo "model=$MODEL reasoning=$REASONING output=$OUTPUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# --system-prompt '': gpt-oss keeps only its built-in system message, as at deployment.
python -u sdpo/data/regen_sharegpt.py \
  --model "$MODEL" --data "$DATA" --output "$OUTPUT" \
  --tensor-parallel-size "${TP:-1}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" --max-tokens "${MAX_TOKENS:-4096}" \
  --batch-size "${BATCH_SIZE:-500}" --max-samples "${MAX_SAMPLES:-0}" \
  --checkpoint-every "${CKPT_EVERY:-10000}" \
  --temperature "${TEMPERATURE:-0.7}" --top-p "${TOP_P:-0.9}" \
  --dtype bfloat16 --system-prompt '' --reasoning-effort "$REASONING"
rc=$?
echo "end=$(date -Is) rc=$rc"
exit $rc
