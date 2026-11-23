#!/bin/bash
# Tree-EAL with an EAGLE-3 draft head (default target gpt-oss-20b).
#
# Stage 0 (once): make a fresh head with a draft vocab from the training data
#   python sdpo/scripts/eagle3/init_eagle3_draft.py --target $BASE --data $TRAIN --out $INIT
# Stage 1: plain EAGLE-3 warm start (training-time test, soft CE):
#   sbatch -J e3_ttt --export=ALL,TAG=oss20b_e3_ttt,EAL_MODE=eagle3_ttt,DRAFT=$INIT,LR=1e-4 sdpo/slurm/eagle3_tree_eal_train.slurm
# Stage 2: a Tree-EAL variant from the warm-started head:
#   sbatch -J e3_rsd --export=ALL,TAG=oss20b_e3_rsd,EAL_MODE=sample_rsd,DRAFT=$SAVEROOT/oss20b_e3_ttt/hf_epoch_1 sdpo/slurm/eagle3_tree_eal_train.slurm
# EAL_MODE: any of eagle3_tree_model.EAGLE3_MODES (sample_rsd, sample_minpq,
# soft_topk, top4_kl_eal, greedy_topk, budget, path_topk, sample_anyk, ...).
#
# No per-epoch accept-length eval is queued (eval_small_lm_tree.py is small-LM only).
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yf3005/sdpo_eagle3_tree_eal/logs/train_%x_%j.log

set -uo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
ENV=${ENV:-/scratch/yf3005/envs/cerebras-sd}
BASE=${BASE:-/scratch/yf3005/.huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee}
DATA=${DATA:-/scratch/yf3005/gto_data/gptoss20b/split}
TRAIN=${TRAIN:-$DATA/train.jsonl}
VAL=${VAL:-$DATA/val80.jsonl}
DRAFT=${DRAFT:?set DRAFT to an EAGLE-3 head dir}
SAVEROOT=${SAVEROOT:-/scratch/yf3005/sdpo_eagle3_tree_eal}
TAG=${TAG:?set TAG}
SAVE=$SAVEROOT/$TAG
EAL_MODE=${EAL_MODE:-sample_rsd}
DS=${DS:-sdpo/scripts/tree_eal/ds_zero2_a100_2gpu.json}
if [ "$EAL_MODE" = eagle3_ttt ]; then LR=${LR:-1e-4}; else LR=${LR:-1e-5}; fi

source "$ENV/bin/activate"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$SAVEROOT/hf_datasets_cache}
export MASTER_PORT=$((29500 + RANDOM % 1000))
export EVAL_SBATCH=""

mkdir -p "$SAVE" "$SAVEROOT/logs"
cd "$REPO"
echo "node=$(hostname) job=$SLURM_JOB_ID part=$SLURM_JOB_PARTITION start=$(date -Is) tag=$TAG mode=$EAL_MODE"
echo "head=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l) train=$TRAIN draft=$DRAFT lr=$LR ds=$DS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

python -m deepspeed.launcher.runner --master_port "$MASTER_PORT" sdpo/main_small_lm_tree_v2.py \
  --draft_type eagle3 \
  --basepath "$BASE" --draftpath "$DRAFT" \
  --trainpath "$TRAIN" --testpath "$VAL" \
  --deepspeed_config "$DS" --savedir "$SAVE" \
  --eal_mode "$EAL_MODE" \
  --top_k_per_depth "${TOPK:-4,3,2,1,1,1,1}" --tree_budget 0 \
  --temperature "${TEMP:-0.5}" --path_topk "${PATH_TOPK:-4}" \
  --n_anchors "${N_ANCHORS:-4}" --n_draws 2 \
  --anchor_kl_coef "${ANCHOR_KL_COEF:-0.1}" --eal_aux_coef "${EAL_AUX_COEF:-1.0}" \
  --eal_score_coef 0.0 \
  --ttt_length "${TTT_LENGTH:-7}" \
  --lr "$LR" --max_len "${MAX_LEN:-2048}" \
  --max_train_samples "${MAX_SAMPLES:-60000}" \
  --num_epochs "${EPOCHS:-2}" --seed "${SEED:-0}"
rc=${PIPESTATUS[0]}
echo "end=$(date -Is) rc=$rc"
exit $rc
