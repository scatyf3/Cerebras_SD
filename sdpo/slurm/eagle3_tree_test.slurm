#!/bin/bash
# GPU checks for the EAGLE-3 Tree-EAL port (sdpo/scripts/eagle3/test_eagle3_tree.py).
#   sbatch sdpo/slurm/eagle3_tree_test.slurm            # PART=all
#   sbatch --export=ALL,PART=oss sdpo/slurm/eagle3_tree_test.slurm
#SBATCH --job-name=e3_test
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --partition=h200_tandon,h200_public
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/yf3005/sdpo_eagle3_tree_eal/logs/%x_%j.log

set -uo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
source "${ENV:-/scratch/yf3005/envs/cerebras-sd}/bin/activate"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=/scratch/yf3005/sdpo_eagle3_tree_eal/hf_datasets_cache
cd "$REPO"
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is) head=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
python -u sdpo/scripts/eagle3/test_eagle3_tree.py --part "${PART:-all}"
rc=$?
echo "end=$(date -Is) rc=$rc"
exit $rc
