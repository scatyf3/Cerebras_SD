#!/bin/bash
# Same-recipe Tree-EAL training with --rollout grad (original) vs graph, two
# seeds each, one run per GPU. If graph is equivalent, the grad-vs-graph gap in
# val E[L] should be within the seed-to-seed spread.
#
# Usage:
#   DATA=/path/to/data bash sdpo/scripts/tree_eal/run_rollout_equiv.sh
# Expects $DATA/alpaca_regen_s*.jsonl (from gen_alpaca_regen.py); holds out
# $N_VAL of them as the val set.
set -euo pipefail

DATA=${DATA:-/mnt/hdd/yf/Cerebras_SD_data}
OUT=${OUT:-$DATA/rollout_equiv}
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
PY=${PY:-python}
N_VAL=${N_VAL:-80}
EPOCHS=${EPOCHS:-2}
EAL_MODE=${EAL_MODE:-sample_minpq}
RUNS=${RUNS:-"grad:0 graph:0 grad:1 graph:1"}   # rollout:seed, one per GPU
GPUS=(${GPUS:-0 1 2 3})

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

"$PY" - "$DATA" "$N_VAL" <<'EOF'
import glob, json, random, sys
data, n_val = sys.argv[1], int(sys.argv[2])
rows = [l for f in sorted(glob.glob(f"{data}/alpaca_regen_s*.jsonl")) for l in open(f)]
random.Random(0).shuffle(rows)
open(f"{data}/equiv_val.jsonl", "w").writelines(rows[:n_val])
open(f"{data}/equiv_train.jsonl", "w").writelines(rows[n_val:])
print(f"train {len(rows) - n_val} val {n_val}")
EOF

i=0
for run in $RUNS; do
  ro=${run%:*}; seed=${run#*:}
  dir=$OUT/${EAL_MODE}_${ro}_s$seed
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} "$PY" "$REPO/sdpo/scripts/tree_eal/train_rollout_equiv.py" \
    --rollout "$ro" --seed "$seed" --epochs "$EPOCHS" --eal_mode "$EAL_MODE" \
    --train "$DATA/equiv_train.jsonl" --val "$DATA/equiv_val.jsonl" --out "$dir" \
    > "$dir/stdout.log" 2>&1 &
  i=$((i + 1))
done
wait
