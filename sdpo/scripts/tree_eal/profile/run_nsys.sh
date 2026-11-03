#!/bin/bash
# Per-stage time and GPU utilization of the dense_tree + exact_starts training step, on one GPU.
#   (1) NVTX + hardware GPU metrics only (low overhead): Temporal and Probability
#   (2) + CUDA kernel trace (Temporal): kernel counts, kernel busy fraction, kernel categories
# GPU metrics are device-wide, so run on a GPU nobody else is using (see wait_idle_gpu.sh).
#
# Usage:
#   DATA=/path/to/alpaca_regen.jsonl OUT=/path/to/outdir bash run_nsys.sh <gpu>
set -u
GPU=$1
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=${DATA:?set DATA to a ShareGPT-format jsonl}
OUT=${OUT:-$PWD/nsys_out}
PY=${PY:-python}
mkdir -p "$OUT" && cd "$OUT"
export CUDA_VISIBLE_DEVICES=$GPU TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NS="nsys profile --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop --sample=none --cpuctxsw=none"
METRICS="--trace=nvtx --gpu-metrics-devices=cuda-visible --gpu-metrics-frequency=10000"

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > gpu_state_start.txt
$NS -o metrics_temporal $METRICS "$PY" "$HERE/profile_nvtx.py" --data "$DATA" --n_steps 3 > metrics_temporal.log 2>&1
$NS -o metrics_prob $METRICS "$PY" "$HERE/profile_nvtx.py" --data "$DATA" --n_steps 3 --prob_starts > metrics_prob.log 2>&1
$NS -o kernels_temporal --trace=cuda,nvtx "$PY" "$HERE/profile_nvtx.py" --data "$DATA" --n_steps 2 > kernels_temporal.log 2>&1
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader > gpu_state_end.txt

for r in metrics_temporal metrics_prob kernels_temporal; do
  nsys export --type sqlite --force-overwrite true -o $r.sqlite $r.nsys-rep > /dev/null 2>&1
  python3 "$HERE/analyze_nsys.py" $r.sqlite $r.json > $r.txt 2>&1
done
echo "done: $OUT/{metrics_temporal,metrics_prob,kernels_temporal}.txt"
