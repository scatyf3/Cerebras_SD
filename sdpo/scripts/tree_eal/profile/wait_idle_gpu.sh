#!/bin/bash
# On a shared machine: wait until a GPU stays idle (< 5% util, > MIN_FREE_MB free) for 4 polls
# 15 s apart, then run run_nsys.sh on it. Same environment variables as run_nsys.sh.
HERE=$(cd "$(dirname "$0")" && pwd)
MIN_FREE_MB=${MIN_FREE_MB:-34000}
declare -A ok
while true; do
  while IFS=, read -r idx used total util; do
    idx=${idx// /}; used=${used// /}; total=${total// /}; util=${util// /}
    if [ "$util" -lt 5 ] && [ $((total - used)) -gt "$MIN_FREE_MB" ]; then
      ok[$idx]=$(( ${ok[$idx]:-0} + 1 ))
    else
      ok[$idx]=0
    fi
    if [ "${ok[$idx]}" -ge 4 ]; then
      echo "$(date +%T) GPU $idx idle, starting"
      exec bash "$HERE/run_nsys.sh" "$idx"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)
  sleep 15
done
