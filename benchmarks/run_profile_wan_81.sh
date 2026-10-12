#!/usr/bin/env bash
# Independent diagnostic run; keep the prior benchmark results unchanged.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
result_root="${1:-../outputs/profiles/wan_f81_s50_$(date +%Y%m%d_%H%M%S)}"
export PYTHONPATH="$PWD:$PWD/../Wan2.1${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "Official single-GPU profile: $result_root/single"
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/torchrun \
  --standalone --nnodes=1 --nproc_per_node=1 \
  --module benchmarks.profile_wan --tp-size 1 --output-dir "$result_root/single"
echo "TP2 profile: $result_root/tp2"
CUDA_VISIBLE_DEVICES=0,1 ../.venv/bin/torchrun \
  --standalone --nnodes=1 --nproc_per_node=2 \
  --module benchmarks.profile_wan --tp-size 2 --output-dir "$result_root/tp2"
echo "Completed. Phase/memory reports, operator tables and traces: $result_root"
