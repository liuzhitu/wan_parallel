#!/usr/bin/env bash
# Run on cloud Linux. Single-GPU reference and TP2 share one workload.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-all}"
result_root="${2:-../outputs/benchmarks/wan_f81_s50_$(date +%Y%m%d_%H%M%S)}"
case "$mode" in
  all|single|tp2) ;;
  *) echo "Usage: bash benchmarks/run_wan_81.sh [all|single|tp2] [result_root]" >&2; exit 2 ;;
esac

export PYTHONPATH="$PWD:$PWD/../Wan2.1${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
common=(
  --wan-dir ../Wan2.1
  --ckpt-dir ../models/Wan2.1-T2V-1.3B
  --size '832*480' --frames 81 --steps 50 --seed 42
  --shift 8 --guide-scale 6 --warmup 1 --runs 3
  --t5-cpu --offload-model
)

echo "81 frames / 50 steps; results: $result_root"
if [[ "$mode" == all || "$mode" == single ]]; then
  echo 'Starting official single-GPU reference (1 warmup + 3 measured runs).'
  CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python \
    benchmarks/benchmark_wan_single_gpu.py "${common[@]}" \
    --output-dir "$result_root/single"
fi

if [[ "$mode" == all || "$mode" == tp2 ]]; then
  comparison=()
  if [[ -f "$result_root/single/metrics.json" ]]; then
    comparison=(--baseline-metrics "$result_root/single/metrics.json")
  fi
  echo 'Starting TP2 (1 warmup + 3 measured runs).'
  CUDA_VISIBLE_DEVICES=0,1 ../.venv/bin/torchrun \
    --standalone --nnodes=1 --nproc_per_node=2 \
    --module benchmarks.benchmark_wan_tp2 \
    "${common[@]}" --tp-size 2 --sample-solver unipc \
    --timeout-seconds 3600 --output-dir "$result_root/tp2" \
    "${comparison[@]}"
fi
echo "Completed. Inspect metrics.json and last_run.mp4 under $result_root/single/ and/or $result_root/tp2/."
