


















#!/usr/bin/env bash
# Group9: no changes to vLLM, FlagGems, or competition benchmark scripts.
set -euo pipefail
cd "$(dirname "$0")"
MODE=${MODE:-full}           # smoke|full
PROFILE=${PROFILE:-0}       # optional ixkn-cli profiling, not for timing
REQUIRE_GRAPH=${REQUIRE_GRAPH:-0}
OUT_DIR=${OUT_DIR:-"results/run_$(date -u +%Y%m%d_%H%M%S)_$$"}
mkdir -p "$OUT_DIR"
python3 test_execution_lab_math.py 2>&1 | tee "$OUT_DIR/cpu_tests.log"

extra=()
if [[ "$REQUIRE_GRAPH" == 1 ]]; then
  extra+=(--require-graph)
fi

if [[ "$MODE" == smoke ]]; then
  echo '== Group9 correctness-only GPU smoke test =='
  for lab in L01 L02 L03 L04; do
    echo "-- $lab --"
    python3 triton_execution_lab.py --lab "$lab" \
      --n 1003 --tiny-n 257 --graph-n 513 --block 256 \
      --launches 1,2 --graph-steps 1,4 --workers 2,4 --iters 8 \
      --single-run --strict "${extra[@]}" \
      --out "$OUT_DIR/${lab}_smoke.csv" \
      2>&1 | tee "$OUT_DIR/${lab}_smoke.log"
  done
elif [[ "$MODE" == full ]]; then
  echo '== Group9 GPU event + synchronized wall-time A/B tests =='
  for lab in L01 L02 L03 L04; do
    echo "-- $lab --"
    python3 triton_execution_lab.py --lab "$lab" \
      --strict "${extra[@]}" \
      --dump-asm "$OUT_DIR/asm" --out "$OUT_DIR/${lab}.csv" \
      2>&1 | tee "$OUT_DIR/${lab}.log"
  done
else
  echo "Invalid MODE=$MODE (expected smoke or full)" >&2
  exit 2
fi

if [[ "$PROFILE" == 1 ]]; then
  if ! command -v ixkn-cli >/dev/null 2>&1; then
    echo 'PROFILE=1 requires ixkn-cli on the GPU machine' >&2
    exit 1
  fi
  ixkn-cli --version > "$OUT_DIR/ixkn_version.txt" 2>&1
  ixkn-cli --query-metrics > "$OUT_DIR/ixkn_supported_metrics.txt" 2>&1
  METRICS=${METRICS:-time_duration,llc_hit_rate,active_warps_per_active_cycle,mem_load_throughput}
  # Profiling is separated from benchmark timing. Graph mode deliberately not
  # captured by default (graph visibility depends on profiler/runtime support).
  for spec in 'L01 launches_4 tiny_launch_kernel' \
              'L02 fused fusion_combined' \
              'L02 unfused fusion_stage_a' \
              'L04 grid grid_work_kernel' \
              'L04 persistent_64 persistent_work_kernel'; do
    read -r lab variant kernel <<< "$spec"
    echo "== profiler $lab $variant ($kernel) =="
    ixkn-cli --profile-from-start on --page details --csv --cache-control none \
      --kernel-name "$kernel" --metrics "$METRICS" \
      python3 triton_execution_lab.py --lab "$lab" --variant "$variant" \
        --single-run --strict --out "$OUT_DIR/${lab}_${variant}.capture.csv" \
      2>&1 | tee "$OUT_DIR/${lab}_${variant}.counters.log"
    grep -q "^PASS ${lab} ${variant}:" "$OUT_DIR/${lab}_${variant}.counters.log"
  done
  echo 'Check actual numeric counters, kernel names, and capture coverage manually.'
fi

echo "Group9 results: $OUT_DIR"
