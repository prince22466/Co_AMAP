#!/usr/bin/env bash
# Group7 standalone GEMM experiments. Never changes competition benchmark scripts.
set -euo pipefail
cd "$(dirname "$0")"

MODE=${MODE:-full}       # smoke | full
PROFILE=${PROFILE:-0}   # optional ixkn-cli captures of individual variants
OUT_DIR=${OUT_DIR:-"results/run_$(date -u +%Y%m%d_%H%M%S)"}
mkdir -p "$OUT_DIR"
python3 test_gemm_lab_math.py 2>&1 | tee "$OUT_DIR/cpu_tests.log"

if [[ "$MODE" == smoke ]]; then
  echo '== GPU smoke: nonmultiple sizes validate all tail masks =='
  for lab in G01 G02 G03 G04; do
    echo "-- $lab --"
    python3 triton_advanced_gemm_lab.py --lab "$lab" \
      --m 65 --n 129 --k 97 \
      --tiles 32x64x32,64x64x32 \
      --bm 32 --bn 64 --bk 32 \
      --groups 1,2,8 --splits 2,4 --workers 2,4 \
      --single-run --strict --out "$OUT_DIR/${lab}_smoke.csv" \
      2>&1 | tee "$OUT_DIR/${lab}_smoke.log"
  done
elif [[ "$MODE" == full ]]; then
  echo '== Full Group7 GPU experiments =='
  for lab in G01 G02 G03 G04; do
    echo "-- $lab --"
    python3 triton_advanced_gemm_lab.py --lab "$lab" \
      --dump-asm "$OUT_DIR/asm" --strict --out "$OUT_DIR/${lab}.csv" \
      2>&1 | tee "$OUT_DIR/${lab}.log"
  done
else
  echo "Invalid MODE=$MODE (expected smoke or full)" >&2
  exit 2
fi

if [[ "$PROFILE" == 1 ]]; then
  if ! command -v ixkn-cli >/dev/null 2>&1; then
    echo 'PROFILE=1 requires ixkn-cli' >&2
    exit 1
  fi
  # Do not compare profiler-instrumented times with unprofiled GPU event timings.
  ixkn-cli --version > "$OUT_DIR/ixkn_version.txt" 2>&1
  ixkn-cli --query-metrics > "$OUT_DIR/ixkn_supported_metrics.txt" 2>&1
  METRICS=${METRICS:-time_duration,llc_hit_rate,active_warps_per_active_cycle,mem_load_throughput}
  echo '== Optional kernel-specific BI-V150 counter capture =='
  for spec in 'G02 group_m_1 gemm_grid_kernel' \
              'G02 group_m_8 gemm_grid_kernel' \
              'G03 split_4 gemm_split_k_kernel' \
              'G04 persistent_32 gemm_persistent_kernel'; do
    read -r lab variant kernel <<< "$spec"
    ixkn-cli --profile-from-start on --page details --csv --cache-control none \
      --kernel-name "$kernel" --metrics "$METRICS" \
      python3 triton_advanced_gemm_lab.py --lab "$lab" --variant "$variant" \
        --single-run --strict --out "$OUT_DIR/${lab}_${variant}.capture.csv" \
      2>&1 | tee "$OUT_DIR/${lab}_${variant}.counters.log"
    # A PASS marker verifies completion only, not counter validity or attribution.
    grep -q "^PASS ${lab} ${variant}:" "$OUT_DIR/${lab}_${variant}.counters.log"
  done
  echo 'Inspect numeric counter rows, kernel names and supported metric list manually.'
fi

echo "Group7 results: $OUT_DIR"
