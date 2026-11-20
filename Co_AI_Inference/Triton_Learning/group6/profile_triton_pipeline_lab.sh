#!/usr/bin/env bash
# Group6: standalone Triton software-pipeline experiments.
# Do not run the official 4K/16K competition benchmark through this script.
set -euo pipefail
cd "$(dirname "$0")"

MODE=${MODE:-full}                # smoke | full
PROFILE=${PROFILE:-0}            # optional ixkn-cli hardware profiling
OUT_DIR=${OUT_DIR:-"results/run_$(date -u +%Y%m%d_%H%M%S)"}
mkdir -p "$OUT_DIR"

python3 test_pipeline_math.py 2>&1 | tee "$OUT_DIR/cpu_tests.log"

if [[ "$MODE" == smoke ]]; then
  echo '== Group6 GPU smoke tests: compile, launch and verify ==' 
  python3 triton_pipeline_lab.py --lab P01 --elements 4099 --steps 5 \
    --block 128 --work 2 --single-run --strict \
    --out "$OUT_DIR/P01_smoke.csv" 2>&1 | tee "$OUT_DIR/P01_smoke.log"
  python3 triton_pipeline_lab.py --lab P02 --elements 4099 --steps 6 \
    --block 128 --work 2 --single-run --strict \
    --out "$OUT_DIR/P02_smoke.csv" 2>&1 | tee "$OUT_DIR/P02_smoke.log"
  python3 triton_pipeline_lab.py --lab P03 --m 65 --n 97 --k 96 \
    --bm 32 --bn 64 --bk 32 --stages 1,2 --single-run --strict \
    --out "$OUT_DIR/P03_smoke.csv" 2>&1 | tee "$OUT_DIR/P03_smoke.log"
  python3 triton_pipeline_lab.py --lab P04 --m 64 --n 64 --k-sizes 64,128 \
    --bm 32 --bn 64 --bk 32 --stages 1,2 --single-run --strict \
    --out "$OUT_DIR/P04_smoke.csv" 2>&1 | tee "$OUT_DIR/P04_smoke.log"
elif [[ "$MODE" == full ]]; then
  echo '== Group6 standalone A/B performance tests ==' 
  for lab in P01 P02 P03 P04; do
    echo "--- $lab ---"
    python3 triton_pipeline_lab.py --lab "$lab" --strict \
      --dump-asm "$OUT_DIR/asm" --out "$OUT_DIR/${lab}.csv" \
      2>&1 | tee "$OUT_DIR/${lab}.log"
  done
else
  echo "Unsupported MODE=$MODE (expected smoke or full)" >&2
  exit 2
fi

if [[ "$PROFILE" == 1 ]]; then
  if ! command -v ixkn-cli >/dev/null 2>&1; then
    echo 'PROFILE=1 requires ixkn-cli on the target device' >&2
    exit 1
  fi
  ixkn-cli --version > "$OUT_DIR/ixkn_version.txt" 2>&1
  ixkn-cli --query-metrics > "$OUT_DIR/ixkn_available_metrics.txt" 2>&1
  # Verify supported metric names in ixkn_available_metrics.txt.
  # Override METRICS according to the installed ixkn-cli build if needed.
  METRICS=${METRICS:-time_duration,issue_stall_reason_memory_dependency,active_warps_per_active_cycle,mem_load_throughput}
  echo '== Optional BI-V150 hardware counter probes =='
  for spec in \
    'P01 serial load_compute_order_kernel' \
    'P01 early load_compute_order_kernel' \
    'P02 d0 prefetch_distance_kernel' \
    'P02 d4 prefetch_distance_kernel' \
    'P03 k1024_s1 gemm_pipeline_kernel' \
    'P03 k1024_s3 gemm_pipeline_kernel'; do
    read -r lab variant kernel <<< "$spec"
    label="${lab}_${variant}"
    if [[ "$lab" == P03 ]]; then
      extra=(--stages 1,3)
    else
      extra=()
    fi
    ixkn-cli --profile-from-start off --page details --csv --cache-control none \
      --kernel-name "$kernel" --metrics "$METRICS" \
      python3 triton_pipeline_lab.py --lab "$lab" --variant "$variant" \
      --single-run --strict "${extra[@]}" \
      --out "$OUT_DIR/${label}.native.csv" \
      2>&1 | tee "$OUT_DIR/${label}.counters.log"
    grep -q '^PASS:' "$OUT_DIR/${label}.counters.log" || {
      echo "Missing success marker for $label" >&2
      exit 1
    }
  done
  echo 'Counter captures may be empty or unsupported: examine captured numeric rows and matching kernel names.'
fi

echo "Group6 results: $OUT_DIR"
echo 'Remember: no software-pipeline overlap is established without generated code or hardware counters.'
