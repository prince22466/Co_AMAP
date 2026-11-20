#!/usr/bin/env bash
# Group10: A/B correctness + timing; optional ixkn-cli counters run separately.
# Does not modify FlagGems, vLLM, or any competition benchmark scripts.
set -euo pipefail
cd "$(dirname "$0")"
MODE=${MODE:-full}           # smoke | full
PROFILE=${PROFILE:-0}       # 1 enables ixkn-cli profiling (not timing)
OUT_DIR=${OUT_DIR:-"results/run_$(date -u +%Y%m%d_%H%M%S)_$$"}
mkdir -p "$OUT_DIR"

python3 test_irregular_math.py 2>&1 | tee "$OUT_DIR/cpu_tests.log"

if [[ "$MODE" == smoke ]]; then
  echo '== Group10 correctness-only GPU smoke test =='
  for lab in I01 I02 I03 I04; do
    echo "-- $lab --"
    python3 triton_irregular_lab.py --lab "$lab" \
      --n 257 --source-size 4099 --blocks 128,256 \
      --elements 127,257 --rows 17 --max-len 137 \
      --fill-rates 0.25,0.75 --single-run --strict \
      --dump-asm "$OUT_DIR/asm" --out "$OUT_DIR/${lab}_smoke.csv" \
      2>&1 | tee "$OUT_DIR/${lab}_smoke.log"
  done
elif [[ "$MODE" == full ]]; then
  echo '== Group10 timed A/B experiments =='
  for lab in I01 I02 I03 I04; do
    echo "-- $lab --"
    python3 triton_irregular_lab.py --lab "$lab" --strict \
      --dump-asm "$OUT_DIR/asm" --out "$OUT_DIR/${lab}.csv" \
      2>&1 | tee "$OUT_DIR/${lab}.log"
  done
else
  echo "Invalid MODE=$MODE (expected smoke|full)" >&2
  exit 2
fi

if [[ "$PROFILE" == 1 ]]; then
  if ! command -v ixkn-cli >/dev/null 2>&1; then
    echo 'PROFILE=1 requires ixkn-cli on the target GPU' >&2
    exit 1
  fi
  ixkn-cli --version > "$OUT_DIR/ixkn_version.txt" 2>&1
  ixkn-cli --query-metrics > "$OUT_DIR/ixkn_supported_metrics.txt" 2>&1
  METRICS=${METRICS:-time_duration,llc_hit_rate,mem_load_transactions_per_request}
  # Capture one representative configuration per variant. Do NOT use profiler
  # latencies as the main A/B timing results; profiling perturbs scheduling.
  for spec in \
    'I01 direct gather_direct_kernel' \
    'I01 sorted gather_sorted_kernel' \
    'I02 direct scatter_direct_kernel' \
    'I02 sorted scatter_sorted_kernel' \
    'I03 masked affine_masked_kernel' \
    'I03 tail_split affine_full_kernel' \
    'I04 padded ragged_padded_kernel' \
    'I04 packed ragged_packed_kernel'; do
    read -r lab variant kernel <<< "$spec"
    echo "== profiling $lab $variant =="
    ixkn-cli --profile-from-start on --page details --csv --cache-control none \
      --kernel-name "$kernel" --metrics "$METRICS" \
      python3 triton_irregular_lab.py --lab "$lab" --variant "$variant" \
        --n 65536 --source-size 262144 --blocks 128 \
        --elements 65537 --rows 256 --max-len 511 \
        --gather-patterns uniform --scatter-patterns random --fill-rates 0.25 \
        --single-run --strict --out "$OUT_DIR/${lab}_${variant}.capture.csv" \
      2>&1 | tee "$OUT_DIR/${lab}_${variant}.counters.log"
    grep -q "^PASS ${lab} ${variant}:" "$OUT_DIR/${lab}_${variant}.counters.log"
  done
  echo 'Counter capture != counter validity. Inspect numerical rows and kernel coverage.'
fi

echo "Group10 results: $OUT_DIR"
