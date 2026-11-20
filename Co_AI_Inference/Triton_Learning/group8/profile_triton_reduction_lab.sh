#!/usr/bin/env bash
# Group8: standalone Triton reduction/softmax lab. Never modifies competition benchmarks.
set -euo pipefail
cd "$(dirname "$0")"

MODE=${MODE:-full}       # smoke | full
PROFILE=${PROFILE:-0}   # optional ixkn-cli captures; not a replacement for event timing
OUT_DIR=${OUT_DIR:-"results/run_$(date -u +%Y%m%d_%H%M%S)_$$"}
mkdir -p "$OUT_DIR"
python3 test_reduction_math.py 2>&1 | tee "$OUT_DIR/cpu_tests.log"

if [[ "$MODE" == smoke ]]; then
  echo '== Group8 GPU smoke test: irregular widths and masked tails =='
  for lab in R01 R02 R03 R04; do
    echo "-- $lab --"
    python3 triton_reduction_lab.py --lab "$lab" --rows 17 --cols 137 \
      --warps 4 --chunks 32,64 --stream-chunks 64,128 \
      --single-run --strict --out "$OUT_DIR/${lab}_smoke.csv" \
      2>&1 | tee "$OUT_DIR/${lab}_smoke.log"
  done
elif [[ "$MODE" == full ]]; then
  echo '== Group8 timed A/B experiments =='
  for lab in R01 R02 R03 R04; do
    echo "-- $lab --"
    python3 triton_reduction_lab.py --lab "$lab" \
      --warps 4 --strict --dump-asm "$OUT_DIR/asm" \
      --out "$OUT_DIR/${lab}.csv" \
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
  # Check ixkn_supported_metrics.txt before customizing METRICS for your installation.
  METRICS=${METRICS:-time_duration}
  echo '== Optional native kernel profiler captures (correctness-only launches) =='
  for spec in \
    'R02 flat reduce_tree_kernel' \
    'R02 chunked reduce_chunked_kernel' \
    'R04 fused softmax_fused_kernel' \
    'R04 online softmax_online_kernel'; do
    read -r lab variant kernel <<< "$spec"
    args=(--lab "$lab" --variant "$variant" --single-run --strict \
          --warps 4 --chunks 64 --stream-chunks 256 \
          --out "$OUT_DIR/${lab}_${variant}_capture.csv")
    ixkn-cli --profile-from-start on --page details --csv --cache-control none \
      --kernel-name "$kernel" --metrics "$METRICS" \
      python3 triton_reduction_lab.py "${args[@]}" \
      2>&1 | tee "$OUT_DIR/${lab}_${variant}.counters.log"
    grep -q "^PASS ${lab} ${variant}:" "$OUT_DIR/${lab}_${variant}.counters.log"
  done
  echo 'Read numeric counter rows; PASS only confirms functional execution.'
fi

echo "Group8 results: $OUT_DIR"
