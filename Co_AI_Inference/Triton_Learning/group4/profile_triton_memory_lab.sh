#!/usr/bin/env bash
# Group4 standalone Triton A/B lab; never touches vLLM or official benchmarks.
set -euo pipefail
cd "$(dirname "$0")"

MODE=${MODE:-full}       # smoke or full
OUT_DIR=${OUT_DIR:-"results/run_$(date +%Y%m%d_%H%M%S)"}
PROFILE=${PROFILE:-0}    # 1 enables optional ixkn-cli hardware-counter probes
mkdir -p "$OUT_DIR"

case "$MODE" in
  smoke)
    ARGS=(--lab all --cache-n 4096 --cache-reps 8
          --cache-sizes-mib 0.25,1 --cache-size-mib 1
          --chase-n 512 --pool-entries 4096 --total-loads 16
          --chains 1,2,4 --locality-strides 1,17 --blocks 128,256
          --block 128 --trials 1 --warmup-ms 5 --rep-ms 15 --strict)
    ;;
  full)
    ARGS=(--lab all --strict)
    ;;
  *) echo "Invalid MODE=$MODE (choose smoke or full)" >&2; exit 2 ;;
esac

printf 'Output directory: %s\n' "$OUT_DIR"
python3 triton_memory_latency_lab.py "${ARGS[@]}" \
  --dump-asm "$OUT_DIR/asm" --out "$OUT_DIR/memory_latency.csv" \
  2>&1 | tee "$OUT_DIR/bench.log"

if [[ "$PROFILE" == 1 ]]; then
  if ! command -v ixkn-cli >/dev/null 2>&1; then
    echo 'PROFILE=1 requires ixkn-cli on this device' >&2
    exit 1
  fi
  ixkn-cli --version > "$OUT_DIR/ixkn_version.txt" 2>&1
  ixkn-cli --query-metrics > "$OUT_DIR/ixkn_available_metrics.txt" 2>&1
  METRICS=${METRICS:-time_duration,llc_hit_rate,mem_load_transactions_per_request,mem_load_throughput,issue_stall_reason_memory_dependency,active_warps_per_active_cycle}
  echo 'Profiling M01 (small vs large), M02 (1 vs 4 chains), M04 (contiguous vs strided).'
  echo 'Review which counters are actually supported and populated on BI-V150.'
  for spec in \
    'M01 0' 'M01 1' \
    'M02 0' 'M02 1' \
    'M04 0' 'M04 1'; do
    read -r LAB INDEX <<< "$spec"
    case "$LAB" in
      M01) EXTRA=(--cache-sizes-mib 0.25,64 --cache-n 262144 --cache-reps 64) ;;
      M02) EXTRA=(--chase-n 65536 --total-loads 128) ;;
      M04) EXTRA=(--blocks 128 --locality-strides 1,257 --cache-size-mib 64 \
                  --cache-n 262144 --cache-reps 64) ;;
    esac
    case "$LAB" in
      M01|M04) MATCH=stream_read_kernel ;;
      M02) MATCH=pointer_chase_kernel ;;
    esac
    LABEL="${LAB}_variant${INDEX}"
    ixkn-cli --profile-from-start off --page details --csv --cache-control none \
      --kernel-name "$MATCH" --metrics "$METRICS" \
      python3 triton_memory_latency_lab.py --lab "$LAB" --native \
        --native-index "$INDEX" "${EXTRA[@]}" \
        --out "$OUT_DIR/${LABEL}.native.csv" \
        2>&1 | tee "$OUT_DIR/${LABEL}.counters.log"
    grep -q "PROFILE_OK $LAB" "$OUT_DIR/${LABEL}.counters.log" || {
      echo "Missing successful kernel marker for $LABEL" >&2
      exit 1
    }
  done
fi

echo "Results: $OUT_DIR"
