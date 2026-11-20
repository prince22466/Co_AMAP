#!/usr/bin/env bash
# Groups 12–14: unified standalone Triton kernel experiments.
# No inference-server or competition benchmark parameters are changed.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-${1:-smoke}}"       # smoke | full
PROFILE="${PROFILE:-0}"          # 1: optional ixkn-cli isolated captures
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$"
OUT_DIR="${OUT_DIR:-$ROOT/results_groups_12_14/run_$RUN_ID}"
mkdir -p "$OUT_DIR"
case "$MODE" in smoke|full) ;; *) echo "Usage: $0 [smoke|full]" >&2; exit 2 ;; esac

fail=0
execute() {
  local label="$1"
  shift
  echo "== $label =="
  if "$@" 2>&1 | tee "$OUT_DIR/$label.log"; then
    echo "PASS $label"
  else
    echo "FAIL $label (see $OUT_DIR/$label.log)" >&2
    fail=1
  fi
}

execute cpu_math python3 "$ROOT/test_groups_12_14_math.py"

if [[ "$MODE" == smoke ]]; then
  execute group12 python3 "$ROOT/group12/triton_paged_attention_lab.py" \
    --lab all --contexts 257,4096 --single-run \
    --out "$OUT_DIR/group12.csv"
  execute group13 python3 "$ROOT/group13/triton_shared_async_lab.py" \
    --lab all --n 4096 --matrix-size 128 --tile 32 --single-run \
    --out "$OUT_DIR/group13.csv"
  execute group14 python3 "$ROOT/group14/triton_atomic_sync_lab.py" \
    --lab all --n 4096 --bins 8 --hot-bins 1,8 --single-run \
    --out "$OUT_DIR/group14.csv"
else
  execute group12 python3 "$ROOT/group12/triton_paged_attention_lab.py" \
    --lab all --contexts "${CONTEXTS:-4096,16384}" \
    --dump-asm "$OUT_DIR/asm/group12" --out "$OUT_DIR/group12.csv"
  execute group13 python3 "$ROOT/group13/triton_shared_async_lab.py" \
    --lab all --dump-asm "$OUT_DIR/asm/group13" --out "$OUT_DIR/group13.csv"
  execute group14 python3 "$ROOT/group14/triton_atomic_sync_lab.py" \
    --lab all --dump-asm "$OUT_DIR/asm/group14" --out "$OUT_DIR/group14.csv"
fi

if [[ "$PROFILE" == 1 ]]; then
  if ! command -v ixkn-cli >/dev/null 2>&1; then
    echo "PROFILE=1 requires ixkn-cli; captures unavailable" >&2
    fail=1
  else
    ixkn-cli --version > "$OUT_DIR/ixkn_version.txt" 2>&1 || fail=1
    ixkn-cli --query-metrics > "$OUT_DIR/ixkn_metrics.txt" 2>&1 || fail=1
    METRICS="${METRICS:-time_duration}"
    capture() {
      local label="$1" kernel="$2"
      shift 2
      echo "== Native capture: $label ($kernel) =="
      if ixkn-cli --profile-from-start on --page details --csv \
          --cache-control none --kernel-name "$kernel" --metrics "$METRICS" \
          "$@" > "$OUT_DIR/${label}.counters.log" 2>&1; then
        if ! grep -q 'status=ok' "$OUT_DIR/${label}.counters.log"; then
          echo "Missing correctness marker in $label capture" >&2
          fail=1
        fi
      else
        echo "Failed native capture $label" >&2
        fail=1
      fi
    }
    capture A01_page16 paged_online_decode \
      python3 "$ROOT/group12/triton_paged_attention_lab.py" \
      --lab A01 --variant page16 --contexts 4096 --single-run \
      --out "$OUT_DIR/profile_A01.csv"
    capture H00_tile32 transpose_tiled \
      python3 "$ROOT/group13/triton_shared_async_lab.py" \
      --lab H00 --variant tile32 --single-run \
      --out "$OUT_DIR/profile_H00.csv"
    capture B02_cta grouped_cta_atomic \
      python3 "$ROOT/group14/triton_atomic_sync_lab.py" \
      --lab B02 --variant cta --single-run \
      --out "$OUT_DIR/profile_B02.csv"
    echo "Inspect kernel attribution AND populated numeric counters manually."
    echo "Profiler-instrumented durations must not be compared with event timing."
  fi
fi

echo "Results: $OUT_DIR"
exit "$fail"
