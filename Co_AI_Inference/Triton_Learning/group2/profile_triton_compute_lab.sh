#!/usr/bin/env bash
# BI-V150 native compute-focused profiling. Run only on an otherwise idle GPU.
# No competition model, configuration, dispatch or benchmark files are modified.
set -euo pipefail
cd "$(dirname "$0")"
command -v ixkn-cli >/dev/null || { echo 'ixkn-cli is required' >&2; exit 1; }
OUT=$(mktemp -d "triton_compute_profile_$(date +%Y%m%d_%H%M%S)_XXXXXX")
# Edit METRICS to use exact locally supported names if the installed ixKN differs.
METRICS=${METRICS:-time_duration,matrix_inst,load_store_function_unit_util,issue_stall_reason_pipeline_busy,issue_stall_reason_execution_dependency,active_warps_per_active_cycle,stalled_warps_percentage}
ixkn-cli --version > "$OUT/version.txt"
ixkn-cli --query-metrics > "$OUT/available_metrics.txt"
ixkn-cli --list-sections > "$OUT/sections.txt"
if [ "$#" -eq 0 ]; then set -- C01 C02 C03 C04 C05 C06; fi
for lab in "$@"; do
  case "$lab" in
    C01|C04|C05) kernel=fma_chain_kernel ;;
    C02) kernel=ilp_kernel ;;
    C03) kernel=polynomial_kernel ;;
    C06) kernel=gemm_acc_kernel ;;
    *) echo "Unknown compute lab: $lab" >&2; exit 2 ;;
  esac
  echo "== $lab: ordinary timing =="
  python3 triton_compute_lab.py --lab "$lab" --csv "$OUT/$lab.timing.csv" \
    2>&1 | tee "$OUT/$lab.timing.log"
  for variant in vanilla candidate; do
    echo "== $lab / $variant: native counters =="
    ixkn-cli --profile-from-start off --page details --csv \
      --cache-control none --kernel-name "$kernel" --metrics "$METRICS" \
      python3 triton_compute_lab.py --lab "$lab" --variant "$variant" --native \
        2>&1 | tee "$OUT/${lab}_${variant}.counters.csv"
    grep -q "PROFILE_OK $lab $variant" "$OUT/${lab}_${variant}.counters.csv" || {
      echo "Native profiler did not reach capture completion: $lab/$variant" >&2
      exit 1
    }
  done
done
echo "Output: $OUT"
echo "Inspect actual numeric hardware counter rows; PROFILE_OK is only a capture marker."
