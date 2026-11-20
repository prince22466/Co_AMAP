#!/usr/bin/env bash
# Run separately from vLLM and from ordinary timing. No benchmark changes.
set -euo pipefail
cd "$(dirname "$0")"
OUT=$(mktemp -d "triton_data_profile_$(date +%Y%m%d_%H%M%S)_XXXXXX")
METRICS=${METRICS:-time_duration,llc_hit_rate,mem_load_transactions_per_request,mem_load_throughput,mem_store_throughput,issue_stall_reason_memory_dependency,issue_stall_reason_memory_throttle,active_warps_per_active_cycle,matrix_inst}
ixkn-cli --version > "$OUT/version.txt"
ixkn-cli --query-metrics > "$OUT/available_metrics.txt"
if [[ $# -eq 0 ]]; then set -- D02 D03 D07; fi
for lab in "$@"; do
    case "$lab" in
        D01|D02) kernel=copy_kernel ;;
        D03) kernel='transpose_naive|transpose_tiled' ;;
        D04|D06) kernel=gemm_kernel ;;
        D05) kernel='silu_kernel|mul_kernel|fused_kernel' ;;
        D07) kernel='kv_tokenwise|kv_pagewise' ;;
        D08) kernel='attention_scores|attention_probs|attention_weighted|attention_online|attention_merge' ;;
        *) echo "Unknown lab: $lab" >&2; exit 2 ;;
    esac
    python3 triton_data_lab.py --lab "$lab" --csv "$OUT/$lab.timing.csv" \
        2>&1 | tee "$OUT/$lab.timing.log"
    for variant in vanilla optimized; do
        name="${lab}_${variant}"
        ixkn-cli --profile-from-start off --page details --csv --cache-control none \
            --kernel-name "$kernel" --metrics "$METRICS" \
            python3 triton_data_lab.py --lab "$lab" --variant "$variant" --native \
            2>&1 | tee "$OUT/$name.counters.csv"
        grep -q "PROFILE_OK $lab $variant" "$OUT/$name.counters.csv" || {
            echo "Missing capture completion marker: $name" >&2; exit 1;
        }
    done
done
echo "Results: $OUT. Check actual numeric counter rows and kernel coverage, not just markers."
