#!/usr/bin/env bash

set -euo pipefail

cd /root/Optimization/v17/memory_profile

export PYTHONPATH="/root/Optimization/v17/FlagGems/src:/root/Optimization/v17/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl

unset VLLM_FL_USE_FLAGGEMS_ATTN
unset VLLM_ATTENTION_BACKEND
unset VLLM_BATCH_INVARIANT

OUT="/root/Optimization/v17_memory_counters_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"

# Hardware metrics supported by BI-V150.
METRICS=(
    time_duration
    llc_hit_rate
    llc_mem_req
    l1vk_ld_hit_rate
    l1vk_st_hit_rate
    l1vk_utilization
    mem_load_requests
    mem_store_requests
    mem_load_transactions
    mem_store_transactions
    mem_load_transactions_per_request
    mem_store_transactions_per_request
    mem_load_throughput
    mem_store_throughput
    shared_load_transactions
    shared_store_transactions
    shared_load_throughput
    shared_store_throughput
    load_store_function_unit_util
    issue_stall_reason_memory_dependency
    issue_stall_reason_memory_throttle
    issue_stall_reason_pipeline_busy
    issue_stall_reason_execution_dependency
    issue_stall_reason_synchronization
    warp_execution_efficiency_ml_mem
    warp_execution_efficiency_ml_alu
    active_warps_per_active_cycle
    stalled_warps_percentage
    matrix_inst
    pcie_total_data_received
    pcie_total_data_transmitted
)

METRIC_LIST=$(IFS=,; echo "${METRICS[*]}")

# Record the profiler environment.
ixkn-cli --version > "$OUT/version.txt"
ixkn-cli --query-metrics > "$OUT/available_metrics.txt"
ixkn-cli --list-sections > "$OUT/sections.txt"

profile_case() {
    local CASE="$1"
    local LAYOUT="${2:-shuffled}"
    local KERNEL

    case "$CASE" in
        mm_*)
            KERNEL="mm_kernel"
            ;;
        linear_*)
            KERNEL="linear_kernel"
            ;;
        *_3d)
            KERNEL="kernel_unified_attention|reduce_segments"
            ;;
        attn_*)
            KERNEL="kernel_unified_attention"
            ;;
        *)
            echo "Unknown case: $CASE"
            return 1
            ;;
    esac

    local NAME="${CASE}_${LAYOUT}"

    echo "========================================"
    echo "Profiling: $NAME"
    echo "Kernel: $KERNEL"
    echo "========================================"

    # Non-profiled timing.
    python3 probe.py "$CASE" \
        --layout "$LAYOUT" \
        --warmup 3 \
        --repeats 10 \
        2>&1 | tee "$OUT/${NAME}.timing.log"

    # Native hardware counters.
    ixkn-cli \
        --profile-from-start off \
        --page details \
        --csv \
        --cache-control none \
        --kernel-name "$KERNEL" \
        --metrics "$METRIC_LIST" \
        python3 probe.py "$CASE" \
            --layout "$LAYOUT" \
            --warmup 3 \
            --native \
        2>&1 | tee "$OUT/${NAME}.counters.csv"

    if ! grep -q 'PROBE_JSON ' "$OUT/${NAME}.counters.csv"; then
        echo "WARNING: Missing probe completion marker."
        return 1
    fi
}

# Default: two smoke-test cases.
if [ "$#" -eq 0 ]; then
    set -- mm_qkv attn_4k_decode
fi

for CASE in "$@"; do
    profile_case "$CASE" shuffled
done

echo "========================================"
echo "Results: $OUT"
echo "Inspect actual hardware counter rows."
echo "========================================"
