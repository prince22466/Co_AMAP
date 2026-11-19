# v17 decode KV one-tile lookahead (experimental)

Evidence: memory_profile/mem_profile_records/v17_memory_counters_20261119_230033.
In isolated 4K and 16K decode, ixKN reports 55.65% and 56.14% memory
dependency stalls and 1.71% and 1.55% pipeline-busy stalls. These are warp
stall-reason counters, not fractions of wall time. v5/v14 already reduce
page-table lookup redundancy. This candidate instead changes load scheduling:
prime the first K/V tile, consume its K in QK, issue the next K/V tile before
current softmax and PV, then carry that tile into the next iteration. The
original tile ordering, masks, online softmax, segmentation and output stay
unchanged. Mask lookahead at the segment/sequence end before both page-table
and data loads. All unsupported inputs retain the original path.

The change increases live tile storage, may increase registers/spills, and
Triton/BI-V150 may not realize overlap. This is a candidate, not a proven
speedup. No new user configuration, operator selection, model or benchmark
change. No changes to prefill or mixed-batch execution.

## Validation on BI-V150

Use this branch's v17 sources and exclusive GPU access:

    cd /root/Optimization/v17
    export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
    export VLLM_PLUGINS=fl
    python3 -m pytest -q -s --noconftest \
      vllm-plugin-FL/tests/functional_tests/test_biv150_decode.py \
      vllm-plugin-FL/tests/functional_tests/test_biv150_mixed_paging.py \
      vllm-plugin-FL/tests/functional_tests/test_v17_kv_pipeline.py

Tests compare against the same v17 operator with lookahead disabled only
inside the diagnostic spy, and pinned upstream. They cover shuffled physical
pages, interleaved K/V strides, separate K/V, partial tiles, short/empty
segments, exact page-table width, graph replay and sliding-window fallback.
Matched event timing covers realistic decode concurrency and context lengths.
Inspect per-case timings; do not promote a regression. The timing test is
diagnostic and never changes the official benchmark or production launch gate.

If correctness and timing support continuing, collect matched native counters
using the existing diagnostic probe, with identical seed, shape, physical
pages and K/V strides. Run from v17 with the environment above:

    cd memory_profile
    set -o pipefail
    for arm in control candidate; do
      ixkn-cli --profile-from-start off --page details --csv \
        --cache-control none --kernel-name kernel_unified_attention \
        --metrics time_duration,llc_hit_rate,llc_mem_req,issue_stall_reason_memory_dependency,issue_stall_reason_memory_throttle,active_warps_per_active_cycle,mem_load_throughput \
        python3 probe.py attn_4k_decode --interleaved-kv \
          --decode-pipeline "$arm" --warmup 3 --native \
        2>&1 | tee "decode_4k_${arm}.csv"
    done

Check the PROBE_JSON marker and actual override in each report. Repeat for
attn_16k_decode and attn_4k_decode_3d (for the latter, use the regex
'kernel_unified_attention|reduce_segments' to include both kernels).
Inspect register/shared storage and spills in a separate LaunchStats/Occupancy
capture. Counter semantics and compiler scheduling need verification. Only
then perform unchanged 4K/16K and MATH-500. Target 4K improvement with no
16K regression and permitted accuracy. No GPU tests have been run at PR creation.
