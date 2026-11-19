# v17 — data-movement profiling (diagnostic only)

Run on the existing BI-V150 machine, with exclusive access to the GPU and no
other vLLM server running. No production kernels, dispatch logic, model settings
or benchmark scripts are changed. The serving run keeps the v14 startup
configuration and uses the existing **diagnostic half-run** script, not an
official score run. Profiling instrumentation and profiler replay perturb timing.

## Run

From the deployed v17 directory (the default model and port match the previous
v14 runs):

    cd /root/Optimization/v17/memory_profile
    python3 -m unittest -v test_analysis
    python3 run.py --stage all

If the half-run script is not at the known repository or workspace location:

    python3 run.py --stage all --benchmark /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py

For a shorter first pass, select cases and the 4K serving workload:

    python3 run.py --stage all --cases mm_qkv mm_gate_up attn_4k_decode attn_4k_mixed --workloads 4k

Separate runs (each creates a new output directory):

    python3 run.py --stage native --compare-layouts
    python3 run.py --stage serve
    python3 run.py --stage analyze --out /path/to/v17_memory_run

Use --out for an explicitly chosen **new** output directory outside v17. Keep raw logs.
An incomplete native collection, missing serving trace, or missing boundary
metrics returns a nonzero status after preserving results; inspect status.jsonl.
The serving driver also validates both raw benchmark rows per workload, because
the unchanged half-run benchmark can exit zero even on failed requests.
Raw benchmark CSVs are kept separately under 4k/ and 16k/.
Do not run on an occupied GPU or in parallel with another profiling job.

## What is collected

| Target | Evidence | Files |
| --- | --- | --- |
| GEMM reads/reuse | Four observed v7 prefill shapes plus a small decode linear proxy; BF16, transposed weight; exact tensor payload/FLOPs, event timing, native counters | native/*.timing.log, native/*.counters.csv |
| Attention reads | v17 unified attention, 4K/16K, prefill/decode/mixed, optional contiguous vs shuffled pages; 2D and one segmented case | native/*attn* |
| Intermediate materialization | CPU operator shapes, GPU kernel names, allocator events and device copies; inspect compiled fusion names and tensor lifetimes | operator_shapes.csv, trace_events.csv, trace_overview.csv, traces/ |
| Mixed execution | Sampled scheduled query lengths and prompt progress; classify actual prefill/decode phase plus query-length path; padding proxy; isolated mixed attention counters | batch_metadata.jsonl, batch_steps.csv, batch_summary.csv |
| CPU–GPU dependencies | Memcpy/memset events, synchronization and graph-launch host API scopes, observed kernel interval union | trace_events.csv, trace_overview.csv, windows.jsonl |
| KV capacity | Per-series Prometheus usage, waiting, preemptions, token and prefix cache counters; active logical page tail slack proxy and scheduled replay attribution | metrics.jsonl, capacity.csv, batch_steps.csv, server.log |

## Native counters

The installed ixKN version determines supported metric names. The script saves
--help, --list-sections, --query-metrics and --list-metrics before profiling.
By default it requests all available sections for the selected warmed kernel
region; this can take time. Use --native-sections with an identifier from the
local list, or --native-metrics with comma-separated exact names from the
local query. See https://developer.iluvatar.com/doc-booklets/5.0.0/booklets/ixkn/page_02.html.

The offline analyzer recognizes long-form and supported wide raw CSV headers.
It writes native_metrics.csv (original metric names, values and units),
native_report_coverage.csv and probe_payloads.csv. Unknown formats remain
unverified; a recognized numeric memory-related metric is only coverage,
not validation of HBM bytes. Missing intended kernel coverage or a missing
probe completion marker produces an incomplete status. Inspect raw rows for device memory read/write bytes (or transactions with
documented byte size), L2 hit/miss rate, memory throughput, memory dependency
stalls, occupancy and matrix utilization, where supported. A successful exit
without metric rows is not proof of measurement. Never use ixsmi GPU utilization
as HBM bandwidth. Unsupported counters are unknown, not zero. Native replay
timings are not representative of serving; use the separate timing logs.
Warm timing includes allocation/launch behavior of the existing wrappers.
The isolated probes are shape proxies, not proof of production dispatch;
confirm kernel names in serving traces. Segmented attention requests buffers
but the launcher retains its own gating.

## Serving instrumentation

The script creates a private copy of vllm-plugin-FL under the output folder and
inserts a host-only observation after the existing block-table commit. Every
tenth preparation step by default records scheduled, already-computed and
prompt lengths without copying a GPU tensor or synchronizing. All observed
resumed/repeated-position steps are also emitted. Request IDs remain in an
in-memory high-water map only; the output contains no IDs, prompts, token values
or model weights. Use --no-metadata
for the original plugin tree, or --metadata-every to change diagnostic sampling.
The instrumentation diff and file hashes are saved. The observer deliberately
does not alter scheduling decisions or launch parameters.

Three short torch profiler windows per workload are triggered after a positive
running-request metric, separated by --window-gap. These are chronological
windows, not guaranteed early/middle/late semantic phases. The original decode
graph mode is retained. Trace graph launches without internal kernels mean
visibility is incomplete: never call those gaps idle time. CPU scope sums
overlap. Allocator events are allocation traffic, not physical HBM bytes.
Shape recording can extend tensor lifetime and memory profiling can perturb
capacity. Compare profiler-off telemetry before treating any result as causal.
Each workload has separate before/after Prometheus snapshots; exact counter
deltas require both boundaries and no observed reset. Sampled deltas are
reported separately. Trace CSVs are labeled by workload and capture window.

Both prompt-progress phase and query-length classification are recorded; a
single-token prefill can look like decode to the attention launch gate.
Batch means are sampled/event-enriched, not unbiased workload averages. Padding assumes the target GQA 8-position prefill or 2-position
decode query block; actual active lanes also depend on backend, masks and graph
padding. Active tail slack assumes 16-token pages and is not global
fragmentation, prefix sharing, or actual allocated/free page counts. The observer tracks prior scheduled-position high-water marks and resumed
requests; resumed_replay_scheduled_tokens counts overlapping scheduled positions
until that mark is reached. This quantifies planned replay, not verified GPU
completion or measured recompute time; it assumes the ordinary non-speculative
path and starts with an empty history. Logical KV payload
and idealized query-block payload are not actual HBM bytes.

## Review before optimization

1. Confirm all relevant status entries and actual native counter rows; inspect
   server.log for the expected backend, dtype and cache size.
2. Compare GEMM compulsory bytes vs measured DRAM bytes; compute FLOPs per
   measured byte only with compatible counters and scope. Inspect L2 reuse.
3. Compare attention at the same query and context lengths; consider cache
   layout sensitivity and mixed-batch padding.
4. Locate repeated large intermediates and copy/wait dependencies in traces;
   verify actual fusion and graph visibility before attributing overhead.
5. Check capacity/preemptions and scheduled replay attribution; do not infer completed recomputation from counters.
6. Only then choose a candidate; run unchanged official 4K/16K and MATH-500
   with all profiling disabled. No quantization, speculative decoding, benchmark
   modifications, operator-selection bypass or upstream branch merge.

The suite is diagnostic. No optimization or new throughput claim is made.
