# v15 ranked research queue — hypotheses, not results

## Evidence and priorities

Read [v14 telemetry](../../../v14_profile_runs/20261115_161906_1085_telemetry/summary.md), [v14 trace](../../../v14_profile_runs/20261115_172508_8779_trace/summary.md), [v7 findings](../../../profiling/v7/v7_gemm_optimization_findings.md), [v7 shape inventory](../../../profiling/v7/v7_observed_gemm_shapes.csv), and [v14 report](../../../../submission/v14/report.md). Stage 2 archived instrumented sources alone are not measured results.

- 4K early worker: 47.0% GEMM/Linear, 45.1% attention, 0.5% KV ops; mm_kernel 4715.10 ms of 10115.62 ms summed observed kernel time. 4K middle 38.6% GEMM but graph visibility incomplete; late graph windows are not representative. 16K early/middle GEMM 23.3%/19.2%, attention 73.0%/77.4%. Neither shares nor queue time imply end-to-end speedup. A 20% reduction of all GEMM time in the early 4K window would save 9.4% of its summed kernel time, not 9.4% whole-run time.
- 16K telemetry peak KV usage 99.86%, trace peak 99.98%; maximum 31 running/63 waiting, four new preemptions in each 16K run; 4K peak 62.08%, maximum 64 running, zero new preemptions. Sampled peak device memory ~28,432 MiB of 32,768 MiB. This confirms pressure, not avoidable queueing or available free allocation headroom. KV write kernels only 0.2–0.3% of observed 16K early/middle time.
- v7 grouped MM had 7.43% less summed median time for four M=2048 BF16 shapes, wide projection 17.79% lower latency but one projection regressed 5.91%; no demonstrated serving improvement. v15 vendor mm.py currently uses row-major program IDs despite GROUP_M=8 argument, and linear.py already has grouped mapping. mm.py already passes accumulator in its fp8_fast_accum=True branch (name is historical; BF16 path must retain actual dtype). Do not blindly transplant all v7/v8 changes.

## G1 — grouped MM scheduling, first cheap experiment

Actual path: FlagGems/src/flag_gems/runtime/backend/_iluvatar/ops/mm.py; existing libtuner config under _iluvatar/tune_configs.yaml. Inspect worker attribution before modifying. Reintroduce correct grouped program-id mapping inside existing operator, compare group 1/2/4/8 or resource/shape-derived policy, preserve row-major for small-M where justified. Handle partial groups with local pid modulo group width; preserve index width, strides, masks, split-K, dtypes and tuning/dispatch. Measure all projection families, weighted by production call count, not best shape. Reject if wide improvement is offset by other hot shapes. No hardcoded MiniCPM workload or device name predicates for new optimizations when general shape/resource condition works.

Historical BF16 (M,K,N) at M=2048 and small M=1..256:
- (M,2048,12288), (M,2048,2560), (M,2048,2048), (M,6144,2048).
- A row-major, B strides (1,K). Linear (M,2048,130560), M=1/256; verify actual weight strides and bias.
- Recollect actual v15 M and calls before choosing weighting. Test nonaligned M/N/K, alternate strides, FP16/BF16, mixed dtypes where supported, mm.out, bias, 1D/3D linear, split-K initialization/reduction and large-index logic. Never activate unsafe split-K or add bias multiple times.

## G2 — tile/dataflow and linear improvements

Inspect emitted Triton/ISA and supported profiler metrics: tile utilization, registers/spills, shared memory, L2, memory transactions and tensor-core activity. Explore K-loop pipeline, load/vectorization, tail masks, accumulator and epilogue fusion, small-M tiling, safe FP32 split-K reduction where useful, persistent tiles only if supported. linear.py uses acc += tl.dot; compare direct accumulation with FP32 and original TF32 behavior preserved. Include copies, zero/init, scratch, reduction and allocation costs; check graph replay. Only operator-internal changes; no dispatch-only swap or simply altering serving flags. Isolate and measure each candidate. Prefer no new persistent scratch; if one needs memory, quantify impact on 16K cache.

## K1 — memory accounting and real non-KV savings, highest KV priority

Start at vllm-plugin-FL/vllm_fl/worker/worker.py (MemorySnapshot, memory_profiling_fl, WorkerFL.determine_available_memory), worker/model_runner.py (persistent buffers, KV initialization, graph capture), compilation/graph.py and actual allocator/API paths. Scheduler_fl.py extends AsyncScheduler for structured output, not a proven KV allocator. Read installed upstream allocator/scheduler for diagnosis only. Record exact layer count, KV heads/head dim, dtype bytes, block size, cache groups/specs, tokens/blocks; derive bytes per logical token including layout and padding. Reconcile memory cap with weights, activations, static buffers, allocator reserved vs allocated, graph pools, non-torch memory, KV and peak safety headroom. Separate reclaimable temporary allocations from live buffers. Record when profiling computes available cache memory relative to graph capture; do not simply increase returned bytes or subtract hypothetical savings.

Try lifetime shortening, eliminate duplicate buffers/copies, safe workspace reuse, memory-efficient operator fusion, and graph-safe allocation reuse where real savings exist. Only count additional blocks when actual allocation changes at identical flags and normal memory accounting. Verify address stability and alias/lifetime safety across asynchronous streams and graph replay; stress maximum supported contexts within allocated capacity, mixed batches, repeated startup/replay, no OOM or token loss. Report actual block delta, not theoretical free VRAM. If no real reclaimable memory exists, stop this line rather than overriding safety.

## K2 — block utilization and recomputation accounting

Instrument allocation/free/refcount, prefix-cache hit/eviction, live tokens, internal fragmentation, preemption/recompute token count and per-phase wall-clock. Distinguish active references from cached/free blocks. Estimate useful capacity bound given bytes per token and measured live context. If avoidable duplicate allocations or copies exist, implement a general exact fix inside v15-owned storage/operator path. Prove identical logical KV for shuffled pages, tail, shared prefixes, cancellation, preemption/resume, interleaved requests, and cache isolation; check block-table/slot mapping and graph capture. No change to block size, admission policy, cache dtype, prefix-cache policy or token retention. Pure scheduler/allocator policy experiments with ambiguous eligibility are diagnostic-only until organizer approval. No direct merge of upstream code.

## K3 — KV write/read efficiency only if evidence supports it

Inspect existing reshape_and_cache and reshape_and_cache_flash implementations and invocation. Prior v6 write tiling did not establish serving gain. Optimize real duplicate memory transactions or fused workspace elimination only after showing impact on capacity or critical-path time. A faster write kernel does not create more physical KV slots. Preserve v14 dense attention and fallback, all cache values and dtype.

## Experiment order

First cycle: baseline cheap shape/memory inventory; G1 targeted A/B; K1 memory attribution and one justified candidate. Then G2/K2 based on measured potential. Use serial GPU tests, independently reversible diffs and short serving screens. Re-rank using frequency-weighted kernel savings, measured capacity/recompute savings, actual serving impact and risk. Test stack only after individual contributions are established. Stop spending on hypotheses without measurable causal evidence; preserve negative results.
