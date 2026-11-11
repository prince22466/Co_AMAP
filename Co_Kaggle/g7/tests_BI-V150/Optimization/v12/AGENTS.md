# v12 Codex autoresearch contract — MiniCPM5-2B / BI-V150

This is the **authoritative contract**. Read `program.md` for commands and `autoresearch/RESEARCH_QUEUE.md` for candidate hypotheses. Maintain an append-only `autoresearch/EXPERIMENTS.md` and current `autoresearch/HANDOFF.md`. Work in the authorized Remote-SSH IDE; these files do not launch an unattended agent.

## Scope, rules and objective

- Edit/test **only** `/root/Optimization/v12`; verify checkout and imported modules before any test. Do not touch or import v0–v11 or modify `/workspace/vllm-plugin-FL` / `/workspace/FlagGems`. The external competition half-run script is **read-only**.
- v12 operator sources start from the optimized v10 implementation (BF16 GQA prefill `BLOCK_M=64`, decode paged-KV reuse). Do not import v11's unproven MM change. PR #237 contains a separate untested sparse prototype, **not** part of v12 main; review individual ideas rather than merge the PR.
- Preserve MiniCPM5-2B weights, official benchmark workloads/scripts, generation configuration, and fixed `vllm serve` flags. Forbidden: enabling quantization/speculation by hyperparameters; benchmark tuning; dispatch-only/operator-selection changes without an actual operator optimization; removing major dispatch/fallback logic; direct merge of latest FlagOS framework. Optimize only real attention operator computation/memory behavior.
- Historical v10 reported half-runs: **4K 2372.38**, **16K 1872.70** total tok/s. Target a **repeatable >=5% end-to-end throughput improvement** in either workload (4K >= **2490.999** OR 16K >= **1966.335**). Cross both the fixed anchor and +5% versus a **fresh, matched original-v12 control**. These are research goals, not proven scores. Always disclose both workloads and TTFT/TPOT, and assess the competition's two-scenario average score and TTFT limits; do not accept a material regression in the other case just to clear an OR target. Smaller observed improvements remain valuable research evidence, but do not meet the goal.
- Competition: MATH-500 Level-3 >=95%, no significant core-ability loss, no incomparable model output behavior. The reported v10 97.1% is **not** v12 accuracy evidence. Exact dense-output-preserving approaches are preferred. Approximate sparsity is research-only pending long-context capability checks, human-owned unchanged MATH-500 and organizer eligibility determination; **passing MATH-500 alone is insufficient**.

## Research order (re-rank after measurement)

1. **Profile** real GPU-attention critical path for 4K/16K uncached prefill, mixed prefill/decode and pure decode. Track shapes, GPU kernel time vs wall time, KV bytes, occupancy, spills, cache hits and idle gaps; estimate achievable throughput impact.
2. **Exact attention first:** improve work partition, online-softmax overhead, KV staging/page reuse or split-K/Flash-Decoding inside existing Triton attention. It already computes tiled online-softmax; renaming or switching to another backend is not an optimization.
3. **Sparse exploration in parallel when promising:** local+global, selective nonlocal KV pages, query-aware page selection or hybrid sparse decode. Skip actual KV loads and QK/PV operations; include selector overhead. Test retrieval from the **middle** of 16K context, not just the beginning/end. Compare both an explicit sparse reference and the original dense output.
4. Ring Attention only if a **real multi-GPU topology and communication bottleneck** are present; otherwise record as not applicable and focus on single-device pipelining.
5. One reversible operator hypothesis at a time. Keep independently measured exact and approximate candidates separate; preserve fallback, paging, masking, prefill/decode and CUDA-graph behavior.

## Per-candidate acceptance funnel

1. **Evidence:** document production bottleneck, proposed code change, expected Amdahl benefit, semantic/rule risk.
2. **Correctness:** BF16/FP16 where relevant; shuffled 16-token pages, query tails, nonzero context, full/mixed/decode and 2D/3D paths. For sparse, validate *both* sparse-reference correctness and drift from dense. Wrong outputs/NaNs/OOB => reject/revert candidate.
3. **Kernel A/B:** identical tensors, warmup, interleaved repeats, p50/spread, real skipped KV tiles/loads, registers/SMEM/spills; show production kernel actually executes. No kernel-only speedup counts as serving proof.
4. **Short serving A/B:** both 4K and 16K with matched server restarts, order/caching and source hashes; record completed requests, TTFT, TPOT, total tok/s. Only proceed if a plausible end-to-end improvement appears.
5. **Quality and eligibility:** for approximate methods, perform beginning/middle/end long-context retrieval tests at 4K/8K/16K (multiple seeds, distractors) and compare dense baseline. A material regression rejects candidate. Human runs unchanged MATH-500 for selected code. Organizer ruling is needed before treating behavior-changing sparsity as a submission option.
6. **Half-run gate:** use the **unchanged external** `benchmark_throughput_serve_half.py` (2 repetitions for each exact case) only after the short-serving go decision and a verified allocation deadline. Each pair needs four SUCCESS rows with all requests. Repeat matched A/B across independent server restarts before attributing a causal gain.

## Time and preservation

- Historical complete two-case half-run takes about **90–120 min**. Absolute start floor **90 min**, but the guarded estimate (105 min run +20 margin +25 preservation reserve) currently requires **>=150 min** for one run; update upward if slower. Do not invent/reset the operator-confirmed allocation deadline or launch a run that the guard rejects. Preserve at least 25 min for evidence/PR handoff.
- At session start, every <=30 min, after valuable/negative results, before a half-run and near credit/deadline pressure: append a checkpoint; push it on a non-main PR branch. The checkpoint helper **only commits evidence**, not inference-code edits: commit/push tested kernel code separately when preserving a candidate. Back up ignored raw logs/traces off-instance.
- For **verified >=5%** improvements, publish an evidence-backed candidate PR. For smaller or inconclusive results, checkpoint the findings and any promising operator edits without claiming success; continue researching within the allocation. **Never auto-merge**. Stop at GPU deadline/blocker with best verified SHA, rejected hypotheses, raw evidence paths and concrete next action. Do not claim unrun experiments.

Competition: https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn
