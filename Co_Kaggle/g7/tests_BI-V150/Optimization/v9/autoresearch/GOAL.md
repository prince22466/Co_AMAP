# v9 Codex autoresearch contract

Keep trying genuine BI-V150 kernel/operator optimizations until one verified
milestone is met: 4K total tokens/s >= 2700 or 16K >= 1500, OR a declared
wall-clock time budget is exhausted, OR remote access/evidence is blocked.

A milestone is verified only with unmodified full official workloads,
MATH-500 >= 0.95, all requests successful, other workload loss <= 1%,
and mean TTFT loss <= 1% in both workloads. Benchmark pairing, hardware
and server settings must match; rerun interleaved A/B near the noise floor.
The competition scores two workloads: the OR threshold is a research stop
condition, not license to regress the other.

Priority 1 — 16K attention: memory dependency chains, shared-memory footprint,
paged-KV address load, Q/K/V tile reuse, spills and occupancy (strong evidence).
Priority 2 — mixed prefill/decode interference: isolated/mixed phase causal
profiling, then kernel-level correction (moderate evidence).
Priority 3 — 4K attention and prefill GEMM: locality, tl.dot reuse, grouped
scheduling in existing operators (strong kernel-time evidence).

Use ONE coherent hypothesis at a time:
BASELINE -> PROFILE -> PATCH -> GPU TEST -> FULL A/B -> MATH500 -> SCORE ->
ACCEPT+CHECKPOINT PR or REJECT+ROLLBACK -> NEXT.
Do not modify benchmark scripts. Microbenchmarks are only diagnostic.
Prior v4/v6/v7 stack results are negative evidence, not reusable wins.

After each round write .autoresearch-work/round-report.json with fields:
- round: integer matching the supervisor iteration
- status: improved | goal_reached | no_gain | blocked
- gate_report: path to the evaluate.py JSON, or null
- pr_url: real newly opened GitHub PR URL, or null
- notes: concise evidence, next hypothesis or blocker

Only use status improved/goal_reached if gate JSON has passed=true and
the matching goal status, and pr_url is a real PR you opened. Report
blocked rather than fabricate a result or quietly lower gate criteria.
Each accepted result gets a separate PR immediately, even below targets.
Build later changes atop last accepted best and leave earlier checkpoint
PR open for independent review. Never merge without human review.
