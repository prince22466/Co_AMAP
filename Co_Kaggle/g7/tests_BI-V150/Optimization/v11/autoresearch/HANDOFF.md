# v11 latest research handoff

Updated UTC: 2026-11-10T09:33:43+00:00

Status: hypothesis (researcher-reported)

Source HEAD before checkpoint: b4afd39b41faa9de2962d244b4510e75aafdbac6

Remaining GPU time: unknown

Finding: Follow-up 4K trace: BF16 MLP-sized projection (2048x2048) x (2048x12288) contributes 942.19 ms of 1806.45 ms sampled mm kernel time

Evidence/provenance: Trace CPU-to-GPU external-id attribution: 672 such calls, median 1402.18 us; second (2048x6144)x(6144x2048) 487.28 ms of sampled mm sum; first 20 profiled iterations, CUDA graphs and serving wall share unresolved; source .autoresearch-work/trace-associate.py

Small CSV evidence committed when --push works: none (external backup required)

Next action: Screen a shape-specific tile-order data-reuse idea against identical BF16 inputs; require correctness and reject regressions; no serving claim

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
