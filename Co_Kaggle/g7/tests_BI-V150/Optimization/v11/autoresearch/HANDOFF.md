# v11 latest research handoff

Updated UTC: 2026-11-10T09:05:59+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: d8a0e99de350a34faff114a4e7a890a4efa10cea

Remaining GPU time: unknown

Finding: Experimental prefill page-table reuse passed 21 attention/decode/mixed GPU tests; alternating same-input kernel A/B is in progress

Evidence/provenance: Remote .autoresearch-work/candidate-tests.log; candidate preserves decode lookup and BLOCK_M=64, checks BF16 GQA tails and mixed paging; correctness alone is not performance evidence

Small CSV evidence committed when --push works: none (external backup required)

Next action: Inspect interleaved GPU timings; revert candidate if weak or unverified; no half-run under deadline

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
