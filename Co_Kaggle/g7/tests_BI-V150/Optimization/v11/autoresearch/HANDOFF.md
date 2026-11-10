# v11 latest research handoff

Updated UTC: 2026-11-10T16:35:58+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 6b415ddad7a2a43d0bf718df29fc07188f44faa3

Remaining GPU time: unknown

Finding: Candidate short-serving repeat after fresh restart improved versus candidate first: 4K 2509.82 and 16K 3235.04 total tok/s; all requests successful

Evidence/provenance: Remote quick-new/candidate2-{4k,16k}.log, 4K 16/16 TTFT 5885.25 ms TPOT 59.07 ms; 16K 8/8 TTFT 18742.66 ms TPOT 169.12 ms; candidate-first 2054.31 and 2596.40; control-first 2464.25 and 3196.90; order and cold-start sensitivity unresolved

Small CSV evidence committed when --push works: none (external backup required)

Next action: Fresh control repeat in reversed order, compare and investigate production execution; no causal win claimed

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
