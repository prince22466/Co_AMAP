# v11 latest research handoff

Updated UTC: 2026-11-10T16:29:57+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 3545c62889bbc94ae9dae853f1c97af5380e5db6

Remaining GPU time: unknown

Finding: First matched short-serving pair regressed: control to candidate 4K 2464.25 to 2054.31 tok/s and 16K 3196.90 to 2596.40 tok/s; all short requests successful

Evidence/provenance: Remote .autoresearch-work/quick-new/{control,candidate}-{4k,16k}.log; 4K control/candidate TTFT 6014.21/6309.28 ms TPOT 60.01/79.23 ms; 16K TTFT 19046.19/20042.71 ms TPOT 170.58/237.96 ms; seed 0 concurrency 8; candidate server fresh compilation had slower graph capture; diagnostic only

Small CSV evidence committed when --push works: none (external backup required)

Next action: Repeat balanced candidate then control with independent restarts, inspect cache/autotune and confirm candidate execution; no expensive half-run until serving screen justifies it

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
