# v11 latest research handoff

Updated UTC: 2026-11-10T16:21:27+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: f84547ff78a075e572d5fb2999f50b8fbcd1dfe9

Remaining GPU time: unknown

Finding: Matched short control diagnostics completed: 4K 16/16, total 2464.25 tok/s; 16K 8/8, total 3196.90 tok/s

Evidence/provenance: Remote .autoresearch-work/quick-new/control-4k.log and control-16k.log; seed 0, fixed random inputs, concurrency 8, output 128; diagnostic only, 4K TTFT 6014.21 ms TPOT 60.01 ms; 16K TTFT 19046.19 ms TPOT 170.58 ms; candidate comparison pending

Small CSV evidence committed when --push works: none (external backup required)

Next action: Restart matching server on candidate, run same probes, verify candidate execution and compare both without treating screen as half-run

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
