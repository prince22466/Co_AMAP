# v11 latest research handoff

Updated UTC: 2026-11-10T09:38:14+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: ce54f19b5842ad8949e6aeddc72c1b3362a5a241

Remaining GPU time: unknown

Finding: Promising but unpromoted MM lead replicated in separate process: 2048x2048 by 2048x12288 BF16 grouped tile mapping reduces kernel median 1.40223 to 1.16975 ms and 1.41153 to 1.17371 ms

Evidence/provenance: Two runs of eight alternating GPU event samples, separate seeds; candidate outputs match control and torch reference for observed shape, small-M and tail cases; additional transposed-input shape 1.45587 to 1.17392 ms and numerically matches. 4K trace sampled this shape at 942.19 of 1806.45 ms MM kernel sum. Remote mm-screen.log and mm-screen-2.log; candidate backup .autoresearch-work/mm-candidate.py and .patch local. No complete correctness/fallback coverage, production candidate trace, serving A/B or independent serving replication.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Restore control; next allocation run focused operator tests and matched short serving probes, then guarded independent A/B only if justified

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
