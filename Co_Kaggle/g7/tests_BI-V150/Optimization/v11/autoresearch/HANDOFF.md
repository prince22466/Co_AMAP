# v11 latest research handoff

Updated UTC: 2026-11-10T05:09:46+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 8fa3dbc01137f3c3eece5d90911f4b0a75756d7b

Remaining GPU time: unknown

Finding: Final 16K control repetition running; cumulative server preemptions increased from 3 to 5

Evidence/provenance: Server vllm:num_preemptions_total=5 sampled near 05:09 UTC; cache near capacity; three successful raw rows already preserved; no wall-time attribution

Small CSV evidence committed when --push works: none (external backup required)

Next action: Complete control and back up all evidence before production profiling

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
