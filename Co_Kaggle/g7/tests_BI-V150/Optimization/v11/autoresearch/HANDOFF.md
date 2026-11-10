# v11 latest research handoff

Updated UTC: 2026-11-10T09:09:39+00:00

Status: hypothesis (researcher-reported)

Source HEAD before checkpoint: 9bc87e97de15053bf1aa02486eba99234d58b095

Remaining GPU time: unknown

Finding: Second bounded hypothesis: explicitly pipeline the 16K prefill KV tile loop with two in-flight stages while preserving non-prefill loop defaults

Evidence/provenance: 16K initial prefill trace: attention 840 launches and 9.995 s sampled GPU sum, median 11.337 ms range 5.822 to 19.975 ms; this is not wall time; first page-gather candidate rejected

Small CSV evidence committed when --push works: none (external backup required)

Next action: Run GPU correctness and alternating same-input kernel screen; reject if slow or unverified before 25-minute reserve

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
