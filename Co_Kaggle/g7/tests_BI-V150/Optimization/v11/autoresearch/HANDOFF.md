# v11 latest research handoff

Updated UTC: 2026-11-10T16:43:10+00:00

Status: hypothesis (researcher-reported)

Source HEAD before checkpoint: ad84a15b57a442a28fb51377b19cd308877e1ee7

Remaining GPU time: unknown

Finding: Next hypothesis: extend the same grouped MM tile-order data reuse to the second traced MLP projection (2048x6144 by 6144x2048)

Evidence/provenance: 4K sampled GPU profile: first projection 942.19 ms and second projection 487.28 ms of 3891 ms GPU kernel sum; single-shape serving screen did not demonstrate a repeatable win; no wall-time inference

Small CSV evidence committed when --push works: none (external backup required)

Next action: Prove second-shape numerical correctness, alternating kernel A/B and paired two-workload serving screening; reject on material regression

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
