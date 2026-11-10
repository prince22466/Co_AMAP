# v11 latest research handoff

Updated UTC: 2026-11-10T16:48:31+00:00

Status: hypothesis (researcher-reported)

Source HEAD before checkpoint: 1c12bff3c2b03b9b7b3d9aa32cfe60bdb50ebeb2

Remaining GPU time: unknown

Finding: Hypothesis: fuse softmax-rescaled accumulator into the BF16 GQA prefill P-by-V dot to reduce accumulator dependency and data movement

Evidence/provenance: 16K short profile sampled unified attention 79.25 percent of GPU kernel sum; existing prefill BLOCK_M=64 and two stages preserved; v10 resource profile suggests accumulator pressure but fresh register and wall share data missing

Small CSV evidence committed when --push works: none (external backup required)

Next action: Run numerical/fallback suite and alternating same-input kernel measurements; reject if numerical or performance screen fails

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
