# v11 latest research handoff

Updated UTC: 2026-11-10T16:04:39+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 5eda087954f3d3b0b8dd1c67c9c8d0e593447495

Remaining GPU time: unknown

Finding: Prepared unrun GPU candidate tests for traced BF16 projection, transposed and strided inputs, FP16 and small/tail fallback cases; previous GPU deadline expired

Evidence/provenance: Local ignored .autoresearch-work/test_grouped_mm_candidate.py passes Python compilation only; candidate remains unapplied; prior repeated kernel A/B and full control evidence in PR #236

Small CSV evidence committed when --push works: none (external backup required)

Next action: Once operator confirms new allocation and deadline, initialize its budget, verify provenance, apply candidate in v11, execute candidate tests on BI-V150, and run matched diagnostics if correctness passes

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
