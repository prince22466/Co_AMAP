# v11 latest research handoff

Updated UTC: 2026-11-10T09:39:51+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 31b8d461dfc53e61cc6419472044830ce63012ea

Remaining GPU time: unknown

Finding: Final follow-up: replicated 16.6 to 16.8 percent kernel-time reduction on observed BF16 2048x2048 by 2048x12288 MM shape; candidate unpromoted and original MM restored

Evidence/provenance: Two exact local off-instance logs .autoresearch-work/remote-handoff/mm-screen.log sha256 41b6905a and mm-screen-2.log sha256 cc7f5271 match remote. Saved provisional candidate .autoresearch-work/mm-candidate.py sha256 a6edcade and mm-candidate.patch sha256 f869ab51, plus mm-screen-first.py and mm-screen.py scripts. Original mm sha256 4cb0cff3 restored and Git inference tree clean. Traced MM projection consumed 942.19 ms of 3891 ms sampled 4K GPU sum; not a wall-time estimate.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Next allocation apply candidate only in v11; add focused noncontiguous, fallback, numerical and real-shape tests; verify candidate executes in production, matched short 4K and 16K serving diagnostics, then guarded matched A/B with independent replication if justified; human MATH-500 remains pending

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
