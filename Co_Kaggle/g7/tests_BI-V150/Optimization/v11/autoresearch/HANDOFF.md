# v11 latest research handoff

Updated UTC: 2026-11-10T09:07:45+00:00

Status: rejected (researcher-reported)

Source HEAD before checkpoint: fe8c3f5ac31d387417ce4721d67c5555aaccc842

Remaining GPU time: unknown

Finding: REJECTED: BF16 GQA prefill page-index reuse passed 21 tests but regressed paired kernel medians by 29.99 percent at context 4096 and 29.88 percent at context 14336; reverted

Evidence/provenance: Identical BF16 query=2048 and randomized 16-token pages; alternating 10 samples: 5.7616 to 7.4898 ms and 17.4620 to 22.6797 ms; max output difference 0; remote .autoresearch-work/kernel-screen.log and candidate-tests.log; original source SHA256 6e3e8f7cbc24140a9cfd3ef3d2ed8d99cf7ae0f4e0147ac3d9d9b0775925931a restored; no serving test

Small CSV evidence committed when --push works: none (external backup required)

Next action: Back up negative raw logs; inspect another safe hypothesis only if time permits; preserve 25-minute handoff reserve

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
