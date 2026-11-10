# v11 latest research handoff

Updated UTC: 2026-11-10T09:12:34+00:00

Status: rejected (researcher-reported)

Source HEAD before checkpoint: 2572fd9949695eada86badefbcfc10e20d4e0d00

Remaining GPU time: unknown

Finding: REJECTED: explicit two-stage prefill loop passed 21 GPU tests but did not improve paired kernel medians

Evidence/provenance: Alternating 10 samples, BF16 query 2048 with randomized 16-token pages: context 4096 control 5.7891 ms candidate 5.7782 ms; context 14336 control 17.4541 ms candidate 17.4999 ms; outputs identical; remote .autoresearch-work/pipeline-tests.log and pipeline-screen.log; control SHA256 6e3e8f7cbc24140a9cfd3ef3d2ed8d99cf7ae0f4e0147ac3d9d9b0775925931a restored

Small CSV evidence committed when --push works: none (external backup required)

Next action: Back up both screen logs and traces, verify PR and stop at preservation reserve; no candidate promoted

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
