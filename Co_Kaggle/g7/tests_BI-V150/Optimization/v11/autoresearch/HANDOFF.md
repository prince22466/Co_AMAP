# v11 latest research handoff

Updated UTC: 2026-11-10T16:14:22+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 31e1ef8d6fd496491378a7d0ff3396623806c289

Remaining GPU time: unknown

Finding: Provisional grouped-MM candidate passed seven focused BI-V150 tests including observed BF16 shape, FP16, transposed and strided inputs, fallback tails and strided output

Evidence/provenance: Remote .autoresearch-work/mm-tests-new.log: 7 passed; candidate mm sha256 a6edcade7c0805cb4eeae41828d1d089b1b76ae3b383176779bd7fe7a89ed276; initial control/model/benchmark hashes unchanged; no serving evidence yet

Small CSV evidence committed when --push works: none (external backup required)

Next action: Run full attention regression suite, then matched control/candidate short 4K and 16K diagnostics with identical restarts and order

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
