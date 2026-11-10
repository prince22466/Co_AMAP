# v11 latest research handoff

Updated UTC: 2026-11-10T05:22:51+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: f53d6d915ae4c3f909f110fa1d6a04a66c59120f

Remaining GPU time: unknown

Finding: Fresh starting-code control complete: 4K mean 2368.78 tok/s, repeats 2211.04 and 2526.52; 16K mean 1859.34, repeats 1824.16 and 1894.52; all requests successful

Evidence/provenance: run guard_20261110T042048_3897 elapsed 60.71 min; four SUCCESS rows validated by evaluate.py; off-instance local backup .autoresearch-work/remote-handoff/guard_20261110T042048_3897 plus server and GPU telemetry logs; source and model hashes in preceding checkpoints; no speedup or accuracy claim

Small CSV evidence committed when --push works: autoresearch\evidence\guard_20261110T042048_3897_4d33913621d4_raw_runs_20261110_122048.csv (sha256 4d33913621d4d7db32b7c00be5afffa169a8fc7d8f9b965e85b0dc38945aa3c0); autoresearch\evidence\guard_20261110T042048_3897_83f6edc1c9ec_summary_20261110_122048.csv (sha256 83f6edc1c9ec39dcd0e6b3c74e9f8816d908bd86fd580f2238c0b9075211f064)

Next action: Profile production phases and screen one operator hypothesis; admit candidate half-run only if justified and time permits

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
