# v11 latest research handoff

Updated UTC: 2026-11-10T09:36:26+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 75a2074564f1cfc83be1de3fc83c7b287c6bfd53

Remaining GPU time: unknown

Finding: PROVISIONAL lead: shape-specific grouped MM tile ordering 1.40223 ms control vs 1.16975 ms candidate, approximately 16.6 percent kernel-time reduction on traced 2048x2048 by 2048x12288 BF16 projection

Evidence/provenance: 8 alternating GPU event samples on identical transposed weights; outputs matched control and torch reference; fallback 57x2048x12288 and 7x2051x257 also matched; .autoresearch-work/mm-screen.log; no independent repetition, complete test coverage, production invocation proof for candidate or serving A/B

Small CSV evidence committed when --push works: none (external backup required)

Next action: Repeat kernel screen independently, back up full source diff and logs, restore unverified candidate before deadline; no speedup or progress gate claim

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
