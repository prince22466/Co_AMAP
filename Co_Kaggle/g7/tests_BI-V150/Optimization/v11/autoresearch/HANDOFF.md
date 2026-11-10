# v11 latest research handoff

Updated UTC: 2026-11-10T16:42:49+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: b2920586c9d431b661a2d7284bd773c7b44fc6a5

Remaining GPU time: unknown

Finding: Balanced short-serving screen is inconclusive and not a half-run go: control1/candidate1/control2/candidate2 4K 2464.25/2054.31/2473.56/2509.82; 16K 3196.90/2596.40/3194.16/3235.04

Evidence/provenance: All requests successful; control and candidate fresh restarts, sequence A1 B1 B2 A2; candidate1 graph capture compiled cold, repeat startup warm; 4K candidate2 vs control2 +1.47 percent, 16K +1.28 percent, candidate1 regressed materially; remote quick-new logs, seed 0 concurrency 8; short diagnostics only

Small CSV evidence committed when --push works: none (external backup required)

Next action: No-go for half-run of single-shape candidate; investigate actual second traced MLP projection and screen a narrowly extended tile-reuse hypothesis; preserve candidate and all raw logs

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
