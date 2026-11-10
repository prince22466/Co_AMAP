# v11 latest research handoff

Updated UTC: 2026-11-10T09:14:36+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: e3330c0ddfe1117259da795509771cedcaad10cc

Remaining GPU time: unknown

Finding: Final restoration verified: 2812 tracked inference working-tree files match remote bytes; both experimental operators rejected and no v11 speedup verified

Evidence/provenance: 4K control 2211.04/2526.52 and 16K 1824.16/1894.52 total tok/s; all 4 rows SUCCESS; two candidate tests 21/21 and alternating A/B logs backed up in local .autoresearch-work/remote-handoff; profiler logs and compressed traces backed up; checkpoint evidence committed to draft PR; no owned tmux sessions remain

Small CSV evidence committed when --push works: none (external backup required)

Next action: Next allocation: capture phase-resolved serving traces including graph replay and real wall shares; investigate 16K KV pipeline or real small-M GEMM only with evidence; reproduce baseline and matched independent A/B; human MATH-500 >=95 percent for chosen candidate

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
