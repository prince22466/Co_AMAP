# v15 latest research handoff

Status: setup only. No Codex session, GPU run, measured v15 gain or accuracy test has been performed.

## Research state (maintained by Codex; checkpoint helper preserves this section)

Original production trees: see AGENTS.md. Pin checkout SHA and verified GPU deadline at session start. First action: read the v0–v14 prior-work register and reuse existing profiling/capacity evidence. Run offline helper preflight, verify actual worker imports, then select the single highest-value missing resource or allocation-lifetime measurement. No historical experiment, rejected stack or generic baseline survey. Screen only a genuinely new justified candidate. No candidate source SHA or raw v15 results yet.

<!-- latest-checkpoint -->

## Latest checkpoint

Updated UTC: 2026-11-16T17:00:07+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: d356e5b158e3006972d44313f769072823b95c0e

Remaining GPU time: unknown

Finding: 4K MM Python dispatch shape frequency diagnostic

Evidence/provenance: Baseline source 1b680a7781ff0ba07c12d681708695fd5ccbf5eea18787d was not edited except temporary logger; correct pinned commit is 1b680a7781ff0ba07c12d681708695fd5ccda55e. Instrumented 4K quick screen: 8/8 requests, 1786.12 total tok/s, mean TTFT 10553.19 ms, mean TPOT 64.71 ms; NOT an uninstrumented baseline. After server startup, Python logger saw 672 each M=2048 K=2048 N=2560,2048,12288 and M=2048 K=6144 N=2048, plus 42 each of these at M=57. Logger does not observe graph replay; phase markers overwritten so interval is after application startup, including warmup; no representative full-run frequency claim. Raw startup log SHA256 8f2247c3eebb71ce4794211b33bf1670014622d14487c775ffd8aaec67090c99 and bench log 9e72c34de58a826d3132317dcbeabdb290ad694895cb597b21f265822f974c9a backed up locally; logger removed and server stopped.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Profile resource bottleneck of observed 2048x2048x12288 MM; use native counters and test only new K-loop hypothesis.

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
