# v15 — GEMM and KV-cache capacity autoresearch

v15 starts with identical production source trees to v14 (FlagGems and vllm-plugin-FL). Setup only: no new inference optimization or GPU result is implied.

For Codex Terra 6.1 on the authorized BI-V150 Remote-SSH host, read [AGENTS.md](AGENTS.md), [program.md](program.md), and [research queue](autoresearch/RESEARCH_QUEUE.md). Maintain [experiment ledger](autoresearch/EXPERIMENTS.md) and [handoff](autoresearch/HANDOFF.md).

Research starts beyond v0–v14: apply the prior-work register before any GPU experiment. Reuse prior measurements and test coverage; do not repeat old profiling, rejected stacks, grouped-MM, KV-write tiling, attention work or generic capacity discovery. Prioritize new GEMM mechanisms and actual KV-capacity savings.

Goal: independently repeatable >=5% total tok/s gain over a fresh matched v15-original control in either unchanged 4K or 16K half-run, ideally >=10%, without material regression in the other workload or loss of model capability. Historical v14 full-run results are linked in the research queue; only fresh matched half-runs determine this research goal.

Competition: https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn
