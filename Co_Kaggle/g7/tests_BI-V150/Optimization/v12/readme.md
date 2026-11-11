# v12 — Attention autoresearch (BI-V150 / MiniCPM5-2B)

**Purpose:** seek repeatable end-to-end speedups without losing long-range model abilities. The existing v12 kernels are v10-equivalent; this PR changes only the research plan and evidence helpers, **not inference code**.

Read **[AGENTS.md](AGENTS.md)** for non-negotiable constraints, **[program.md](program.md)** for remote commands and the Codex task, and **[autoresearch/RESEARCH_QUEUE.md](autoresearch/RESEARCH_QUEUE.md)** for ranked hypotheses. Append findings to **[autoresearch/EXPERIMENTS.md](autoresearch/EXPERIMENTS.md)**.

**Target (total tok/s, either scenario):** reported v10 4K 2372.38 / 16K 1872.70; seek +5% (2490.999 / 1966.335), ideally +10% (2609.618 / 2059.970), **and independently replicated matched gains against unchanged v12**. Always measure both scenarios, TTFT and quality.

**Research priority:** exact I/O-aware attention and paged-KV optimizations first; gated block-sparse/query-aware KV selection as an exploratory second track. Ring Attention is conditional on real multi-GPU hardware. Sparse attention can change outputs and invalidate the competition entry despite passing MATH-500 >=95%; keep it research-only without separate eligibility and long-context evidence.

**Status:** setup only. No Codex session, GPU test, accuracy test or serving speedup is implied by these files. The unmerged [PR #237](https://github.com/prince22466/Co_AMAP/pull/237) is an unverified sparse prototype, not v12 main code.

Competition: https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn
