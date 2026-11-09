# BI-V150 MiniCPM5-2B — v11 Codex Terra 6.1 autoresearch

**Starting point:** v11 FlagGems and vLLM plugin source Git trees equal optimized v10 at v11 creation (2026-11-09); v10's BF16 GQA prefill `BLOCK_M=64` and decode page lookup are preserved. This is a **docs/research-harness setup only**, not a claim of new kernel speedup.

**Fixed reported v10 development half-run reference:** 4K 2372.38 / 16K 1872.70 Total tok/s (`profiling/v10/summary.txt`). **Minimum +5% OR:** 4K >=2490.999 or 16K >=1966.335. **Stretch +10% OR:** 4K >=2609.618 or 16K >=2059.970. Measure both using the unchanged exact-case half-run; confirm true repeated matched uplift against *fresh v11 starting-code controls*. Favor >=10% if resource budget permits, do not suppress regressions. MATH-500 evaluation is human-owned, and the project's acceptance threshold is >=95% on the unchanged accuracy test.

**Workflow:** existing authorized Remote-SSH IDE, `/root/Optimization/v11` only. [AGENTS.md](AGENTS.md) is authoritative; [program.md](program.md) contains commands and the paste-ready Codex task; [autoresearch/EXPERIMENTS.md](autoresearch/EXPERIMENTS.md) is the living evidence ledger. The [CSV evaluator](autoresearch/evaluate.py) checks read-only benchmark output and never establishes accuracy or causal performance gain.

**Initial priorities (must reprofile):** 4K mixed prefill/decode paged-attention lookup/data movement, 4K small-M GEMM/Linear, 16K prefill attention KV pipeline. Hard rules: no quantization/speculative sampling, benchmark tuning, dispatch-only speedups, major operator-selection deletion, or upstream FlagOS framework merge. PRs are draft/evidence-backed; never auto-merge. Files alone do not launch Codex or an unattended job.
