# BI-V150 MiniCPM5-2B optimization — v10

**Starting code:** same tracked FlagGems and vLLM plugin operator sources as optimized v9, including the two-stage BF16 GQA prefill-attention kernel and mixed-paging correctness test. This repository-only setup does **not** claim a new performance gain; it prepares the remote Codex research workflow.

**Goal:** repeatable +6% relative to reported v9 **development half-run** means: 4K 2256.31 -> **>=2391.6886** tok/s, OR 16K 1518.06 -> **>=1609.1436** tok/s. Measure both exact cases using the untouched provided benchmark. These v9 values are reported reference anchors, not verified paired current-machine controls. Also obtain matched v10 baseline/candidate measurements to establish a causal gain.

**Research order:** fresh v10 4K decode GEMM/Linear profile, mixed prefill/decode attention, 16K attention data movement; KV-cache/idle only if fresh evidence warrants. No model/config/benchmark changes, backend/dispatch-only switching, quantization, speculative sampling, or latest FlagOS merges. MATH-500 deferred for human evaluation.

**Remote:** `/root/Optimization/v10` (Codex in existing Remote-SSH IDE). See [AGENTS.md](AGENTS.md), [program.md](program.md), and [autoresearch](autoresearch/README.md). Logs/CSV/traces remain in ignored `.autoresearch-work`; only evidence-backed v10 kernel changes and tests should be proposed in future PRs.
