# BI-V150 MiniCPM5-2B — v12 sparse-attention exploration

**Status: experimental, unverified. Not competition-submission code.**

Starting operator trees are copied from **v10** (the last reported high-throughput/accuracy-qualified candidate), not v11's grouped-MM prototype. In particular the v10 BF16 GQA prefill `BLOCK_M=64`, decode page lookup, and all framework dispatch/fallback logic are retained.

Goal: quantify the benefits and losses of **real K/V tile skipping** in a block-sparse causal attention operator for MiniCPM5-2B. The first training-free hypothesis is **first S tokens (prefix/global) + most recent W tokens (local)**. It is approximate attention, not output-equivalent to the dense model. A sequence/query-block-aware sparse tile iterator is introduced behind explicit **default-off per-call test arguments** in the existing Triton unified attention operator; only verified BF16 GQA causal prefill shapes qualify. Other cases must continue on the dense path. No framework operator switch, quantization, speculative decoding, benchmark edits, serving-flag changes, or upstream merge.

Why: v11's sampled 16K GPU kernel sum attributed 79.25% to attention in a short diagnostic, although **this is not 16K serving wall-clock share**. Sparse KV access is a higher-upside hypothesis than further single-shape MM tile scheduling. MiniCPM5-2B was not pretrained as a sparse-attention model: large accuracy and long-range retrieval regressions are possible. Changing its behavior may be disallowed under competition's comparability rule even when MATH-500 >=95%; **seek organizer determination before submitting**.

Research matrix (not tuned official-scoring hyperparameters):
- Dense control: W=0, S=0, unmodified v10 behavior.
- Local only: W=4096, S=0.
- Local+prefix: W=4096, S=128.
- Wider local+prefix: W=8192, S=128.
- Diagnostic small windows (e.g. W=32, S=16) **only for numerical/reference tests**.

**No v12 GPU tests or half-runs have been executed by this PR.** Kernel changes require compilation, direct sparse-reference correctness, mixed paging/prefix-cache checks, GPU A/B, production-path proof, matched serving evaluations and longer-context capability tests. See [program.md](program.md) and [EXPERIMENTS.md](EXPERIMENTS.md). Preserve all raw test and profiler evidence and document SHA-based code provenance.

Fixed reported comparison anchor: v10 development half-run 4K **2372.38** / 16K **1872.70** total tok/s. Re-run matched dense control on same GPU rather than comparing only to these historical summaries. The competition scores the average of 4K and 16K scenario improvement; don't maximize 16K by sacrificing 4K. MATH-500 acceptance is >=95%, but is not a sufficient sparse-attention correctness gate.

Competition: https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn
