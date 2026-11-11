# v12 — Attention research queue and experimental decision matrix

Do not start a long experiment before reproducing an actual v12 production bottleneck. The existing v10-derived kernel already performs tiled QK, online softmax and PV in a single operator, with paged KV, `BLOCK_M=64` BF16 GQA prefill and decode page-lookup specialization. **Naively swapping to an advertised FlashAttention backend without genuine operator improvements is prohibited.**

## Phase-resolved bottleneck clues (historic only)

- v11 short sampled 16K profile reported **79.25% attention in summed GPU kernel time**, not 16K half-run serving wall time. A fresh v12 4K/16K profile is required.
- MiniCPM shapes observed in production diagnostics: BF16, 16 query heads, 2 KV heads, head dimension 128, 16-token KV pages, 2048-query-token prefill chunks, including 14,336 prior KV tokens. Revalidate in current serving.
- 4K prefix-cache warming and control run order can dominate tiny measured differences; always record hit rates.
- Compare separate prefill, mixed prefill/decode and pure decode. CUDA graphs may mask work from naive CPU trace summarization. Track per-kernel actual GPU events and serving wall time.

## Candidate ranking (rank changes after G0 profile)

| Track | Hypothesis | Expected workload / risk | Proof of genuine operator optimization | Acceptance tests |
| --- | --- | --- | --- | --- |
| E1 exact | FlashAttention-2 work partitioning / pipelined online softmax | 16K prefill; low model risk, medium engineering | Fewer re-scale/non-MMA ops, fewer K/V loads, improved K/V reuse, less SMEM synchronization, measured register spills | Dense output closeness at mixed paged layouts; GPU event A/B |
| E2 exact | Split-K/Flash-Decoding style long-KV parallelism + stable reduction | Long 16K decode and long-context mixed; low model risk, extra merge-launch overhead | Parallel KV ranges with numerically correct log-sum-exp partial combination, measured wall-time win at batch=64; preserve 2D/3D | Exhaustive corner cases, reduction ordering, graph replay and preemption |
| E3 exact | KV tile prefetch/double buffer + page-table address reuse | Mixed 4K and 16K, possible memory latency hiding | Actual memory transaction, instruction and dependency reduction, no dispatch-only change | Shuffled/padded/partial 16-token pages; cold/warm cache |
| S1 approximate | Local window + global prefix / attention sink blocks | 16K: potentially large work removal, severe forgotten-middle risk | **Skip K/V tile loads and both dot operations** for middle blocks, count visited tiles | Dense drift; beginning/middle/end needle and reasoned retrieval at 4K/8K/16K |
| S2 approximate | Query-aware top-K KV page selection (Quest-like) | 16K decode: preserves nonlocal candidate pages, selection overhead risk | Cheap Q/K page bound/minmax or representative summary + actual K/V gather; **net** compute/bytes savings | Recall of dense top-attention pages, adversarial middle retrieval, extra-page summary costs |
| S3 approximate | Hybrid dense prefill + sparse decode / global anchors / dilated blocks | 4K decode/16K decode, quality and behavior change | Source-defined kernel-local page selection with default dense; actual skipped ops | No loss of long-context, no surprise mixed-batch behavior |
| R conditional | Distributed Ring Attention | Only authorized multi-device serving with real inter-device KV traffic | Overlap measured KV communication and blockwise computation, same outputs | Confirm >1 GPU and relevant multi-GPU workload; otherwise reject as inapplicable |
| O exploratory | Structured split/reduce scheduling and query-group reuse | Exact alternatives if E1/E2 stall | New kernel pipeline, not backend-selection-only changes | Full fallback and numerical equivalence |

### Why the ranking

**Exact FlashAttention-style** improvements improve I/O scheduling and tiling while retaining all attention tokens: a useful benchmark-independent optimization. **Sparse attention** can reduce the number of attended tokens, but the base model was not trained with the candidate mask; very long middle-dependency questions may break even while MATH-500 is above 95%. Compare both paths in the first research allocation.

**Query-aware sparse** must select nonlocal pages efficiently enough that the selector itself does not consume the savings; no pretrained helper model. If a top-K method reads all K, performs full QK and then skips PV only, it is unlikely to save enough. Prefer page summaries and strict overhead accounting.

**Ring Attention** is a multi-device context-parallel method. A one-card BI-V150 run has no ring; single-GPU pipelining/double-buffering belongs under E1/E3 instead.

## Sparse testing methodology

- Define `q_abs_pos` precisely, causal mask per query row and 16-token KV page. Prefix/global tokens and local-window blocks may overlap: **never double-count** contributions to online softmax. Include `q_len=1` and mixed `q_len=[1,63,2048]` with nonzero prefixes, 2D/3D fallback and irregular page positions.
- Select initial **diagnostic** candidates independently of official benchmark/evaluator: dense, local W4096, local W4096 + first 128, local W8192 + first 128, and Query-aware page top-K with matched *actual* selected-token budgets. Treat these as research parameters, not competition serve hyperparameters or model training parameters.
- Synthetic long-context tests: exact retrieval of a randomly placed random key/value at positions 1%, 25%, 50%, 75%, 99%; variants with near-duplicate distractors, cross-sentence/multi-hop reasoning, different seeds; compare the dense baseline's success on identical prompts. Test 4K and 16K full inputs and 8K intermediate context. Cache/prefix must not conceal omitted work. A prompt that only queries the latest window is not sufficient.
- Report **retained-token ratio, tile-visit ratio, K/V bytes, QK/PV dot operations, selector time, kernel time, serving wall time**, and retrieval accuracy. Do not claim that token ratio = kernel speedup.
- Quality criteria: project MATH-500 >=95% on unchanged user-controlled test; no material long-context regression; no unexplained full-model distribution drift; organizer eligibility for approximation. If disallowed by model-behavior comparability, retain as research only and prioritize exact optimization.
- Do NOT perform prompt-specific sparse patterns, tune on benchmark inputs, or silently activate sparsity during score runs.

## Operator code locations

- `vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/triton_unified_attention_biv150.py`
- `vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/attention_biv150.py`
- Regression suites: `vllm-plugin-FL/tests/functional_tests/test_biv150_attention.py`, `test_biv150_decode.py`, `test_biv150_mixed_paging.py`.
- Existing experimental sparse candidate in separate unmerged [PR #237](https://github.com/prince22466/Co_AMAP/pull/237); **do not treat it as tested v12 code or use PR-wide branch merge**.

## References for inspiration, NOT drop-in code updates

- FlashAttention IO-aware exact attention: https://arxiv.org/abs/2205.14135
- FlashAttention-2 parallelism and work repartition: https://arxiv.org/abs/2307.08691
- Quest query-aware KV page sparsity: https://proceedings.mlr.press/v235/tang24l.html
- StreamingLLM attention sinks: https://proceedings.iclr.cc/paper_files/paper/2024/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html
- Ring Attention (multi-device): https://proceedings.iclr.cc/paper_files/paper/2024/hash/1119587863e78451f080da2a768c4935-Abstract-Conference.html
