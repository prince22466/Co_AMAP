# v12 sparse attention research ledger

**2026-11-10 — plan/prototype only; no BI-V150 GPU experiments performed in this PR.**

| Candidate | BF16 prefill mask | GPU correctness | Kernel A/B | Production proof | 4K/16K serving | MATH-500/long-context |
| --- | --- | --- | --- | --- | --- | --- |
| A | Dense v10 baseline | Historical v10 only; recheck | Pending | Existing v10 only | Historical 2372.38/1872.70 reference (not fresh matched) | Historical v10 97.1% MATH-500 reported |
| B | Window 4096, sink 0 | Pending | Pending | Not enabled | Not run | Not run |
| C | Window 4096, sink 128 | Pending | Pending | Not enabled | Not run | Not run |
| D | Window 8192, sink 128 | Pending | Pending | Not enabled | Not run | Not run |

Full dataset and log trace references are needed before updating a cell to verified. Unit tests that compare against a **sparse** reference do not establish dense equivalence. This is an intentionally approximate attention algorithm. Never imply sparse attention is competition-compliant merely because the rules permit operator kernel rewrites.

## Hypothesis and technical caveats

- Sparse attention is only useful when actual K/V loads and QK/PV dot tiles are omitted; setting scores to `-inf` after dense calculation has no meaningful benefit.
- For each query block, retain the first S global tokens plus all KV tiles that overlap the union of the most recent W tokens of its valid queries. Mask within boundary tiles per **query row** to enforce exact sparse semantics and causal direction.
- Mixed prefill/decode and 3D decode must remain dense at the start; do not silently alter already-enabled GQA or page lookup optimizations.
- The 4K benchmark may be prefix-cache sensitive; per-repetition data and cache-hit rates matter.
- The 16K diagnostic attention GPU kernel-sum share of 79.25% is not an Amdahl wall-time share for half-run serving.
- Objective is **measured GPU time and tokens/s at acceptable model behavior**, not maximum theoretical sparsity. Preserve skipped-region retrieval evidence and worst cases.

## Experiment record template

### ID / UTC / source SHA / allocated time
- Shape, W/S, batch layout, page size, Q/context lengths:
- Dense reference output error (max abs / relative / cosine) and sparse-reference error:
- BF16 correctness, finite outputs, causal/tail/page/mixed/supported-feature tests:
- Alternating kernel A/B (median p10/p90; compile excluded; sampled tiles):
- Proof of use in serving and correctness of chosen fallback:
- Matched short-serving 4K and 16K, with TTFT/TPOT and success counts:
- Full unchanged half-run raw CSVs and checksums, success rows, source hashes:
- MATH-500 Level 3 accuracy AND long-context retrieval/ability checks:
- Rules/semantics risk assessed; eligible/pending/rejected:
- Decision, PR, next experiment:
