# v12 sparse-attention GPU experiment — execution plan

## 0. Confirm provenance
```bash
cd /root/Optimization/v12
git rev-parse HEAD
git status --short
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl
sha256sum /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py
```
Record actual allocation deadline before scheduling any expensive runs. No remote GPU action was performed during this PR's creation.

## 1. GPU correctness: dense by default, sparse vs explicit reference
```bash
cd /root/Optimization/v12
python3 -m pytest -q -s --noconftest \
 vllm-plugin-FL/tests/functional_tests/test_biv150_attention.py \
 vllm-plugin-FL/tests/functional_tests/test_biv150_decode.py \
 vllm-plugin-FL/tests/functional_tests/test_biv150_mixed_paging.py \
 vllm-plugin-FL/tests/functional_tests/test_biv150_sparse_v12.py
```
The sparse experimental test calls `unified_attention(..., experimental_sparse_window_tokens=W, experimental_sparse_sink_tokens=S)`. The normal backend call passes neither argument, so it remains dense. Verify this by spy/invocation tests. Extend tests for 7/9/65 query tails, 16-token shuffled pages, nonzero KV context, local+sink mask, no sinks, very long context, segmented decode, mixed prefill/decode, and empty/short context. Test wrong/unsupported features fail closed.

## 2. Operator-level GPU A/B (not scoring)
Measure alternating dense vs sparse in the **same pinned package/engine** for:
- prefill: 2048 fresh queries after 0/4096/8192/14336 KV prefix tokens;
- exact GQA BF16 16Q/2KV/128 head, standard 16-token paged cache;
- query/block masks representative of 16K prefill and mixed batches;
- W in {4096,8192}, S in {0,128}; optionally smaller W solely for unit testing.

Use warmed event timing, independent seeded repetitions, identical K/V layout and no host synchronization in timed regions. Record kernel medians/variance, visited tile counts, occupancy/registers/SMEM/spills and model-output error vs **dense** attention. A local+prefix mask can be internally correct and still degrade model quality.

## 3. Determine whether production rollout is worth attempting
If sparse kernel wins without unacceptable output drift, wire the operator's experimental arguments through `attention_biv150.py` **as a documented code-level experimental branch**, preserving dense fallbacks and no vLLM serve CLI flag changes. First verify actual production invocation via source hash, logs and GPU trace. Do **not** claim performance impact from unit/standalone kernel tests.

Run short serving diagnostics with identical restarts and workload/seed/order, then quality tests (MATH-500 unchanged Level 3, long-context needle/retrieval at 4K/8K/16K, reasoning consistency, repeated seeds). An accuracy result below 95% immediately rejects a candidate, but >=95% alone does not establish comparability.

## 4. End-to-end development half-run, if admitted
Serve (unchanged competition flags):
```bash
cd /root/Optimization/v12
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl
unset VLLM_FL_USE_FLAGGEMS_ATTN VLLM_ATTENTION_BACKEND VLLM_BATCH_INVARIANT
vllm serve /workspace/MiniCPM5-2B \
 --port 9031 \
 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
 --served-model-name minicpm \
 --gpu-memory-utilization 0.85 \
 --max-model-len 131072
```
Use the untouched external half-run script only, after source/kernel proof; two repetitions each of [4096,1024,64,256] and [16384,1024,64,128]. Preserve four SUCCESS rows per full half-run. Do A1/B1/A2/B2 with distinct fresh starts and comparable cache conditions where allocation permits. Never modify benchmark scripts or change model/generation/serve flags; don't claim official scores. Retain original raw CSVs and compare **both** workloads, TTFT, throughput and accuracy. Stop an unpromising experiment early; preserve GPU allocation.

## Acceptance
- Sparse kernel correctness with explicit sparse reference and non-NaN, paged-KV safe operations.
- Independent matched serving uplift; ideally >=5–10% improvement for at least one workload without material regression in the other.
- No material long-context ability loss; unchanged official MATH-500 Level 3 >=95%; competition rule review before any sparse candidate enters submission.
- No inference optimization by dispatch-only switching, hyperparameter-based quantization/speculative sampling, benchmark edits or upstream merges.

## Repeatable research record
Append each attempt to `EXPERIMENTS.md` with source SHA, candidate W/S, observed GPU timings, error summary, serving results, raw path + hash, correctness and rejection reason. Preserve honest findings even if sparse attention is ultimately unsuitable.
