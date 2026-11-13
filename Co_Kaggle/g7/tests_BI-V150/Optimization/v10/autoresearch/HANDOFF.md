# v10 handoff — 2026-11-08

Historical PR: https://github.com/prince22466/Co_AMAP/pull/229 (merged). Results below predate the device-independent refactor; the kernel file is relocated without any source-content change. Revalidate the refactored backend before using these results as its performance evidence.

## Historical code and verification (before the device-independent refactor)

- Remote workspace: `/root/Optimization/v10`. The **candidate is restored**; no owned server or benchmark is running.
- In the existing specialized BF16 GQA prefill path, `BLOCK_M` changes from 32 to 64: eight query positions reuse a KV tile instead of four. The two-stage pipeline and feature/dispatch gates are preserved.
- Regression tests cover 7-, 9- and 65-token query tails, long prefill, decode, mixed paging and fallback. The reviewed test-allocation issue is fixed by adding a spare page-table/cache entry, matching the pinned reference's padded-table contract. Numerical tolerances are unchanged.
- Corrected GPU suites: **21 passed for candidate and 21 for control**. CSV helper: **17 passed**.
- Candidate attention SHA-256: `6e3e8f7cbc24140a9cfd3ef3d2ed8d99cf7ae0f4e0147ac3d9d9b0775925931a`.

## Completed development half-run pair

Unmodified provided benchmark, `RUNS=2`, `SKIP_FIRST=0`, both exact workloads and all requests successful:

| Workload | Control A1 repetitions | Candidate B1 repetitions | Control mean | Candidate mean | Matched change |
| --- | --- | --- | ---: | ---: | ---: |
| 4K | 2103.78 / 2474.28 | 2078.02 / 2677.88 | 2289.03 | 2377.95 | +3.88% |
| 16K | 1488.68 / 1537.11 | 1854.37 / 1894.23 | 1512.89 | 1874.30 | +23.89% |

Values are Total tok/s; means are the benchmark's reported summaries. Candidate 16K is +23.47% versus the fixed reported v9 anchor of 1518.06, above the 1609.1436 target. Candidate 4K is +5.39% versus its fixed v9 anchor and remains below the 2391.6886 target. These are development half-runs, not official full-run scores.

For A1 -> B1, 16K mean TTFT improved 345098.05 -> 273240.49 ms, TPOT 299.55 -> 244.30 ms, and output rate 89.00 -> 110.25 tok/s. The corresponding 4K metrics also improved. Retain the large first/second 4K spread when interpreting the gain.

The independent control A2 completed 4K at 2102.89 / 2473.88 (mean 2288.39) and its first 16K repetition at 1489.88. Its last 16K repetition was **interrupted at the user's request**; A2 is incomplete and B2 was not run. Independent complete matched confirmation and human MATH-500 were outstanding at this historical handoff.

## Provenance and artifacts

- Remote session root: `/root/Optimization/v10/.autoresearch-work/validation_20261108/`.
- Complete control: `A1_GXj1WMMX/benchmark_results/`; complete candidate: `B1_rWYb0Sgt/benchmark_results/`; interrupted control: `A2_hqg16pBi/` with `USER_STOPPED.txt`.
- Local copies and comparison JSON: the original local `v10/.autoresearch-work/validation_20261108/` directory. Bulk artifacts are Git-ignored.
- Source manifests cover 2,812 serving files: A1/A2 match exactly; only the intended attention source differs in B1. Logged server arguments match. Automatic KV capacity was 521840 tokens in A1 and 521936 in B1/A2; no capacity or scheduling flag was adjusted.
- Model hashes/package versions, test logs, compiler-resource audit and telemetry are retained beside the run directories. Static compiler fields are not runtime spill/bandwidth measurements.

## Run the candidate yourself

Terminal A, from the restored candidate workspace (port 9031 must be free):

```bash
cd /root/Optimization/v10
mkdir -p .autoresearch-work/serve_runs
serve_log=$(mktemp "$PWD/.autoresearch-work/serve_runs/human_XXXXXXXX.log")
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL:${PYTHONPATH:-}"
export VLLM_PLUGINS=fl
unset VLLM_FL_USE_FLAGGEMS_ATTN VLLM_ATTENTION_BACKEND VLLM_BATCH_INVARIANT
set -o pipefail
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 2>&1 | tee "$serve_log"
```

Terminal B, after the health endpoint is ready:

```bash
curl -f http://127.0.0.1:9031/health
cd /root/Optimization/v10
mkdir -p .autoresearch-work/bench_runs
run_dir=$(mktemp -d "$PWD/.autoresearch-work/bench_runs/human_XXXXXXXX")
cd "$run_dir"
sha256sum /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py > benchmark_sha256.txt
set -o pipefail
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]' \
  2>&1 | tee benchmark.log
```

Do not modify the provided benchmark. Verify four raw SUCCESS rows with 256/256 requests for each 4K repetition and 128/128 for each 16K repetition. Reserve at least the documented 90-minute start floor; completion can take longer. A fresh server restart and the same workload order are required for matched comparisons.

For a historical kernel-only comparison within the current shared Triton backend (not a full pre-refactor dispatch comparison), the saved control source is `.autoresearch-work/validation_20261108/control_attention.py`; saved candidate source is `candidate_attention.py`. To perform a control measurement, stop the owned server first, replace only `vllm-plugin-FL/vllm_fl/dispatch/backends/triton_unified_attention.py` with the saved control, and restart with identical flags. Restore the saved candidate afterward. Control attention SHA-256 is `c651c1f0172feadf1ec1f286fe4466b25c45e25496dc1b1b31237a25946fd55b`.
