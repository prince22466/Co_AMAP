# v13 validation runbook

Run in the authorized BI-V150 environment. Do not overwrite or edit v10. For matched comparisons, start each version separately with identical environment, arguments and benchmark order. Keep port 9031 free before starting a new owned server.

## 1. GPU correctness (v13)

```bash
cd /root/Optimization/v13
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PLUGINS=fl
unset VLLM_FL_USE_FLAGGEMS_ATTN VLLM_ATTENTION_BACKEND VLLM_BATCH_INVARIANT
python3 -m pytest -q -s --noconftest \
  vllm-plugin-FL/tests/functional_tests/test_biv150_attention.py \
  vllm-plugin-FL/tests/functional_tests/test_biv150_decode.py \
  vllm-plugin-FL/tests/functional_tests/test_biv150_mixed_paging.py
```

## 2. Start serving (Terminal A)

Set `version=v13` or `version=v10` for a read-only matched control. Use a fresh terminal for each version, and ensure the selected version's source paths take precedence. Stop only the owned server after the run.

```bash
version=v13
cd "/root/Optimization/$version"
mkdir -p .autoresearch-work/serve_runs
serve_log=$(mktemp "$PWD/.autoresearch-work/serve_runs/serve_XXXXXXXX.log")
export PYTHONPATH="$PWD/FlagGems/src:$PWD/vllm-plugin-FL${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PLUGINS=fl
unset VLLM_FL_USE_FLAGGEMS_ATTN VLLM_ATTENTION_BACKEND VLLM_BATCH_INVARIANT
set -o pipefail
python3 -c 'import flag_gems, vllm_fl; print(flag_gems.__file__); print(vllm_fl.__file__)'
vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 2>&1 | tee "$serve_log"
```

Verify both source imports are from the selected version. In v13, the server log on BI-V150 must report the shared `vllm_fl.dispatch.backends.triton_attention.OptimizedTritonAttentionBackend` when the framework selects stock Triton. Confirm kernel execution through profiling when possible; backend selection alone is not proof of workload specialization.

## 3. Unmodified benchmark (Terminal B)

For development half-runs use the supplied external half-run script, unchanged. Use the same version label as the running server. For the final full-run comparison use the supplied `benchmark_throughput_serve.py` instead, with the same cases and original settings, for both versions.

```bash
version=v13
curl -f http://127.0.0.1:9031/health || exit 1
cd "/root/Optimization/$version"
mkdir -p .autoresearch-work/bench_runs
run_dir=$(mktemp -d "$PWD/.autoresearch-work/bench_runs/run_XXXXXXXX")
cd "$run_dir"
benchmark=/workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve_half.py
sha256sum "$benchmark" > benchmark_sha256.txt
set -o pipefail
python3 "$benchmark" \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]' \
  2>&1 | tee benchmark.log
```

Verify all repetitions succeeded with 256/256 requests for 4K and 128/128 for 16K. Retain raw CSVs, benchmark hashes, model/environment provenance, server logs and both throughput/latency results. Use matched repeat/interleaved runs to distinguish regressions from run-to-run variability. Historical v10 results are not a substitute for a fresh control.

## 4. Accuracy and portability

Run the unchanged competition MATH-500 Level 3 evaluation using the original evaluator configuration. Require the competition accuracy threshold. Test additional compatible GPU platforms separately with actual backend selection and numerical correctness; leave any alternative framework-selected attention backend untouched. Do not treat a mocked dispatch test as hardware validation.
