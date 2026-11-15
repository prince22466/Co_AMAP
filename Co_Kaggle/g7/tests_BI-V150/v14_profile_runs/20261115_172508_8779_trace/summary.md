# v14 profiling — diagnostic only

These runs are not official scores. Kernel shares use summed observed GPU kernel durations; trace gaps are not whole-run GPU idle time. CPU scopes overlap. Missing data are not zeros.

## 4K
- Completed requests: 256.0 / 256
- Diagnostic throughput: 2221.29 tok/s (historical v14 official: 2817.15)
- Mean TTFT / TPOT: 6989.52 / 128.74 ms
- P99 TTFT / ITL: 43180.43 / 416.17 ms
- Metrics samples: 124; errors: 0; ixsmi errors: 0/124

- GPU telemetry: {"parsed_samples": 124, "gpu_util_sample_mean_pct": 71.41129032258064, "gpu_util_peak_pct": 100, "vram_used_peak_mib": 28436, "vram_total_mib": [32768]}

Key telemetry (each series shown separately):

- `vllm:kv_cache_usage_perc{engine="0",model_name="minicpm"}`: {"samples": 124, "minimum": 0.0, "maximum": 0.6208376770711964, "sample_mean": 0.4860019544347668, "last": 0.0}
- `vllm:num_preemptions_created{engine="0",model_name="minicpm"}`: {"samples": 124, "minimum": 1794763669.0542495, "maximum": 1794763669.0542495, "sample_mean": 1794763669.0542495, "last": 1794763669.0542495}
- `vllm:num_preemptions_total{engine="0",model_name="minicpm"}`: {"samples": 124, "minimum": 4.0, "maximum": 4.0, "sample_mean": 4.0, "last": 4.0, "delta": 0.0}
- `vllm:num_requests_running{engine="0",model_name="minicpm"}`: {"samples": 124, "minimum": 0.0, "maximum": 64.0, "sample_mean": 54.733870967741936, "last": 0.0}
- `vllm:num_requests_waiting{engine="0",model_name="minicpm"}`: {"samples": 124, "minimum": 0.0, "maximum": 59.0, "sample_mean": 2.338709677419355, "last": 0.0}
- `vllm:num_requests_waiting_by_reason{engine="0",model_name="minicpm",reason="capacity"}`: {"samples": 124, "minimum": 0.0, "maximum": 59.0, "sample_mean": 2.338709677419355, "last": 0.0}
- `vllm:num_requests_waiting_by_reason{engine="0",model_name="minicpm",reason="deferred"}`: {"samples": 124, "minimum": 0.0, "maximum": 0.0, "sample_mean": 0.0, "last": 0.0}

Histogram sums/counts and prefix-cache counters: see `analysis.json` (per label, per workload delta). Raw device utilization is in `gpu_smi.jsonl`; single-GPU BI-V150 table values are parsed when recognized. No bandwidth/occupancy inference is made from ixsmi.

## 16K
- Completed requests: 128.0 / 128
- Diagnostic throughput: 1932.24 tok/s (historical v14 official: 2073.93)
- Mean TTFT / TPOT: 259676.66 / 234.19 ms
- P99 TTFT / ITL: 529657.64 / 974.53 ms
- Metrics samples: 238; errors: 0; ixsmi errors: 0/239

- GPU telemetry: {"parsed_samples": 239, "gpu_util_sample_mean_pct": 86.48535564853556, "gpu_util_peak_pct": 100, "vram_used_peak_mib": 28436, "vram_total_mib": [32768]}

Key telemetry (each series shown separately):

- `vllm:kv_cache_usage_perc{engine="0",model_name="minicpm"}`: {"samples": 238, "minimum": 0.0, "maximum": 0.9998466916048323, "sample_mean": 0.8407389876904954, "last": 0.0}
- `vllm:num_preemptions_created{engine="0",model_name="minicpm"}`: {"samples": 238, "minimum": 1794763669.0542495, "maximum": 1794763669.0542495, "sample_mean": 1794763669.0542495, "last": 1794763669.0542495}
- `vllm:num_preemptions_total{engine="0",model_name="minicpm"}`: {"samples": 238, "minimum": 0.0, "maximum": 4.0, "sample_mean": 1.915966386554622, "last": 4.0, "delta": 4.0}
- `vllm:num_requests_running{engine="0",model_name="minicpm"}`: {"samples": 238, "minimum": 0.0, "maximum": 31.0, "sample_mean": 26.201680672268907, "last": 0.0}
- `vllm:num_requests_waiting{engine="0",model_name="minicpm"}`: {"samples": 238, "minimum": 0.0, "maximum": 63.0, "sample_mean": 27.058823529411764, "last": 0.0}
- `vllm:num_requests_waiting_by_reason{engine="0",model_name="minicpm",reason="capacity"}`: {"samples": 238, "minimum": 0.0, "maximum": 63.0, "sample_mean": 27.058823529411764, "last": 0.0}
- `vllm:num_requests_waiting_by_reason{engine="0",model_name="minicpm",reason="deferred"}`: {"samples": 238, "minimum": 0.0, "maximum": 0.0, "sample_mean": 0.0, "last": 0.0}

Histogram sums/counts and prefix-cache counters: see `analysis.json` (per label, per workload delta). Raw device utilization is in `gpu_smi.jsonl`; single-GPU BI-V150 table values are parsed when recognized. No bandwidth/occupancy inference is made from ixsmi.

### 16k_early — a31e4f45e4e3_8786.async_llm.1794763727832566421.pt.trace.json.gz
- GPU kernel events: 0; summed kernel time: 0.00 ms
- Observed GPU span/busy/gap: 0.00 / 0.00 / 0.00 ms
- WARNING: No GPU kernels recorded; not proof of GPU idleness.
- Time-aligned metrics samples: 3 (see analysis.json)

### 16k_middle — a31e4f45e4e3_8786.async_llm.1794764032097655550.pt.trace.json.gz
- GPU kernel events: 0; summed kernel time: 0.00 ms
- Observed GPU span/busy/gap: 0.00 / 0.00 / 0.00 ms
- WARNING: No GPU kernels recorded; not proof of GPU idleness.
- Time-aligned metrics samples: 3 (see analysis.json)

### 16k_late — a31e4f45e4e3_8786.async_llm.1794764481178008683.pt.trace.json.gz
- GPU kernel events: 0; summed kernel time: 0.00 ms
- Observed GPU span/busy/gap: 0.00 / 0.00 / 0.00 ms
- WARNING: No GPU kernels recorded; not proof of GPU idleness.
- Time-aligned metrics samples: 2 (see analysis.json)

### 4k_early — a31e4f45e4e3_8786.async_llm.1794764903204815053.pt.trace.json.gz
- GPU kernel events: 0; summed kernel time: 0.00 ms
- Observed GPU span/busy/gap: 0.00 / 0.00 / 0.00 ms
- WARNING: No GPU kernels recorded; not proof of GPU idleness.
- Time-aligned metrics samples: 4 (see analysis.json)

### 4k_middle — a31e4f45e4e3_8786.async_llm.1794765023413164316.pt.trace.json.gz
- GPU kernel events: 0; summed kernel time: 0.00 ms
- Observed GPU span/busy/gap: 0.00 / 0.00 / 0.00 ms
- WARNING: No GPU kernels recorded; not proof of GPU idleness.
- Time-aligned metrics samples: 3 (see analysis.json)

### 4k_late — a31e4f45e4e3_8786.async_llm.1794765203084435032.pt.trace.json.gz
- GPU kernel events: 0; summed kernel time: 0.00 ms
- Observed GPU span/busy/gap: 0.00 / 0.00 / 0.00 ms
- WARNING: No GPU kernels recorded; not proof of GPU idleness.
- Time-aligned metrics samples: 3 (see analysis.json)

### 16k_early — rank0.1794763728553639627.pt.trace.json.gz
- GPU kernel events: 8532; summed kernel time: 8737.94 ms
- Observed GPU span/busy/gap: 10551.34 / 8738.55 / 1812.78 ms
- Attention: 73.0% of summed observed kernel time
- GEMM / Linear: 23.3% of summed observed kernel time
- Activation: 1.2% of summed observed kernel time
- Normalization: 1.2% of summed observed kernel time
- Other: 1.1% of summed observed kernel time
- KV cache op: 0.3% of summed observed kernel time
- Sampling: 0.0% of summed observed kernel time
  - `kernel_unified_attention`: 6381.94 ms
  - `mm_kernel`: 2019.75 ms
  - `triton_poi_fused_mul_silu_slice_1`: 103.78 ms
  - `triton_red_fused_fused_add_rms_norm_2`: 62.69 ms
  - `triton_poi_fused_3`: 60.47 ms
- Time-aligned metrics samples: 3 (see analysis.json)

### 16k_middle — rank0.1794764032881062638.pt.trace.json.gz
- GPU kernel events: 7521; summed kernel time: 9991.22 ms
- Observed GPU span/busy/gap: 10051.59 / 9992.39 / 59.20 ms
- Attention: 77.4% of summed observed kernel time
- GEMM / Linear: 19.2% of summed observed kernel time
- Activation: 1.0% of summed observed kernel time
- Normalization: 1.0% of summed observed kernel time
- Other: 0.7% of summed observed kernel time
- Sampling: 0.6% of summed observed kernel time
- KV cache op: 0.2% of summed observed kernel time
  - `kernel_unified_attention`: 7733.24 ms
  - `mm_kernel`: 1906.62 ms
  - `triton_poi_fused_mul_silu_slice_1`: 97.54 ms
  - `triton_red_fused_fused_add_rms_norm_2`: 59.49 ms
  - `triton_poi_fused_3`: 56.68 ms
- Time-aligned metrics samples: 3 (see analysis.json)

### 16k_late — rank0.1794764481553818439.pt.trace.json.gz
- GPU kernel events: 1872; summed kernel time: 346.46 ms
- Observed GPU span/busy/gap: 10223.18 / 351.10 / 9872.08 ms
- WARNING: Graph launches present; verify graph kernel visibility before interpreting gaps.
- Sampling: 67.8% of summed observed kernel time
- GEMM / Linear: 21.0% of summed observed kernel time
- Other: 11.2% of summed observed kernel time
  - `_topk_topp_kernel`: 229.38 ms
  - `linear_kernel`: 72.71 ms
  - `true_div_func_kernel_rank_2`: 13.05 ms
  - `true_div_func_kernel_rank_1`: 7.57 ms
  - `softmax_kernel_inner`: 6.23 ms
- Time-aligned metrics samples: 2 (see analysis.json)

### 4k_early — rank0.1794764904048486655.pt.trace.json.gz
- GPU kernel events: 18564; summed kernel time: 10115.62 ms
- Observed GPU span/busy/gap: 10265.40 / 10117.99 / 147.41 ms
- GEMM / Linear: 47.0% of summed observed kernel time
- Attention: 45.1% of summed observed kernel time
- Activation: 2.4% of summed observed kernel time
- Normalization: 2.3% of summed observed kernel time
- Other: 1.6% of summed observed kernel time
- Sampling: 1.1% of summed observed kernel time
- KV cache op: 0.5% of summed observed kernel time
  - `mm_kernel`: 4715.10 ms
  - `kernel_unified_attention`: 4558.75 ms
  - `triton_poi_fused_mul_silu_slice_1`: 241.05 ms
  - `triton_red_fused_fused_add_rms_norm_2`: 146.44 ms
  - `triton_poi_fused_3`: 140.03 ms
- Time-aligned metrics samples: 4 (see analysis.json)

### 4k_middle — rank0.1794765024029783163.pt.trace.json.gz
- GPU kernel events: 6695; summed kernel time: 4177.23 ms
- Observed GPU span/busy/gap: 10382.98 / 4181.10 / 6201.88 ms
- WARNING: Graph launches present; verify graph kernel visibility before interpreting gaps.
- Attention: 51.6% of summed observed kernel time
- GEMM / Linear: 38.6% of summed observed kernel time
- Sampling: 3.6% of summed observed kernel time
- Other: 2.0% of summed observed kernel time
- Activation: 1.9% of summed observed kernel time
- Normalization: 1.9% of summed observed kernel time
- KV cache op: 0.4% of summed observed kernel time
  - `kernel_unified_attention`: 2153.45 ms
  - `mm_kernel`: 1570.81 ms
  - `_topk_topp_kernel`: 148.35 ms
  - `triton_poi_fused_mul_silu_slice_1`: 80.08 ms
  - `triton_red_fused_fused_add_rms_norm_2`: 49.15 ms
- Time-aligned metrics samples: 3 (see analysis.json)

### 4k_late — rank0.1794765203385602269.pt.trace.json.gz
- GPU kernel events: 2604; summed kernel time: 1053.50 ms
- Observed GPU span/busy/gap: 10133.93 / 1065.80 / 9068.13 ms
- WARNING: Graph launches present; verify graph kernel visibility before interpreting gaps.
- Sampling: 75.5% of summed observed kernel time
- GEMM / Linear: 13.3% of summed observed kernel time
- Other: 11.3% of summed observed kernel time
  - `_topk_topp_kernel`: 784.36 ms
  - `linear_kernel`: 139.70 ms
  - `true_div_func_kernel_rank_2`: 43.39 ms
  - `true_div_func_kernel_rank_1`: 23.60 ms
  - `softmax_kernel_inner`: 21.89 ms
- Time-aligned metrics samples: 3 (see analysis.json)

## Capture integrity
- 16k_early: start=True, stop=True, files=3, errors=[]
- 16k_middle: start=True, stop=True, files=2, errors=[]
- 16k_late: start=True, stop=True, files=2, errors=[]
- 4k_early: start=True, stop=True, files=2, errors=[]
- 4k_middle: start=True, stop=True, files=2, errors=[]
- 4k_late: start=True, stop=True, files=2, errors=[]

## Next decisions
- If KV usage saturates with waiting/preemptions, quantify queue and recompute overhead.
- If attention dominates valid worker traces, profile exact prefill/mixed/decode shapes with the native BI-V150 profiler.
- If GEMM share grows, prioritize representative shape probes.
- If graph visibility is incomplete, obtain a native timeline before interpreting gaps.
- Confirm any change with profiler-off official 4K/16K and MATH-500.
