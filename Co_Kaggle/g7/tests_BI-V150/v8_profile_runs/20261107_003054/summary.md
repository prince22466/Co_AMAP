# v8 BI-V150 profiling — diagnostic report

**Important:** This is NOT an official competition benchmark result. The benchmark driver was not modified; trace windows can perturb throughput.

## Workloads (unchanged shapes/concurrency, one diagnostic invocation each)

- **16k**: 16384 input + 1024 output, C=64, N=128; 1876.4 s; complete=True; trace windows=early, late, middle. See `bench_16k.log`.
- **4k**: 4096 input + 1024 output, C=64, N=256; 696.6 s; complete=True; trace windows=early, middle. See `bench_4k.log`.

## 16K/4K KV-cache and scheduler telemetry

### startup (1 samples, errors=0)
- kv_cache_usage: {"min": 0.0, "max": 0.0, "last": 0.0}
- running_requests: {"min": 0.0, "max": 0.0, "last": 0.0}
- waiting_requests: {"min": 0.0, "max": 0.0, "last": 0.0}
- preemptions_total: {"min": 0.0, "max": 0.0, "last": 0.0, "delta": 0}
- ixsmi: {"gpu_util_samples": 1, "gpu_util_mean_percent": 0, "gpu_util_peak_percent": 0}

### 16k (367 samples, errors=0)
- kv_cache_usage: {"min": 0.0, "max": 0.9998160637645617, "last": 0.0}
- running_requests: {"min": 0.0, "max": 31.0, "last": 0.0}
- waiting_requests: {"min": 0.0, "max": 63.0, "last": 0.0}
- preemptions_total: {"min": 0.0, "max": 4.0, "last": 4.0, "delta": 4.0}
- ixsmi: {"gpu_util_samples": 367, "gpu_util_mean_percent": 91.8, "gpu_util_peak_percent": 100}

### 4k (136 samples, errors=0)
- kv_cache_usage: {"min": 0.0, "max": 0.6207234825260577, "last": 0.0}
- running_requests: {"min": 0.0, "max": 64.0, "last": 0.0}
- waiting_requests: {"min": 0.0, "max": 60.0, "last": 0.0}
- preemptions_total: {"min": 4.0, "max": 4.0, "last": 4.0, "delta": 0}
- ixsmi: {"gpu_util_samples": 136, "gpu_util_mean_percent": 76.5, "gpu_util_peak_percent": 100}

## Worker GPU kernel traces (approximate capture-label match)

### 84d64a9dab4f_3770.async_llm.1793982903406427434.pt.trace.json.gz (16k_early)
- GPU kernel events: **0**; summed GPU kernel durations: **0.0 ms**
- First-to-last GPU event span: **0.0 ms**, GPU kernel/transfer busy union in that span: **0.0 ms**, gaps within those events **0.0 ms**
- GPU memcpy/memset events: 0; summed 0.0 ms. No events means unavailable or absent transfers in the trace, not zero HBM traffic.
Top named kernels:

### 84d64a9dab4f_3770.async_llm.1793983302168483038.pt.trace.json.gz (16k_middle)
- GPU kernel events: **0**; summed GPU kernel durations: **0.0 ms**
- First-to-last GPU event span: **0.0 ms**, GPU kernel/transfer busy union in that span: **0.0 ms**, gaps within those events **0.0 ms**
- GPU memcpy/memset events: 0; summed 0.0 ms. No events means unavailable or absent transfers in the trace, not zero HBM traffic.
Top named kernels:

### 84d64a9dab4f_3770.async_llm.1793984102288839483.pt.trace.json.gz (16k_late)
- GPU kernel events: **0**; summed GPU kernel durations: **0.0 ms**
- First-to-last GPU event span: **0.0 ms**, GPU kernel/transfer busy union in that span: **0.0 ms**, gaps within those events **0.0 ms**
- GPU memcpy/memset events: 0; summed 0.0 ms. No events means unavailable or absent transfers in the trace, not zero HBM traffic.
Top named kernels:

### 84d64a9dab4f_3770.async_llm.1793984768243673026.pt.trace.json.gz (4k_early)
- GPU kernel events: **0**; summed GPU kernel durations: **0.0 ms**
- First-to-last GPU event span: **0.0 ms**, GPU kernel/transfer busy union in that span: **0.0 ms**, gaps within those events **0.0 ms**
- GPU memcpy/memset events: 0; summed 0.0 ms. No events means unavailable or absent transfers in the trace, not zero HBM traffic.
Top named kernels:

### 84d64a9dab4f_3770.async_llm.1793984959129279737.pt.trace.json.gz (4k_middle)
- GPU kernel events: **0**; summed GPU kernel durations: **0.0 ms**
- First-to-last GPU event span: **0.0 ms**, GPU kernel/transfer busy union in that span: **0.0 ms**, gaps within those events **0.0 ms**
- GPU memcpy/memset events: 0; summed 0.0 ms. No events means unavailable or absent transfers in the trace, not zero HBM traffic.
Top named kernels:

### rank0.1793982904160267081.pt.trace.json.gz (16k_early)
- GPU kernel events: **2844**; summed GPU kernel durations: **6655.507 ms**
- First-to-last GPU event span: **6700.654 ms**, GPU kernel/transfer busy union in that span: **6655.703 ms**, gaps within those events **44.951 ms**
- Attention: 5870.571 ms (88.2% summed kernel time)
- GEMM / Linear: 678.985 ms (10.2% summed kernel time)
- Activation: 34.645 ms (0.5% summed kernel time)
- Normalization: 33.849 ms (0.5% summed kernel time)
- Other: 29.799 ms (0.4% summed kernel time)
- KV cache op: 7.659 ms (0.1% summed kernel time)
- GPU memcpy/memset events: 54; summed 0.195 ms. No events means unavailable or absent transfers in the trace, not zero HBM traffic.
Top named kernels:
  - kernel_unified_attention: 5870.571 ms
  - mm_kernel: 673.922 ms
  - triton_poi_fused_mul_silu_slice_1: 34.645 ms
  - triton_red_fused_fused_add_rms_norm_2: 20.893 ms
  - triton_poi_fused_3: 20.185 ms
  - triton_red_fused_fused_add_rms_norm_0: 12.698 ms
  - reshape_and_cache_kernel_flash: 7.659 ms
CPU related scopes (nested, nonexclusive):
  - execute_new_0_cached_2: 4811.173 ms
  - vllm::unified_kv_cache_update: 659.051 ms
  - cudaMemcpyAsync: 389.045 ms
  - cudaDeviceSynchronize: 329.356 ms
  - AOTDispatcher Runtime Wrapper Prologue: 1.567 ms

### rank0.1793983304312356150.pt.trace.json.gz (16k_middle)
- GPU kernel events: **1326**; summed GPU kernel durations: **5362.691 ms**
- First-to-last GPU event span: **7280.814 ms**, GPU kernel/transfer busy union in that span: **5362.888 ms**, gaps within those events **1917.926 ms**
- Attention: 4980.089 ms (92.9% summed kernel time)
- GEMM / Linear: 330.142 ms (6.2% summed kernel time)
- Other: 17.287 ms (0.3% summed kernel time)
- Normalization: 15.925 ms (0.3% summed kernel time)
- Activation: 15.624 ms (0.3% summed kernel time)
- KV cache op: 3.624 ms (0.1% summed kernel time)
- GPU memcpy/memset events: 27; summed 0.196 ms. No events means unavailable or absent transfers in the trace, not zero HBM traffic.
Top named kernels:
  - kernel_unified_attention: 4980.089 ms
  - mm_kernel: 327.28 ms
  - triton_poi_fused_mul_silu_slice_1: 15.624 ms
  - triton_red_fused_fused_add_rms_norm_2: 9.998 ms
  - triton_poi_fused_3: 9.395 ms
  - _topk_topp_kernel: 5.957 ms
  - triton_red_fused_fused_add_rms_norm_0: 5.808 ms
CPU related scopes (nested, nonexclusive):
  - execute_new_0_cached_30: 8085.203 ms
  - vllm::unified_kv_cache_update: 639.078 ms
  - cudaMemcpyAsync: 87.215 ms
  - cudaDeviceSynchronize: 1.953 ms
  - AOTDispatcher Runtime Wrapper Prologue: 0.857 ms

### rank0.1793984102528614736.pt.trace.json.gz (16k_late)
- GPU kernel events: **1128**; summed GPU kernel durations: **216.492 ms**
- First-to-last GPU event span: **6085.729 ms**, GPU kernel/transfer busy union in that span: **219.294 ms**, gaps within those events **5866.435 ms**
- Other: 172.702 ms (79.8% summed kernel time)
- GEMM / Linear: 43.79 ms (20.2% summed kernel time)
- GPU memcpy/memset events: 376; summed 2.802 ms. No events means unavailable or absent transfers in the trace, not zero HBM traffic.
Top named kernels:
  - _topk_topp_kernel: 145.954 ms
  - linear_kernel: 43.79 ms
  - true_div_func_kernel_rank_2: 7.866 ms
  - true_div_func_kernel_rank_1: 4.572 ms
  - softmax_kernel_inner: 3.738 ms
  - argmax_kernel_inner: 3.263 ms
  - fused_exponential_kernel_f32_unroll8: 2.312 ms
CPU related scopes (nested, nonexclusive):
  - execute_new_0_cached_30: 266.484 ms
  - cudaDeviceSynchronize: 129.238 ms
  - cudaMemcpyAsync: 2.326 ms
  - cudaGraphLaunch: 2.065 ms

### rank0.1793984768855490624.pt.trace.json.gz (4k_early)
- GPU kernel events: **6630**; summed GPU kernel durations: **6176.305 ms**
- First-to-last GPU event span: **6227.885 ms**, GPU kernel/transfer busy union in that span: **6177.197 ms**, gaps within those events **50.688 ms**
- Attention: 4180.628 ms (67.7% summed kernel time)
- GEMM / Linear: 1699.897 ms (27.5% summed kernel time)
- Other: 106.312 ms (1.7% summed kernel time)
- Activation: 86.197 ms (1.4% summed kernel time)
- Normalization: 84.818 ms (1.4% summed kernel time)
- KV cache op: 18.453 ms (0.3% summed kernel time)
- GPU memcpy/memset events: 149; summed 0.892 ms. No events means unavailable or absent transfers in the trace, not zero HBM traffic.
Top named kernels:
  - kernel_unified_attention: 4180.628 ms
  - mm_kernel: 1685.719 ms
  - triton_poi_fused_mul_silu_slice_1: 86.197 ms
  - triton_red_fused_fused_add_rms_norm_2: 52.748 ms
  - triton_poi_fused_3: 50.058 ms
  - _topk_topp_kernel: 48.297 ms
  - triton_red_fused_fused_add_rms_norm_0: 31.382 ms
CPU related scopes (nested, nonexclusive):
  - vllm::unified_kv_cache_update: 724.427 ms
  - execute_new_0_cached_20: 528.539 ms
  - execute_new_0_cached_25: 486.539 ms
  - cudaMemcpyAsync: 421.858 ms
  - execute_new_1_cached_24: 418.997 ms
  - execute_new_1_cached_22: 418.055 ms

### rank0.1793984959342351066.pt.trace.json.gz (4k_middle)
- GPU kernel events: **3370**; summed GPU kernel durations: **3395.321 ms**
- First-to-last GPU event span: **5912.632 ms**, GPU kernel/transfer busy union in that span: **3397.379 ms**, gaps within those events **2515.253 ms**
- Attention: 2326.464 ms (68.5% summed kernel time)
- GEMM / Linear: 808.882 ms (23.8% summed kernel time)
- Other: 172.689 ms (5.1% summed kernel time)
- Activation: 39.635 ms (1.2% summed kernel time)
- Normalization: 38.979 ms (1.1% summed kernel time)
- KV cache op: 8.673 ms (0.3% summed kernel time)
- GPU memcpy/memset events: 179; summed 2.058 ms. No events means unavailable or absent transfers in the trace, not zero HBM traffic.
Top named kernels:
  - kernel_unified_attention: 2326.464 ms
  - mm_kernel: 786.336 ms
  - _topk_topp_kernel: 128.012 ms
  - triton_poi_fused_mul_silu_slice_1: 39.635 ms
  - triton_red_fused_fused_add_rms_norm_2: 24.115 ms
  - triton_poi_fused_3: 23.203 ms
  - linear_kernel: 22.546 ms
CPU related scopes (nested, nonexclusive):
  - execute_new_0_cached_64: 2773.964 ms
  - execute_new_1_cached_61: 498.186 ms
  - execute_new_1_cached_63: 484.714 ms
  - vllm::unified_kv_cache_update: 430.735 ms
  - execute_new_1_cached_62: 429.272 ms
  - execute_new_0_cached_62: 412.282 ms

## Decision / next experiment

- Compare 16K early/middle/late worker traces, and overlay **time-aligned** running, waiting, KV-cache, preemption metrics.
- If the GPU is mostly active with attention kernels, collect BI-V150 vendor memory bandwidth/occupancy counters for that exact kernel and test new query/KV reuse tiling.
- If there are large GPU idle gaps with runnable requests, inspect CPU scheduler/engine-core scopes and synchronization before changing attention.
- If KV usage is near 100% with preemptions/waiting, quantify recompute/preemption overhead and the cost of contention.
- Compare to v8 known total throughput **2331.29 (4K)**, **1251.22 (16K)** only in a separate profiler-off **unchanged official** run.
- A ≥20% throughput improvement requires ≥16.67% reduction in end-to-end elapsed time for fixed work. Kernel-time shares in this report are **not** wall-clock shares.
- Reported GPU event gaps are **only within the observed trace event span**; they are not a GPU-idle fraction of the full request/serving runtime. Also, memory stalls inside active kernels do not appear as idle gaps.
- Basic `ixsmi` utilization is not device bandwidth, cache hit rate, occupancy, or memory-stall data. Collect these via a supported BI-V150 native profiler only after verifying its CLI capabilities.
- Distinguish `async_llm` CPU-only traces from rank-0 worker GPU traces. A zero-GPU-event trace is not evidence that the GPU was idle.

## Output artifact descriptions

- `environment.json`, `server.log`: runtime and v8 provenance.
- `metrics.jsonl`: sampled vLLM /metrics (request queue, KV pressure, preemptions).
- `gpu_smi.jsonl`: sampled BI-V150 utilization/VRAM; `cpu_ps.jsonl`: sampled process CPU/RSS.
- `profile_windows.jsonl` and `torch_traces/`: bounded profiler windows and worker traces.
- `analysis.json`: structured analysis, with omissions explicitly visible.
