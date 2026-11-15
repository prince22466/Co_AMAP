# v14 profiling — diagnostic only

These runs are not official scores. Kernel shares use summed observed GPU kernel durations; trace gaps are not whole-run GPU idle time. CPU scopes overlap. Missing data are not zeros.

## 4K
- Completed requests: 256.0 / 256
- Diagnostic throughput: 2310.67 tok/s (historical v14 official: 2817.15)
- Mean TTFT / TPOT: 5379.12 / 125.15 ms
- P99 TTFT / ITL: 31419.38 / 415.16 ms
- Metrics samples: 120; errors: 0; ixsmi errors: 0/120

- GPU telemetry: {"parsed_samples": 120, "gpu_util_sample_mean_pct": 75.05, "gpu_util_peak_pct": 100, "vram_used_peak_mib": 28432, "vram_total_mib": [32768]}

Key telemetry (each series shown separately):

- `vllm:kv_cache_usage_perc{engine="0",model_name="minicpm"}`: {"samples": 120, "minimum": 0.0, "maximum": 0.6208376770711964, "sample_mean": 0.48986887021933323, "last": 0.0}
- `vllm:num_preemptions_created{engine="0",model_name="minicpm"}`: {"samples": 120, "minimum": 1794759706.6125348, "maximum": 1794759706.6125348, "sample_mean": 1794759706.6125348, "last": 1794759706.6125348}
- `vllm:num_preemptions_total{engine="0",model_name="minicpm"}`: {"samples": 120, "minimum": 4.0, "maximum": 4.0, "sample_mean": 4.0, "last": 4.0, "delta": 0.0}
- `vllm:num_requests_running{engine="0",model_name="minicpm"}`: {"samples": 120, "minimum": 0.0, "maximum": 64.0, "sample_mean": 55.03333333333333, "last": 0.0}
- `vllm:num_requests_waiting{engine="0",model_name="minicpm"}`: {"samples": 120, "minimum": 0.0, "maximum": 52.0, "sample_mean": 1.425, "last": 0.0}
- `vllm:num_requests_waiting_by_reason{engine="0",model_name="minicpm",reason="capacity"}`: {"samples": 120, "minimum": 0.0, "maximum": 52.0, "sample_mean": 1.425, "last": 0.0}
- `vllm:num_requests_waiting_by_reason{engine="0",model_name="minicpm",reason="deferred"}`: {"samples": 120, "minimum": 0.0, "maximum": 0.0, "sample_mean": 0.0, "last": 0.0}

Histogram sums/counts and prefix-cache counters: see `analysis.json` (per label, per workload delta). Raw device utilization is in `gpu_smi.jsonl`; single-GPU BI-V150 table values are parsed when recognized. No bandwidth/occupancy inference is made from ixsmi.

## 16K
- Completed requests: 128.0 / 128
- Diagnostic throughput: 1954.0 tok/s (historical v14 official: 2073.93)
- Mean TTFT / TPOT: 254594.46 / 232.26 ms
- P99 TTFT / ITL: 518313.6 / 971.99 ms
- Metrics samples: 235; errors: 0; ixsmi errors: 0/235

- GPU telemetry: {"parsed_samples": 235, "gpu_util_sample_mean_pct": 88.40851063829787, "gpu_util_peak_pct": 100, "vram_used_peak_mib": 28432, "vram_total_mib": [32768]}

Key telemetry (each series shown separately):

- `vllm:kv_cache_usage_perc{engine="0",model_name="minicpm"}`: {"samples": 235, "minimum": 0.0, "maximum": 0.9986202244434905, "sample_mean": 0.8436935449989497, "last": 0.10004905868645364}
- `vllm:num_preemptions_created{engine="0",model_name="minicpm"}`: {"samples": 235, "minimum": 1794759706.6125348, "maximum": 1794759706.6125348, "sample_mean": 1794759706.6125348, "last": 1794759706.6125348}
- `vllm:num_preemptions_total{engine="0",model_name="minicpm"}`: {"samples": 235, "minimum": 0.0, "maximum": 4.0, "sample_mean": 1.9319148936170212, "last": 4.0, "delta": 4.0}
- `vllm:num_requests_running{engine="0",model_name="minicpm"}`: {"samples": 235, "minimum": 0.0, "maximum": 31.0, "sample_mean": 26.327659574468086, "last": 3.0}
- `vllm:num_requests_waiting{engine="0",model_name="minicpm"}`: {"samples": 235, "minimum": 0.0, "maximum": 63.0, "sample_mean": 26.82127659574468, "last": 0.0}
- `vllm:num_requests_waiting_by_reason{engine="0",model_name="minicpm",reason="capacity"}`: {"samples": 235, "minimum": 0.0, "maximum": 63.0, "sample_mean": 26.82127659574468, "last": 0.0}
- `vllm:num_requests_waiting_by_reason{engine="0",model_name="minicpm",reason="deferred"}`: {"samples": 235, "minimum": 0.0, "maximum": 0.0, "sample_mean": 0.0, "last": 0.0}

Histogram sums/counts and prefix-cache counters: see `analysis.json` (per label, per workload delta). Raw device utilization is in `gpu_smi.jsonl`; single-GPU BI-V150 table values are parsed when recognized. No bandwidth/occupancy inference is made from ixsmi.

## Traces
No trace files found. Expected in telemetry mode; otherwise inspect capture errors.


## Next decisions
- If KV usage saturates with waiting/preemptions, quantify queue and recompute overhead.
- If attention dominates valid worker traces, profile exact prefill/mixed/decode shapes with the native BI-V150 profiler.
- If GEMM share grows, prioritize representative shape probes.
- If graph visibility is incomplete, obtain a native timeline before interpreting gaps.
- Confirm any change with profiler-off official 4K/16K and MATH-500.
