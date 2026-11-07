# MiniCPM5-2B / BI-V150 — FlagOS S2 submission reproduction guide (v8)

> **DRAFT, 2026-11-06.** This file is intended to become the submitted archive's `readme.md`. Commands and checkpoints below use the organizer's specified workload. Do not use historical self-run measurements as official results. Replace `<TEAM_NAME>` and attach the final successful logs before submission.

## 1. Submission and scope

The archive root contains `vllm-plugin-FL/` (framework and compile scripts), `FlagGems/` (source and build scripts), `readme.md` (this file), and `report.pdf` (technical report). The source is drawn from the checked-in `Optimization/v8` snapshot, a consolidation of the v1 prefill attention and v5 decode attention kernel optimizations.

Target: **Iluvatar BI-V150 (32 GiB)** running **MiniCPM5-2B only**. No MuXi/C500 results or optimizations are claimed. Reproduction uses the official model and dataset paths, serving parameters, MATH-500 Level 3 evaluation and two benchmark scenarios. Optimization changes are scoped to the original unified Triton attention operator and its BI-V150 backend integration; no quantization, speculative decoding, or changes to the official benchmark are required.

## 2. Environment

Organizer image paths:

```text
/workspace/MiniCPM5-2B
/workspace/evalscope-datasets/math_500
/workspace/vllm-plugin-FL
/workspace/FlagGems
```

One captured development-machine snapshot (2026-10-25; **recheck on final submission host**):

| Component | Recorded version/value |
|---|---|
| GPU | Iluvatar BI-V150, 32 GiB |
| Linux | Ubuntu 24.04.2 LTS, kernel 5.4.0-148-generic |
| Python | 3.12.11 |
| vLLM | 0.24.0+empty |
| vllm-plugin-fl | 0.3.0rc0+g13eb9be69 |
| FlagGems | 5.3.5 |
| torch | 2.10.0+corex.4.5.0 |
| flagtree | 0.6.2a3+iluvatar3.6 |
| driver | 4.5.0 |

The framework required by the event is `flagos-ai/vllm-plugin-FL` at `flagos-2026-s2` and FlagGems `v5.3.5`. The final archive's precise source revisions must be recorded here after the branch/tag provenance is verified.

## 3. Install source from the submitted ZIP

Unpack the submitted folder so the two source directories become the official paths above (or consistently substitute their actual absolute paths). In the competition environment:

```bash
cd /workspace

python3 -m pip install --no-build-isolation -e /workspace/FlagGems
python3 -m pip install --no-build-isolation -e /workspace/vllm-plugin-FL

export VLLM_PLUGINS=fl
export PYTHONPATH="/workspace/FlagGems/src:/workspace/vllm-plugin-FL:$PYTHONPATH"

python3 - <<'PY'
import inspect
import torch
import vllm, vllm_fl, flag_gems
from vllm_fl.dispatch.backends.vendor.iluvatar.impl import attention_biv150
from vllm_fl.dispatch.backends.vendor.iluvatar.impl import triton_unified_attention_biv150
print("GPU:", torch.cuda.get_device_name(0))
for m in (vllm, vllm_fl, flag_gems, attention_biv150,
          triton_unified_attention_biv150):
    print(m.__name__, inspect.getfile(m))
PY
```

Check that the two BI-V150 attention modules resolve to the installed submission and that the serving/engine process uses that same package; an import check from a shell alone is insufficient. Keep the installation log. **Do not change serving arguments** to obtain speedups.

## 4. Functional tests

Before serving, run the two v8 functional suites (assuming repository's tests remain packaged):

```bash
cd /workspace
export VLLM_PLUGINS=fl
export PYTHONPATH="/workspace/FlagGems/src:/workspace/vllm-plugin-FL:$PYTHONPATH"

python3 -m pytest -q -s --noconftest \
  /workspace/vllm-plugin-FL/tests/functional_tests/test_biv150_attention.py \
  /workspace/vllm-plugin-FL/tests/functional_tests/test_biv150_decode.py
```

Check the exit status, failures and warning output. These tests establish functional behavior only; they are not a replacement for end-to-end accuracy/throughput verification.

## 5. Start the service (terminal A)

Use the **official BI-V150 launch arguments, unchanged**; run from the environment in section 3. Ensure nothing else is listening on port 9031 and record the full server logs.

```bash
export VLLM_PLUGINS=fl
export PYTHONPATH="/workspace/FlagGems/src:/workspace/vllm-plugin-FL:$PYTHONPATH"

vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 \
  2>&1 | tee /workspace/v8_serve.log
```

`PYTHONPATH` and `VLLM_PLUGINS` only direct Python to the intended installed source and enable the organizer's required plugin. Do not set any v1/v5-specific toggle; the selected v8 path is integrated and shape-gated within the original kernel. Verify the server reaches the ready state and actually loads the custom BI-V150 backend.

A simple, non-benchmark smoke test:

```bash
python3 - <<'PY'
import requests
r = requests.post(
    "http://127.0.0.1:9031/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json={
        "model": "minicpm",
        "messages": [{"role": "user", "content": "introduce llm"}],
        "max_tokens": 256,
    },
    timeout=600,
)
r.raise_for_status()
print(r.json())
PY
```

## 6. Accuracy: exact official MATH-500 Level 3 evaluation (terminal B)

With the v8 server running, invoke **the official EvalScope command**:

```bash
evalscope eval \
  --model minicpm \
  --api-url http://127.0.0.1:9031/v1/chat/completions \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets math_500 \
  --dataset-args '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}' \
  --eval-batch-size 8 \
  --timeout 3600 \
  --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
  --work-dir /workspace/evalscope-datasets/level3 \
  --ignore-errors
```

The organizers specify **MATH-500 Level 3**, **105 samples**, and minimum final accuracy **0.95** (baseline 0.962). Record the EvalScope-generated JSON, full console log, dataset count, correctness report, and server log. Because the official command uses `--ignore-errors`, manually ensure no request failures were silently ignored. The v8 profiling summary reports 98.1% accuracy; the raw corresponding final-v8 EvalScope report has **not yet been archived** in this submission draft, so that figure is not certified here.

## 7. Performance: exact official benchmark (terminal B)

Run the *unmodified* bundled benchmark against the same v8 server:

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

The official scenarios are:

| Input length | Output length | Concurrency | Number of requests |
|---:|---:|---:|---:|
| 4096 | 1024 | 64 | 256 |
| 16384 | 1024 | 64 | 128 |

Measure **total tokens/s** and **TTFT**. Keep `/workspace/benchmark_results/summary_*.csv`, any raw per-run files, server log, status for all requests, GPU utilization, software package versions, kernel-selection evidence and environment timestamps. For the official reproduction, do not alter test case sizes, request counts, concurrency, benchmark script, model settings, or inference parameters.

## 8. Evidence and how to verify results

The **historical internal full-run v8 summary** (not yet organizer-reproduced) reports:

| Scenario | v8 total tok/s | v8 mean TTFT (ms) |
|---|---:|---:|
| 4K / 1024 / 64 / 256 | 2331.29 | 8738.76 |
| 16K / 1024 / 64 / 128 | 1251.22 | 428970.85 |

Internal baseline comparison uses 1881.95 and 861.88 total tok/s. These baseline figures and raw baseline measurements live in the companion development repository under `4k16K_rawrun_benchmark.csv` / `4k16K_summary_benchmark.csv`; the v8 numbers are recorded in `profiling/v8/summary`. These are **self-run comparisons** and should not be confused with final official judging.

Before delivery, insert verified final benchmark and accuracy artifact filenames, their SHA-256 checksums, the two source Git SHAs, and successful-run evidence:

```text
source vllm-plugin-FL SHA: TO_FILL
source FlagGems SHA: TO_FILL
benchmark result filename(s): TO_FILL
MATH-500 Level 3 report filename: TO_FILL
accuracy: TO_FILL
4K total tok/s: TO_FILL
16K total tok/s: TO_FILL
4K/16K mean TTFT: TO_FILL
failures/timeouts/preemptions: TO_FILL
```

## 9. Troubleshooting and operating constraints

- **Wrong kernel imported:** Confirm the Python module paths inside the actual engine process; reinstall the submitted editable packages and restart the service. Do not rely only on `PYTHONPATH` of a different shell.
- **Long-context TTFT spikes:** The 32 GiB device cannot necessarily hold 64 independent 16K contexts simultaneously. Record KV-cache utilization, waiting requests and preemptions rather than attributing all time to the attention kernel.
- **Compilation or runtime mismatch:** Validate hardware, driver, Torch/CoreX, flagtree, the pinned FlagOS framework and FlagGems source before changing anything.
- **Accuracy mismatch:** Reuse the unmodified model, dataset, eval command and final optimized code. Treat all skipped/failed items as validation blockers.
- **Fallback behavior:** The v8 kernel uses compile-time specializations for only the supported BF16 16-query-head/2-KV-head/128-dim shapes, and preserves original fallback behavior elsewhere. No separate environment variable is required to turn on the v8-specific optimization.

The final ZIP must be uploaded to the official FlagOS platform before **2026-11-20 23:59 (UTC+8)**. The later upstream code PR requirement for winners is separate from the competition ZIP upload.

Official rules: https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn
