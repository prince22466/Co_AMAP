# Group9 — Triton Kernel Launch & Execution Lab (BI-V150)

Four standalone A/B experiments about **kernel launch scaling, operator fusion, graph replay and persistent work scheduling**. These teaching scripts do **not** change vLLM, FlagGems, the official benchmark, quantization settings, or decoding behavior.

Works with a functioning PyTorch + Triton GPU installation. The `torch.cuda` API also appears on some non-NVIDIA backend stacks, but **CUDA Graph support must be tested on the actual BI-V150 runtime**. A successful capture/replay is never assumed.

## Files

- `triton_execution_lab.py` — GPU kernels, controls, correctness and measurements.
- `execution_lab_common.py` — event timings, synced wall timings, CPU comparison, metadata, CSV and IR dumps.
- `execution_reference.py` — NumPy-only references and persistent-grid scheduling helpers.
- `test_execution_lab_math.py` — CPU-only regression tests.
- `profile_triton_execution_lab.sh` — smoke, timed runs and optional `ixkn-cli` captures.

## Quick start

Place the scripts together under `Co_AI_Inference/Triton_Learning/group9/`:

```bash
python3 test_execution_lab_math.py
MODE=smoke bash profile_triton_execution_lab.sh
bash profile_triton_execution_lab.sh
```

Run one experiment:

```bash
python3 triton_execution_lab.py --lab L01 --out results/L01.csv
python3 triton_execution_lab.py --lab L02 --out results/L02.csv
python3 triton_execution_lab.py --lab L03 --out results/L03.csv
python3 triton_execution_lab.py --lab L04 --out results/L04.csv
```

## Experiments

| Lab | Variants | Controlled question | Important confounder |
|---|---|---|---|
| **L01 Launch scaling** | 1, 2, 4, 8 launches of the same tiny kernel | How does the cost of a repeated launch sequence scale? | Total arithmetic, logical bytes and writes grow with launch count; this is **not** an isolated per-launch measurement |
| **L02 Fusion** | Separate A → B vs fused A+B | Is eliminating a launch and an intermediate tensor beneficial? | Also eliminates memory traffic; fusion benefit cannot be attributed solely to launch overhead |
| **L03 Graph replay** | Eager chain of 1/4/8 kernels vs capture + replay of the exact chain | Does command-buffer replay reduce host dispatch cost? | Capture is excluded from per-replay timing; depends on backend support; event results may not reflect CPU dispatch reduction |
| **L04 Persistent workers** | One CTA per tile vs bounded worker grid (16/32/64) | Does recycling CTAs inside a kernel help? | Grid geometry/occupancy/loop overhead changes; this is persistent **tile scheduling**, not a persistent serving engine |

### Fairness and correctness

- L01 tests use the same tiny kernel per launch. They measure *sequence* execution (not a claim about intrinsic hardware launch latency).
- L02 uses the same input, output and arithmetic; the unfused variant reads/writes a temporary. All allocations are excluded from timing. `enable_fp_fusion=False` reduces but does not completely eliminate compiler transformation differences.
- L03 uses an identical sequence of affine kernels for eager and graph replay. Capture and compilation are excluded from timing. The graph holds long-lived input/intermediate/output tensors. The output is overwritten with NaNs before the first replay to check that replay really executes.
- L04 has identical FP32 FMA work and one input read + one output write per logical element in both variants. Persistent CTA grid size is capped at the total number of tiles.
- All variants compare full results to **CPU** reference data. The Group8 issue demonstrated why FP64 reduction/softmax references should not be executed on an unverified accelerator FP64 backend.
- GPU compiler metadata is recorded where exposed; blank fields mean unavailable. IR text dumps are optional.

### Timing metrics

- **`gpu_event_us`**: GPU event elapsed per call into the variant launcher (one call can launch multiple kernels). Event timing is **not pure Python/driver overhead**.
- **`sync_wall_us_per_call`**: elapsed wall clock per invocation, measured across `--host-batch` calls followed by a synchronization, including Python, launch submission and GPU execution. It is **not pure GPU kernel time** or pure CPU launch overhead.
- `gpu_event_us_per_launch` is only a normalized ratio; it is not measured latency for an isolated single kernel.
- GPU timing uses `triton.testing.do_bench` (with its own cache handling) and is separate from the optional hardware profiler. Do not mix profiler timings with unprofiled results.
- Run several trials and compare medians, especially for tiny launch timings. Device clocks and power state can change results.

### Graph portability

If the runtime cannot capture a `torch.cuda.CUDAGraph`, `L03 graph_*` rows are marked `unsupported`, with the reason in CSV. Eager rows still run. To make missing graph support a fatal error, set `REQUIRE_GRAPH=1` for the runner or use `--require-graph` directly. **Do not interpret a skipped graph as slower than eager.**

### Custom tests

```bash
python3 triton_execution_lab.py --lab L01 --tiny-n 512 --launches 1,2,4,8,16 --out results/launch.csv
python3 triton_execution_lab.py --lab L03 --graph-n 8192 --graph-steps 1,2,4,8 --out results/graphs.csv
python3 triton_execution_lab.py --lab L04 --n 262143 --block 128 --workers 16,32,64,128 --iters 128 --out results/persistent.csv
```

### Optional BI-V150 hardware counters

```bash
PROFILE=1 bash profile_triton_execution_lab.sh
```

This invokes `ixkn-cli`, if installed, and saves counter logs. Check profiler version, numeric counter rows and kernel-name matches. The runner deliberately does not claim that a graph replay was captured by the profiler.

## Interpreting results

- For launch scaling, compare total event time, synced wall time and the normalized per-launch values. The intercept/slope is an *approximate* launch-sequence model, not a hardware latency constant.
- For fusion, use end-to-end GPU time **and** logical global-memory bytes; reductions may be caused by eliminating the intermediate buffer as well as one launch.
- For graph replay, compare `eager_N` and `graph_N` at the same `N`, tensor size, dtype, and kernel chain; ensure the graph row passed correctness and is not `unsupported`.
- For persistent workers, compare `grid` with each `persistent_N`, especially at many tile counts. Some configurations will be slower due to long worker loops or insufficient active CTAs.

## Compatibility / validation

Python and shell syntax and CPU tests are checked without a GPU. **BI-V150 compilation, graph capture, GPU correctness and performance still need to be tested on the target machine.**
