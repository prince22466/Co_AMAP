# Group7 — Advanced GEMM A/B Lab (Triton / BI-V150)

Educational, standalone Triton kernels that study **GEMM tiling, grouped tile
ordering, split-K, and persistent work scheduling**. These are not vLLM
plugins, FlagGems changes, or competition benchmark scripts. No quantization,
speculative decoding, or modifications to framework operator dispatch occur.

## Contents

| File | Purpose |
| --- | --- |
| `triton_advanced_gemm_lab.py` | G01–G04 kernels, command-line sweep, correctness validation |
| `gemm_lab_common.py` | Event timing, correctness, compiler metadata, CSV, CPU mapping helpers |
| `profile_triton_gemm_lab.sh` | Smoke/full runner, optional BI-V150 `ixkn-cli` capture |
| `test_gemm_lab_math.py` | CPU-only tests of tile mapping, split-K coverage and scheduling |
| `README.md` | Experiment design and usage |

## Quick start

Run on the **BI-V150** environment that already has PyTorch + Triton installed:

```bash
cd /root/Optimization/Triton_Learning/group7    # adjust to your location
python3 test_gemm_lab_math.py
MODE=smoke bash profile_triton_gemm_lab.sh
bash profile_triton_gemm_lab.sh
```

Run individual experiments, with CSV and compiler-generated text when available:

```bash
python3 triton_advanced_gemm_lab.py --lab G01 --out results/G01.csv --dump-asm results/asm
python3 triton_advanced_gemm_lab.py --lab G02 --out results/G02.csv --dump-asm results/asm
python3 triton_advanced_gemm_lab.py --lab G03 --out results/G03.csv --dump-asm results/asm
python3 triton_advanced_gemm_lab.py --lab G04 --out results/G04.csv --dump-asm results/asm
```

`--single-run` compiles and checks correctness, without timing. The runner's
`MODE=smoke` uses nonmultiple dimensions to exercise masked tails.

### Why each lab is different

| Lab | Baseline | Alternatives | Interpretation |
| --- | --- | --- | --- |
| G01 Tile shape | `32x64x32` | `64x64x32`, `64x128x32`, `128x64x32` | Throughput versus tile footprint/occupancy |
| G02 Group ordering | `GROUP_M=1` (row-major tile traversal) | `GROUP_M=8,16` | L2/cache reuse due to CTA order |
| G03 Split-K | One complete GEMM kernel | Split-K factors 2, 4, 8 **plus** reduction kernel | More K parallelism versus scratch traffic and second launch |
| G04 Persistent grid | One CTA per tile | Fixed worker grids of 32/64 CTAs | CTA count/scheduling trade-off; each worker may handle multiple tiles |

**G02 grouped tile ordering is not `grouped GEMM` across different independent
matrix pairs.** It modifies ordering of output tiles for a single GEMM.

## Recommended shapes

All dimensions are configurable and support nonmultiples of the tile size.
Input operands are BF16 (default) or FP16, and results are accumulated and
returned in FP32. These are arithmetic input dtypes, not quantization modes.

```bash
# Prefill-like GEMM: moderate M, large N
python3 triton_advanced_gemm_lab.py --lab all --m 512 --n 2048 --k 1024 \
  --out results/prefill.csv

# Decode-like skinny GEMM: very small M, large N and K
python3 triton_advanced_gemm_lab.py --lab G03 --m 16 --n 4096 --k 2048 \
  --splits 2,4,8 --out results/decode_splitk.csv

# Larger matrix where persistence may be more meaningful
python3 triton_advanced_gemm_lab.py --lab G04 --m 2048 --n 2048 --k 1024 \
  --workers 32,64,128 --out results/persistent.csv
```

Other useful arguments:

```text
--tiles 32x64x32,64x64x32,64x128x32    # G01 sweep
--bm 64 --bn 64 --bk 32                # G02/G03/G04 fixed tile
--groups 1,8,16                        # G02 (1 is baseline)
--splits 2,4,8                         # G03 (1 is automatic baseline)
--workers 32,64                        # G04 (ordinary grid is baseline)
--warps 4 --stages 2
--warmup-ms 30 --rep-ms 100 --trials 3
--variant group_m_8 --lab G02 --single-run # isolated profiler invocation
--strict                               # fail on first bad/unsupported kernel
```

## Measurement protocol and limitations

1. **Correctness first.** Every configuration is compiled and checked against
   `torch.matmul(A.float(), B.float())`, with FP32 output and tolerances
   `atol=0.02, rtol=0.006`. With large K or a different BI-V150 compiler,
   consider whether expected FP32 accumulation variation requires adjusting
   tolerances, but do not silently accept failed comparisons.
2. The FP32 reference is evaluated *outside* the timer. The output tensor and
   G03 workspaces are allocated *outside* the timer. Compilation is also
   excluded. The timer uses `triton.testing.do_bench` GPU event timing.
3. The benchmark alternates A/B configurations across trials. Repeat on an
   otherwise idle GPU; compare medians and variability, not a single run.
4. G03 measures the entire **GEMM + workspace reduction** sequence, including
   both kernel launch costs and temporary reads/writes. It does not include
   `torch.empty` allocation, which is explicitly excluded from all variants.
5. G04 is a manually **bounded persistent CTA work grid**, not a device-specific
   TMA, Warp Specialization, Stream-K, or native hardware persistence feature.
   The number of workers is clamped to the total output tile count.
6. Timings report **effective TFLOP/s = 2*M*N*K / elapsed time**. This is
   useful-work throughput, not hardware-issued FLOPs. Extra split-K reduction
   instructions and transfers are not counted in that numerator.
7. The metadata fields `kernel*_registers_per_thread`, `kernel*_spills`,
   `kernel*_shared_bytes` are reported only where exposed by the backend.
   Real occupancy, cache hit rates, and bank conflicts require device counters.
8. If `--dump-asm` is supplied, record textual backend IR/ISA where available.
   Source-level grouping or pipelining does **not** prove cache reuse or
   asynchronous execution; verify hardware counters and compiled code.
9. These are isolated educational GEMM kernels; any proposed competition
   optimization must be applied to a real kernel and checked for `math_500`
   accuracy and 4K/16K **total tokens/s**, without benchmark-script changes.

### Optional BI-V150 hardware profiling

```bash
PROFILE=1 bash profile_triton_gemm_lab.sh
```

The runner captures selected variants independently and saves
`ixkn_supported_metrics.txt`. `ixkn-cli` versions differ: override `METRICS`
with supported names and inspect whether numeric counter rows actually match
requested kernels. Do not compare profiler-instrumented times against normal
GPU-event timings. Profiling can generate large output files.

### Troubleshooting

- `No GPU visible`: Run in the GPU container with the installed Triton stack.
- `Unsupported dot` or compilation failure: verify BF16/FP16 matrix instruction
  support, device Triton backend, and try a smaller supported tile. Failed
  configurations are recorded and cause nonzero exit status.
- `G04 persistent` unsupported: inspect the nested loop compilation output;
  this is an optional experimental kernel for compiler compatibility.
- Slow split-K results: expected on large M*N or small K when workspace
  traffic and second launch exceed extra K parallelism benefits.
- Slow grouped ordering: grouping can make cache behavior worse for some
  shapes and devices. This is not automatically an optimization.

## Next analysis

Compare `G01`/`G02` for cache/tile effects, `G03` for skinny/large-K
underutilization, and `G04` for work-grid sizing. The main result is a map of
which kernel mechanism is beneficial **for the selected BI-V150 shape**.
