# Group5 — Triton Shared Memory / Tiling / Synchronization A/B Lab

Target: the **Iluvatar BI-V150** with an existing PyTorch + Triton installation. Designed to live at:

`Co_AI_Inference/Triton_Learning/group5/`

No changes to vLLM, FlagGems, model parameters, competition scoring scripts, or inference configuration.

## Experiments

| ID | A | B | Meaning / important caveat |
|---|---|---|---|
| S01 — tiled transpose | 1D output-coalesced transpose with strided input reads | 2D tiled `tl.trans` conversion | Explores global coalescing and layout conversion. `tl.trans` **does not guarantee shared-memory traffic**; confirm from compiler output. |
| S02 — GEMM tile reuse | `BM=BN=32`, `BK=32` | `BM=BN=64`, `BK=32` | Larger output tiles can reuse A/B across more FMAs. Also changes grid size, register use, occupancy, matrix-instruction lowering, and possibly shared memory. Not a controlled shared-memory-only A/B. |
| S03 — CTA synchronization | FP32 FMA loop | Same loop + one `tl.debug_barrier()` per step | Measures barrier/scheduling overhead when the compiler emits barriers. This is **not** a shared-memory producer/consumer correctness demonstration. |
| S04 — explicit Gluon layout (opt-in) | `SwizzledSharedLayout(max_phase=1)` | `SwizzledSharedLayout(max_phase=4)` | Allocates a `TILE×TILE` shared-memory buffer; stores and reloads identical values via different register layouts. Same global I/O and logical arithmetic. Requires backend support for experimental Gluon; actual bank conflicts require corroborating IR or hardware counters. |

**Why S04 is opt-in:** Triton's experimental Gluon shared-memory descriptors and layout lowering are not guaranteed to be supported by every vendor backend. The core tests use `triton.language` and should be tried first. This is **not** an NVIDIA-only kernel: no vendor-specific PTX or async-copy instruction is used.

## Files

- `triton_shared_memory_lab.py` — GPU kernels, correctness checks, configurable A/B timing and CSV.
- `smem_lab_common.py` — shared timing, compiler metadata, output and formulas.
- `profile_triton_shared_lab.sh` — recommended smoke/full runner, optionally enables S04.
- `test_smem_lab_math.py` — CPU-only reference tests.
- `README.md` — this document.

## Quick start

```bash
cd Co_AI_Inference/Triton_Learning/group5
python3 test_smem_lab_math.py

# Compile & check correctness, no timed benchmark:
MODE=smoke bash profile_triton_shared_lab.sh

# Measure S01–S03 with alternating A/B order:
bash profile_triton_shared_lab.sh

# Try the explicit shared-memory Gluon experiment on your BI-V150 backend:
RUN_GLUON=1 MODE=smoke bash profile_triton_shared_lab.sh
RUN_GLUON=1 bash profile_triton_shared_lab.sh
```

Direct `--lab all` runs S01–S03; add `--include-gluon` to include S04.

S04 might fail at import/compile time if BI-V150's Triton backend does not support Gluon. Preserve the error; do **not** substitute CUDA-specific kernels or claim that the bank conflict A/B ran successfully.

### Individual A/B tests

```bash
python3 triton_shared_memory_lab.py --lab S01 --m 512 --n 1024 --out results/S01.csv
python3 triton_shared_memory_lab.py --lab S02 --m 512 --n 512 --k 512 --out results/S02.csv
python3 triton_shared_memory_lab.py --lab S03 --elements 262144 --barrier-steps 64 --out results/S03.csv
python3 triton_shared_memory_lab.py --lab S04 --smem-tile 64 --dump-asm results/asm --out results/S04.csv
```

For a single variant under a profiler:

```bash
python3 triton_shared_memory_lab.py --lab S04 --variant A --single-run --out results/S04_A_probe.csv
python3 triton_shared_memory_lab.py --lab S04 --variant B --single-run --out results/S04_B_probe.csv
```

`--single-run` **compiles + launches + validates once**; it does not produce a latency figure. Hardware profiling should be a separate process from the normal event-timing runs.

## Experiment controls and correctness

- S01, S04: full FP32 output exact comparison (`torch.testing.assert_close` with zero tolerance).
- S02: full BF16-input / FP32-output GEMM comparison to PyTorch FP32 matmul. Tolerance (`atol=0.03, rtol=0.02`) allows differences in matrix accumulation order; tighten it when backend precision permits.
- S03: full FP32 comparison to a PyTorch scalar recurrence with FP32 rounding tolerance, same input and FMA work for both variants.
- Each A/B uses the **same inputs, shapes and output buffers**. A/B timed trials alternate order (AB, BA, AB) and use event-based `triton.testing.do_bench`.
- Compilation and correctness checks happen outside timing. `trials`, `warmup-ms`, and `rep-ms` are configurable.
- Default matrices are 512×512 for quick tests; use larger/odd-size matrices to study launch floor and masks separately.
- `--strict` immediately raises on a failed configuration. A non-strict run still exits nonzero if any selected experiment has errors or if no configuration succeeds; do not silently treat errors as performance data.
- S02's modeled operand bytes count *logical per-program A/B requests*, **not actual HBM reads**. Cache reuse, masking and compiler lowering can change physical memory requests.

## Inspect the generated code and hardware counters

```bash
python3 triton_shared_memory_lab.py --lab S04 \
    --dump-asm results/smem_ir --out results/S04_with_ir.csv
```

Compare S04 A and B in generated IR/assembly. Require **nonzero** `shared_bytes_compiler`, verify shared-memory load/store and layout details, then examine BI-V150 hardware profiler counters if available:

- Shared/LDS load/store transactions, bank conflicts or serialization (if exposed by `ixkn-cli --query-metrics`)
- `active_warps_per_active_cycle` and stall reasons
- `time_duration`, cache/global transactions, memory request throughput

Not all BI-V150 counter names above are guaranteed. Query available metrics and use only supported ones. `shared_bytes_compiler` alone only indicates allocation and does not establish bank-conflict severity.

### Device warp width

S04 uses a layout whose lanes-per-warp parameter must match the actual backend. The script checks `torch.cuda.get_device_properties(0).warp_size` where available; otherwise it assumes 64 for BI-V150 and 32 for other GPUs. Override `--warp-size 64` or `--warp-size 32` if the installed backend differs. Confirm the compiled `warp_size_backend` field when supplied.

### How to interpret negative results

- A faster tiled transpose may simply have better **global coalescing**, not fewer shared-memory conflicts.
- A larger GEMM tile may slow down due to occupancy/register pressure even if it improves modeled operand reuse.
- `tl.debug_barrier` may be expensive even without any shared-memory communication; do not treat a barrier speedup as data-race evidence.
- Equal S04 timing can mean the backend already eliminates bank conflicts, the swizzles lower identically, global memory dominates, or the hardware lacks the expected bank behavior. Inspect IR and counters to distinguish these.

## Supporting Triton references

- Triton matmul tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- Triton `tl.debug_barrier`: https://triton-lang.org/main/python-api/generated/triton.language.debug_barrier.html
- Gluon layout tutorial: https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html
- Gluon shared-memory allocation: https://triton-lang.org/main/gluon/api/generated/triton.experimental.gluon.language.allocate_shared_memory.html
- Gluon swizzled shared layout: https://triton-lang.org/main/gluon/api/generated/triton.experimental.gluon.language.SwizzledSharedLayout.html
- Gluon `bank_conflicts` API (if supported by installed version): https://triton-lang.org/main/gluon/api/generated/triton.experimental.gluon.language.bank_conflicts.html

## Validation performed without BI-V150

`python -m py_compile`, `bash -n`, and the CPU reference unit tests. **GPU compilation, kernel accuracy and timing cannot be confirmed without the target GPU and installed Triton backend.**
