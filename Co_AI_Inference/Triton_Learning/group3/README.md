# Triton ILP + Occupancy/Resource A/B Labs

**Goal:** Learn two GPU compute-performance mechanisms on the BI-V150 (or any supported GPU) without changing MiniCPM, vLLM, FlagGems, or the competition benchmarks.

## Prerequisites

Use the existing GPU environment with working `torch`, the GPU backend, and `triton`. The lab relies only on `torch`, `triton`, and `numpy` (already used by PyTorch/Triton stacks). In particular, do **not** install a random stock Triton wheel over the device vendor's Triton backend.

The lab creates its own inputs and uses Triton's event-based `triton.testing.do_bench` with quantile timings. It does **not** use CUDA graphs or a vendor-specific profiler; optional IR dumps are backend-dependent.

## Start with smoke test

```bash
cd triton_ilp_occupancy_lab
MODE=smoke bash profile_triton_ilp_occupancy.sh
```

Then run a full sweep:

```bash
bash profile_triton_ilp_occupancy.sh
```

Run scripts independently:

```bash
python3 triton_ilp_lab.py --n 262144 --blocks 128,256 --warps 4 --iters 256 \
  --dump-asm results/ilp_asm --out results/ilp.csv

python3 triton_occupancy_lab.py --n 262144 --blocks 128,256,512 \
  --warps 4,8 --accumulators 1,2,4,8,16 --fma-per-element 512 \
  --dump-asm results/occ_asm --out results/occupancy.csv
```


## Experiment 1: ILP (FMA dependency)

- A: `serial_fma` performs **4 × ITERS** chained `tl.fma` operations per element, with one live FMA chain.
- B: `parallel_fma` performs the **same number** of `tl.fma` operations, split across four independent chains.
- Both load the same three FP32 inputs, write one FP32 output, and use the same launch parameters.
- The final arithmetic outputs are **not the same function**. Each is validated independently against a CPU reference. Compare **timing/FLOPs** rather than bitwise output equality.
- If B improves significantly, the machine can hide FMA dependency latency with more ILP; if not, consider compiler scheduling, instruction throughput, register demand, grid saturation, or launch floor.

## Experiment 2: Occupancy / register pressure / warp scheduling

- Sweep block size (`BLOCK`), active warp configuration (`num_warps`), and concurrently live accumulators (`NUM_ACC`).
- Keep the total **logical** FMA count per output element fixed by setting `ITERS = fma_per_element // NUM_ACC`.
- The input and output memory traffic are the same for all configs, apart from cache-layout and hardware effects.
- **Compare the same (BLOCK, NUM_ACC) with different num_warps to isolate warp configuration.**
- Changing `NUM_ACC` simultaneously changes dependency depth, register demand, and reduction overhead; it is a *resource/ILP trade-off*, not a pure occupancy A/B.
- Compiled `registers_per_thread` and `spills` can indicate resource pressure. Actual SM occupancy, waves, and scheduling stalls require BI-V150 hardware counters.

## CSV fields

`latency_us`: median of trial medians from GPU event timings.

`trial_medians_us`: each trial's kernel-latency median.

`logical_gflop`: 2 FLOPs per FMA. For ILP, `2 * n * (4 * iters) / 1e9`.

`effective_tflops`: logical GFLOPs / elapsed milliseconds. This is an operation-count-based proxy. Real machine instruction count, compiler optimization, special instructions, and loop overhead may differ; compare generated assembly/IR.

`registers_per_thread`, `spills`, `shared_bytes`, `backend_warp_size`: *when exposed* by the backend. Missing values do not mean zero.

`.env.json`: device name, tool versions, and arguments for reproducibility.

## Correctness and fairness

- No measured compilation; all specializations are compiled and checked before measurement.
- CPU FP64 reference verifies representative output samples for each variant with FP32-tolerant thresholds; this is a fast sanity check, not a rigorous all-elements FP32 reference test.
- Same N, dtype, input buffers, data values, and operation-count formula across A/B variants.
- Every benchmark should be repeated across independent runs (and, ideally, interleave A/B orders); short kernels may be dominated by host scheduling, GPU clock variation, and launch overhead.
- Inspect assembly/IR with `--dump-asm`. A compiler may transform operations; source-level FLOP counts are not proof of machine-level executed instruction count.
- The above results are microbenchmarks, **not** evidence of faster MiniCPM 4K/16K total tokens/s. Apply any promising finding to a real vLLM/FlagGems kernel, run correctness tests, and separately evaluate full serving throughput.
