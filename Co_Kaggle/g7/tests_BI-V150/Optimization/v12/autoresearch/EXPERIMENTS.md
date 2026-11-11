# v12 autoresearch — append-only results

**Initial status: no v12 GPU experiments have run for this setup PR.** Numbers below are reported historical anchors, NOT independently matched evidence.

| Scenario | v10 reported tok/s | >=5% goal | Fresh v12 matched control |
|---|---:|---:|---|
| 4K [4096,1024,64,256] | 2372.38 | 2490.999 | Not run |
| 16K [16384,1024,64,128] | 1872.70 | 1966.335 | Not run |

The standalone unmerged PR #237 has a **default-off, untested** sparse kernel prototype. It is not part of v12 main or proof of a speedup.

## Experiment entry template (append, do not erase failures)

### [UTC] [E1/E2/E3/S1/S2/S3] — [hypothesis / provisional / verified / rejected]
- Code HEAD; diff and source/model/benchmark hashes; original dense control SHA:
- Actual GPU bottleneck and estimated end-to-end effect; workload/cache/graph state:
- Correctness and numerical metrics vs **dense**; if sparse, vs its explicit sparse oracle:
- Kernel A/B: median/spread, bytes or tile/FLOP counts, repeats, resource counters, logs:
- Actual v12 production invocation and short 4K/16K serving A/B:
- Long-context ability versus dense (beginning/middle/end across seeds), when approximate:
- Half-run: original four SUCCESS rows, 256/128 request counts, TTFT/TPOT, matched replication:
- MATH-500 (human) and organizer eligibility, if selected; no undocumented assumptions:
- Decision, accepted SHA or rollback, checkpoint/PR, raw evidence off-instance, next step:

The helper scripts check deadlines and CSV structure; they **cannot** certify model quality, reproducibility or eligibility. Preserve negative evidence, failures and rejected candidates.

### 2026-11-11 10:23 UTC — session setup, provisional
- Allocation confirmed until 16:00 UTC (17:00 Warsaw); budget guard initialized once.
- Remote checkout was an extracted source tree without research files or its own Git checkout; missing research files were copied from local main bbdfba2665e6f3aa8c9d17d7acca8173a181ee8b. The attention source SHA-256 matches local: `6e3e8f7cbc24140a9cfd3ef3d2ed8d99cf7ae0f4e0147ac3d9d9b0775925931a`. Complete tree identity is not yet verified.
- One idle BI-V150, PyTorch 2.10.0; plugin and FlagGems import from `/root/Optimization/v12`; external half-run SHA-256 `873f17f932d4643222be9903ba3b569c1863c4e2d246ceafbe6f75ebe7163ffc`.
- 21 research helper tests passed; baseline attention correctness is running. No throughput or accuracy results yet.
- Next: verify full source provenance, finish correctness, profile production attention, screen exact and sparse candidates. GitHub branch checkpoint uses connector because remote Git access is not yet established.

### 2026-11-11 10:25 UTC — baseline correctness and source verification
- Remote attention, decode, and mixed-paging suite: **21 passed**, one Triton benchmark fallback warning, 14.97 s. Research helper suite: 21 passed.
- SHA-256 compared 5,177 tracked files from local main: all 5,139 present remote files matched; 38 `FlagGems/.github` metadata files were absent. No source content mismatch observed. The remote extracted tree still is not a Git checkout; source identity should not be described as a checked-out Git SHA.
- A dedicated baseline server was started with fixed runbook flags; its first graph capture is in progress. Baseline quick serving diagnostics are queued behind health readiness. No A/B measurements yet.
- One physical GPU was detected, so multi-GPU Ring Attention is not applicable in this allocation.

### 2026-11-11 10:29 UTC — E3 prefill page-table reuse, rejected
- Hypothesis: reuse the existing decode page-table lookup in standard BF16 GQA prefill, reducing redundant physical-page lookups. Isolated experimental copy; original production source unchanged. Imported from `.autoresearch-work/screen`, baseline/variant text SHA-256 after Python newline normalization `263c73746e48fd16989b6c6587a485a1f7505f303ba93ff78d377159f4fecac6` / `7c7c4556a0f2441fd48c92296515d0e6317611b226a551cb4f48fbd6a288d162` (raw production source SHA recorded above).
- 12 interleaved paired GPU-event measurements per shape, identical random BF16 tensors with shuffled 16-token pages; comparisons were bit-identical for these cases. q=63/context=4097: 0.5355 vs 0.6190 ms (+15.6%); q=2048/context=2048: 3.5781 vs 4.6124 ms (+28.9%); q=2048/context=14336: 17.5197 vs 22.8435 ms (+30.4%). q=1/context=16377 is unchanged code and measured 2.3934 vs 2.4323 ms. These are isolated operator timings, not production traces; registers, spills and detailed production GPU share remain unmeasured.
- Reject; no serving A/B justified. Not a verified speedup. Logs and candidate/harness are in `.autoresearch-work/screen/`; off-instance backup pending.
- Baseline non-scoring quick run (concurrency 8, output 128): 4K 16/16 successful, 2459.48 total tok/s, mean TTFT 6038.05 ms, TPOT 60.03 ms; 16K 8/8 successful, 3196.27 total tok/s, mean TTFT 19050.60 ms, TPOT 170.62 ms. No matched restart and not an official workload; cache conditions are not controlled. 4K/16K logs under `.autoresearch-work/quick/baseline/`.
- Next: obtain production attention profile and test independent exact dataflow hypothesis; explore sparse separately with a middle-context quality gate.
