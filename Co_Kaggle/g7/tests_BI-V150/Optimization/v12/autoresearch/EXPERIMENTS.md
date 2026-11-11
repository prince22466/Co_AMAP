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
