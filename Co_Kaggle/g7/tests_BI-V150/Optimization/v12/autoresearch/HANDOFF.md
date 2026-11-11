# v12 autoresearch handoff

**Status: research in progress; no validated throughput improvement.** Allocation ends 2026-11-11 16:00 UTC (17:00 Warsaw). The deadline guard was initialized once.

- Baseline main revision `bbdfba2665e6f3aa8c9d17d7acca8173a181ee8b`; remote is an extracted tree, not a Git checkout. All 5,139 present tracked v12 source files match local hashes; 38 GitHub metadata files are absent. Production operator SHA-256 `6e3e8f7cbc24140a9cfd3ef3d2ed8d99cf7ae0f4e0147ac3d9d9b0775925931a`. Benchmark SHA `873f17f932d4643222be9903ba3b569c1863c4e2d246ceafbe6f75ebe7163ffc`.
- Baseline helper and GPU regression suites: 21 passed each. Non-scoring short serving baselines: 4K 2459.48 total tok/s, TTFT 6038.05 ms, TPOT 60.03 ms (16/16); 16K 3196.27 total tok/s, TTFT 19050.60 ms, TPOT 170.62 ms (8/8). Not official and no matched A/B.
- Exact prefill page-table reuse: reject; isolated kernel 15.6–30.4% slower on tested prefill shapes. No serving A/B.
- PR #237 sparse prototype: 4 oracle tests fail; original dense also differs from the independent CPU/GPU FP32 reference in the full-window case. This pre-existing discrepancy requires investigation before sparse correctness can be determined; do not treat sparse as validated or contest eligible.
- Raw evidence is under `/root/Optimization/v12/.autoresearch-work/`; screen, quick and baseline test logs backed up under the local workspace's `Optimization/v12/.autoresearch-work/remote_backup/`. See append-only EXPERIMENTS.md for details and limitations.
- Production server trace in progress under `.autoresearch-work/profile/` using `ixsys`; analyze attention GPU share vs wall time and profile prefill/mixed/decode. Continue independent exact hypotheses, and fix/understand the numerical oracle discrepancy before sparse screening. Do not run half-benchmarks until the required serving gate and budget admission pass. Save logs off-instance and checkpoint changes on the draft PR branch.
- GitHub is unreachable directly from the GPU host; PR updates are being made through the connected GitHub app. No operator edits have been accepted or committed. Never auto-merge.

## Latest candidate checkpoint (2026-11-11 10:51 UTC)

- Stage-one BF16 GQA prefill candidate is committed on the PR branch and passed 21 GPU regression tests. Its isolated kernel was 10.9–14.8% faster on three sampled prefill shapes, but this is not serving proof.
- First quick serving A1/B1 was mixed: 4K 2459.48 -> 2263.25 tok/s, TTFT 6038 -> 7245 ms; 16K 3196.27 -> 3567.85 tok/s, TTFT 19051 -> 16711 ms. No causal claim; no half-run decision until replicated and the 4K regression is explained.
- Remote currently has original source restored (SHA `6e3e8f7cbc24140a9cfd3ef3d2ed8d99cf7ae0f4e0147ac3d9d9b0775925931a`) and fresh control server A2 is starting. Candidate source and test backups are under `.autoresearch-work/stage1/`. Next: finish A2, restart candidate B2 with same conditions, compare both scenarios and TTFT/TPOT, prove kernel invocation and resource behavior, then decide whether budget and rules permit any half-run.
- Production trace and SQLite are backed up off-instance in the local workspace, hashes in EXPERIMENTS.md. The independent dense-vs-FP32 oracle discrepancy remains unresolved and must not be hidden by the passing vLLM comparison tests.
