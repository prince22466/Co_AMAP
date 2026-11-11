# v12 Codex autoresearch — append-only experiment evidence

**Program initialized 2026-11-11. Status: UNRUN.** No GPU correctness tests, new kernel A/B, v12 serving half-run, or v12 MATH-500 results have been performed in this setup PR.

## Fixed goals and evidence standard

| Workload | v10 reported ref (not matched) | +5% target | +10% target | Fresh v12 control |
| --- | ---: | ---: | ---: | --- |
| 4K (4096/1024/64/256) | 2372.38 | 2490.999 | 2609.618 | PENDING |
| 16K (16384/1024/64/128) | 1872.70 | 1966.335 | 2059.970 | PENDING |

For accepted result, require unchanged two-case half-run (`RUNS=2 SKIP_FIRST=0`), four SUCCESS rows, valid request counts 256/256 and 128/128 each repetition, same GPU and original dense baseline source, independent matched A/B replication and no unapproved other-workload regression. Report TTFT/TPOT and prefix-cache effects. Project quality gate >=95% MATH-500 is human-owned; reported v10 97.1% is historical only. Sparse candidates need separate long-context capability, comparison to dense outputs, and organizer eligibility.

### Baseline sources
- v12 main sources: same operator code as optimized v10 at v12 creation, while its prior AGENTS/program docs still incorrectly pointed at v10. This setup PR replaces those instructions.
- v11 short diagnostic 16K attention share ~79.25% of **summed GPU kernel time** (not serving wall time): historical prioritization clue only.
- PR #237 held a distinct unverified default-off sparse prototype; it is **not** in v12 main and cannot be treated as passed/active.
- Competition rules: https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn

## Current queue / hypotheses (re-rank after production traces)

| ID | Technique | Type | Status | Next evidence |
| --- | --- | --- | --- | --- |
| E1 | FlashAttention-2-style dataflow/pipeline inside existing kernel | exact | Not run | Current real 16K profile, quick equal-input A/B |
| E2 | Split-K/Flash-Decoding partials and stable merge | exact | Not run | Decode/long-prefix critical path |
| E3 | Paged-KV tile prefetch and page-index reuse | exact | Not run | Memory latency / tile dependence |
| S1 | Local+global-prefix block sparse | approximate | Not run | Explicit reference, long-middle retrieval |
| S2 | Query-aware page summaries / top-K page selection | approximate | Not run | Selector overhead, dense top-page recall |
| S3 | Hybrid selective decode or global anchors | approximate | Not run | Skip counts vs capability drift |
| R1 | Ring Attention | multi-device exact | Inapplicable until proved multi-device | Device count + comm bottleneck |

## GPU session provenance checklist

- Actual GPU owner/allocation/deadline (UTC) and budget guard state:
- Git branch, HEAD, code path, dirty status and operator SHA256:
- Model hash, exact benchmark SHA256, vLLM/Corex/Triton/FlagGems versions, BI-V150 GPU metrics:
- Proof loaded modules and production GPU kernel launches come from v12:
- Fresh v12 starting-source control A raw and summary CSVs, each row, cache states:
- Dense reference tests, mixed pages, BF16 tails, NaNs, 4K/16K:
- Approximate only: sparse-reference correctness, dense drift, middle-token retrieval and allowed-semantics ruling:
- Off-instance backed-up raw traces/logs, paths + SHA256, PR checkpoint:
- Exact next action and remaining allocation:

## Per-experiment record format

### [ID] — [UTC] — [hypothesis/provisional/verified/rejected/blocked]
- Source SHA, control/candidate patches, touched operator logic:
- Measured production bottleneck and Amdahl bound (GPU-sum versus wall-time clearly labeled):
- Numerical correctness, sparse/dense reference distinction, safety/rules:
- Kernel A/B (median,p10,p90, N, shape, launch count, K/V loads/tiles, occupancy, spills):
- Exact production invocation trace and unchanged dispatch/fallback confirmation:
- Short 4K and 16K serving A/B: raw numbers, warmup/caching, all errors:
- Long-context retrieval 4K/8K/16K: random positions, seeds, successful dense and candidate answers:
- Full 4K/16K unchanged half-run: four rows, success counts, TTFT/TPOT, matched replication:
- Accuracy/MATH-500 status: human only, organizer sparse eligibility status:
- Decision, rollback proof/accepted SHA, PR link, preserved evidence path+hash, next action:

**Never erase negative results or relabel a partial benchmark as successful.** A checkpoint helper and CSV parser cannot certify model capability, accuracy, independent reproducibility or official competition compliance.
