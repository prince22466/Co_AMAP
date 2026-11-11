# v12 autoresearch — append-only results

**Initial status: no v12 GPU experiments have run for this setup PR.** Numbers below are reported historical anchors, NOT independently matched evidence.

| Scenario | v10 reported tok/s | +5% | +10% | Fresh v12 matched control |
|---|---:|---:|---:|---|
| 4K [4096,1024,64,256] | 2372.38 | 2490.999 | 2609.618 | Not run |
| 16K [16384,1024,64,128] | 1872.70 | 1966.335 | 2059.970 | Not run |

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
