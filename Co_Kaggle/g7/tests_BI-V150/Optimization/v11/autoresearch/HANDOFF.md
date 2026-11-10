# v11 latest research handoff

Updated UTC: 2026-11-10T16:49:49+00:00

Status: provisional (researcher-reported)

Source HEAD before checkpoint: 67e4492d8442fb437b69bf6d7204206ed245d402

Remaining GPU time: unknown

Finding: Credit-pressure handoff: user requested grouped MM prototype and major findings in draft PR; code and seven GPU-tested cases committed as 67e4492d, no verified serving win

Evidence/provenance: Candidate hash a6edcade; paired kernel 1.402 to 1.170 ms and 1.412 to 1.174 ms. A1/B1/B2/A2 short 4K 2464.25/2054.31/2509.82/2473.56 and 16K 3196.90/2596.40/3235.04/3194.16; all requests successful; 7 focused plus 21 attention tests passed. Second-projection extension regressed 0.72554 to 0.76789 ms and was rejected. Full short logs and negative screen logs backed up locally under .autoresearch-work/remote-handoff; raw half-run evidence tracked on PR.

Small CSV evidence committed when --push works: none (external backup required)

Next action: Next session inspect cold versus warm autotuning and graph replay, prove grouped candidate executes in production; repeat matched short serving screens and only admit half-runs if justified; no promotion without independent A/B and human MATH-500

Incomplete half-runs are never verified results. Ignored raw artifacts must be backed up outside the rented instance. This handoff does not verify accuracy or establish a causal speedup.
