# FlagOS Season 2 / Track 2 — v8 submission draft

**Status: DRAFT, not the official submitted ZIP.** Last reviewed: 2026-11-06.

This directory stages the submission for **MiniCPM5-2B inference throughput optimization on Iluvatar BI-V150** using the selected **v8 = v1 prefill + v5 decode** kernel integration. The competition deadline is **2026-11-20 23:59 (UTC+8)**. The final artifact must be submitted on the FlagOS platform, not merely merged as a PR here.

## Files in this draft

- [`report.md`](report.md): editable source for the required PDF technical report.
- [`report.pdf`](report.pdf): generated review draft of the technical report; regenerate after any report edits.
- [`readme.md`](readme.md): submission-root reproduction guide, intended to become the final archive's `readme.md`.
- [`build_submission.sh`](build_submission.sh): stage the existing v8 code and create `<team-name>.zip`; requires the PDF to have been rendered.
- **Not included in this PR:** a duplicate v8 source tree, benchmark output, model weights, and unpublished machine-side logs. The packaging script reads the already-versioned v8 tree.

**Authoritative source:** [Optimization/v8](../../Optimization/v8). This PR does **not** modify any serving parameters, kernels, operator selection, weights, dataset, or official benchmark code. The committed PDF is a **generated draft** and must be regenerated and checked if its Markdown source changes.

## Official ZIP layout

```text
<team-name>.zip
└── <team-name>/
    ├── FlagGems/
    ├── vllm-plugin-FL/
    ├── readme.md
    └── report.pdf
```

The event page requires source code, a technical report (PDF), and a reproducible readme. Its specified framework lineage is `vllm-plugin-FL` branch `flagos-2026-s2` and `FlagGems` tag `v5.3.5`. Before release, verify the actual v8 source ancestry and installed versions against these anchors.

## Build the ZIP (after reviewing this draft)

From a checkout containing `Optimization/v8`, use the checked-in `report.pdf` as a draft or regenerate the Markdown technical report to PDF after review. One option, if Pandoc and XeLaTeX are installed:

```bash
cd Co_Kaggle/g7/tests_BI-V150/submission/v8
pandoc report.md --standalone --pdf-engine=xelatex \
  -V geometry:margin=22mm -V fontsize=10pt -o report.pdf
TEAM_NAME='REPLACE_WITH_REGISTERED_TEAM_NAME' bash build_submission.sh
```

Alternatively set `REPORT_PDF=/absolute/path/to/reviewed/report.pdf` when calling the packaging script. PDF production is a **submission-preparation** operation; it does not change the evaluation/benchmark environment.

The script uses the v8 code exactly as present in `Optimization/v8`. It copies the two trees and drops only transient VCS/cache artifacts from the temporary packaging directory. The registered team name must be substituted before packaging.

## Evidence ledger

| Item | Current evidence | Caveat |
|---|---|---|
| Final v8 selection | [profiling/v8/summary](../../profiling/v8/summary) | Summary-only; raw full-run v8 CSV and log still need to be archived |
| Historical self-run baseline | [4k16K_rawrun_benchmark.csv](../../4k16K_rawrun_benchmark.csv), [4k16K_summary_benchmark.csv](../../4k16K_summary_benchmark.csv) | Not an official organizer measurement |
| v8 source + tests | [Optimization/v8/readme.md](../../Optimization/v8/readme.md) | Must reproduce kernel selection on the serving process |
| v1 MATH-500 Level 3 evidence | [profiling/v1/summary.txt](../../profiling/v1/summary.txt) | 98.1% is logged for v1; does not substitute for a final v8 run |
| v8 accuracy statement | [profiling/v8/summary](../../profiling/v8/summary) | Reports 98.1%; attach the matching v8 EvalScope output before final submission |
| Environment snapshot | [HW_SW_info.txt](../../HW_SW_info.txt), [FlagOS_Env.txt](../../FlagOS_Env.txt) | Captured in Oct 2026; check final runtime |
| Long-context analysis | [profiling/v5/v5_decode_16k_bottleneck_findings.md](../../profiling/v5/v5_decode_16k_bottleneck_findings.md) | Diagnostic findings, not all v8-run measurements |

## Release gate — to close before official submission

- [ ] Fill in registered team name; confirm team members and credentials outside public repo.
- [ ] Verify the v8 `vllm-plugin-FL` ancestor and `FlagGems v5.3.5` source revision; record Git SHAs for both.
- [ ] Verify the installed v8 code is imported in the actual serving/engine process, not only an isolated Python interpreter.
- [ ] Re-run the two **unmodified** official full workloads and archive raw logs, summaries, shell commands, timestamps, GPU utilisation and environment. Confirm every request succeeded.
- [ ] Run **all 105 MATH-500 Level 3** samples with the official EvalScope command on the final v8 code and keep its full JSON/log; verify accuracy **>=0.95** and no ignored failures.
- [ ] Check 4K and 16K TTFT against their corresponding baseline runs, plus total tok/s.
- [ ] Have a reviewer validate report data, plots and wording. Produce and inspect final `report.pdf`.
- [ ] Create archive, inspect the root paths, run `unzip -t`, and test the documented deployment from the archive in a clean official environment.
- [ ] Upload the named ZIP to FlagOS by the deadline. **This repository PR is not the official upload.**
- [ ] If eligible as an award winner, follow the separate organizer requirement to submit a named upstream code PR after judging (within 3 working days after judging ends, per event page).

## Compliance

No quantization-by-hyperparameter, speculative decoding, official benchmark edits, framework operator-selection removal, operator substitution without optimization, or wholesale update/merge of newer FlagOS inference code is proposed. The submission report describes only the tuned attention kernel mechanics present in **v8**. The work focuses on throughput subject to accuracy and latency constraints.

Official rules: https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn
