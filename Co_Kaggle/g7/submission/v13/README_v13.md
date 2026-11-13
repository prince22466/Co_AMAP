# Submission v13 — prince22466

## Sources and evidence

- [v13 source](../../tests_BI-V150/Optimization/v13/) — device-independent refactor of v10; all production changes relative to v0 are in vllm-plugin-FL.
- [v13 profiling](../../tests_BI-V150/profiling/v13/) — official full-run CSVs and half-run/accuracy summary.
- [Historical baseline](../../tests_BI-V150/4k16K_summary_benchmark.csv) — first full-run block (1881.95 / 861.88 tok/s), consistent with the previous submission comparison.
- [Technical report source](report.md) and [submission PDF](report.pdf).
- [Reproduction instructions](readme.md).

## Verified results (full-run)

| Workload | Historical baseline | v13 | Gain |
|---|---:|---:|---:|
| 4K | 1881.95 | 2752.60 | +46.26% |
| 16K | 861.88 | 1890.18 | +119.31% |

All values are total tokens/s. MATH-500 threshold recorded as >=0.95: True; exact v13 accuracy and complete evaluation logs are not in the profiling folder. Half-run results are internal diagnostics only.

## Submission archive

Run `bash build_submission.sh` in this directory to create `prince22466.zip`. The existing packaging layout is retained:

```text
prince22466.zip
└── prince22466/
    ├── FlagGems/       # unchanged vs v0; optional component retained
    ├── vllm-plugin-FL/ # v13 source
    ├── readme.md
    └── report.pdf
```

The archive includes v13 source, reproduction instructions and the Chinese technical report. The editable report and figures remain in this repository; they are not required inside the submission ZIP.

## Compliance

No quantization-by-hyperparameter, speculative decoding, official benchmark modifications, removal of the main framework selection logic, substitution with an unoptimized operator, or direct merge of newer FlagOS inference code. v13 retains the original dispatch and specializes only selected stock dense Triton attention, using the actual prefill and decode kernel optimizations. No claim of performance on untested hardware.

Official rules: https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn
