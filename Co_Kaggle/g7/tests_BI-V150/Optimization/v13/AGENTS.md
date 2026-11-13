# v13 — validation instructions

v13 is a device-independent refactor of v10, not a new autoresearch optimization project. Read [readme.md](readme.md) and [program.md](program.md).

- Edit only `Co_Kaggle/g7/tests_BI-V150/Optimization/v13` (remote `/root/Optimization/v13`). Do not edit v10, other versions or externally installed sources. v10 may be run read-only as a matched control from its own workspace.
- Preserve the original dense Triton computation, BF16 GQA prefill tiling, two-stage pipeline, decode page lookup and all existing workload/fallback conditions. Preserve the original framework dispatch and non-Triton selections. Do not force alternative hardware to use Triton.
- No quantization/speculative sampling by changing hyperparameters; no benchmark-script tuning, dispatch-only performance shortcuts, removal of framework selection logic or merging the latest FlagOS branch. Do not change model weights, serving flags or benchmark inputs for score manipulation.
- Validate import paths, the final server backend path, attention/decode/mixed correctness, unchanged 4K/16K benchmarks and MATH-500. Compare v13 with fresh matched v10 controls, record both workloads and disclose variance and regressions. Do not claim performance gains or cross-device compatibility without evidence.
- Keep generated logs and traces in ignored `.autoresearch-work/` under the respective version. Do not stop unrelated services. Reserve at least 90 minutes before starting a half-run; full completion may take longer. No unattended research is authorized.
