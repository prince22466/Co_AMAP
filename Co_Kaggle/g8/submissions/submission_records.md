# submission records, with public scores and the submission-nb - Version x and its descriptions

| Submission and Description | public score | comments |
| --- | ---: | --- |
| submission-nb - Version 1 | 1.40815 | `submission-nb - Version 1.ipynb`; initial submission to check if it works |
| submission-nb - Version 2 | 0.06701 | `submission-nb - Version 2.ipynb`; replaces V1's all-zero actions with a metadata-driven HVAC feedback controller: schedule-aware comfort bands, per-zone VAV damper/reheat modulation, per-AHU PID supply-temperature and pressure control, cooling economizer/recirculation logic, idle-equipment defaults, and heating-demand-based hot-water reset. Public score improved from 1.40815 (V1) to 0.06701 (V2; lower is better, ~95.2% reduction). Remaining limitations: fixed PID gains and AHU-level mean-zone error may miss conflicting zone demands; no full local simulator validation is documented. |
| submission-nb - Version 3 | 0.06207 | `submission-nb - Version 3.ipynb`; preserves V2 PID/economizer with guarded outlier correction and demand-gated hot water. |
