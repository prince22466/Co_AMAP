# Evaluation archive audit

Source: `submission-7-zip_First30Tasks.zip`
Diagnostics: `2.2-result-serialization`

Integrity errors: **0**.

| Task | Resolved | Patch chars | Task seconds | Observed tokens | Usage coverage | Pending tools | Helper calls |
|---|---|---:|---:|---:|---:|---:|---:|
| fastapi_15661 | False | 306 | 378.96 | 218707 | 25/25 | 0 | 1 |
| fastapi_15588 | False | 1313 | 242.89 | 193964 | 21/21 | 0 | 0 |
| fastapi_15589 | False | 7860 | 357.37 | 412200 | 46/46 | 0 | 0 |
| fastapi_15030 | False | 847 | 220.14 | 184400 | 20/20 | 0 | 0 |
| fastapi_14962 | False | 0 | 585.18 | 535332 | 56/56 | 0 | 0 |
| fastapi_15280 | False | 9433 | 877.69 | 455397 | 48/48 | 0 | 0 |
| fastapi_14964 | False | 2096 | 249.25 | 215739 | 27/27 | 0 | 0 |
| fastapi_15023 | False | 2817 | 386.80 | 423861 | 45/45 | 0 | 0 |
| fastapi_14986 | False | 1696 | 402.72 | 318140 | 34/34 | 0 | 0 |
| fastapi_14978 | False | 2204 | 323.41 | 363372 | 40/40 | 0 | 0 |
| requests_7505 | False | 1938 | 263.41 | 208050 | 22/22 | 0 | 0 |
| requests_7502 | False | 6555 | 557.42 | 411951 | 47/47 | 0 | 0 |
| requests_7433 | False | 560 | 481.42 | 384712 | 44/44 | 0 | 0 |
| requests_7427 | False | 1488 | 455.61 | 316503 | 34/34 | 0 | 0 |
| requests_7315 | False | 522 | 155.74 | 274482 | 32/32 | 0 | 0 |
| requests_7328 | False | 3318 | 693.59 | 432558 | 45/45 | 0 | 0 |
| requests_7309 | False | 692 | 305.71 | 329595 | 41/41 | 0 | 0 |
| requests_7205 | False | 503 | 128.91 | 123704 | 16/16 | 0 | 0 |
| rich_4070 | False | 1077 | 473.30 | 605915 | 60/60 | 0 | 0 |
| rich_4077 | True | 423 | 147.67 | 234942 | 25/25 | 0 | 0 |
| rich_4079 | False | 366 | 422.38 | 400612 | 44/44 | 0 | 0 |
| rich_4076 | False | 404 | 409.16 | 383272 | 43/43 | 0 | 0 |
| rich_4075 | False | 0 | 249.71 | 368157 | 45/45 | 0 | 0 |
| rich_4006 | False | 709 | 858.19 | 288963 | 32/32 | 0 | 0 |
| fastapi_14258 | False | 504 | 243.26 | 333444 | 42/42 | 0 | 0 |
| fastapi_14851 | True | 0 | 216.97 | 389611 | 45/45 | 0 | 0 |
| fastapi_14873 | False | 1111 | 282.56 | 392558 | 45/45 | 0 | 0 |
| fastapi_13920 | False | 4442 | 544.35 | 375406 | 38/38 | 0 | 0 |
| fastapi_14794 | True | 791 | 206.50 | 263149 | 27/27 | 0 | 0 |
| fastapi_14786 | True | 404 | 120.57 | 207026 | 23/23 | 0 | 0 |

## Integrity errors

None recorded.

## Measurement caveats

- fastapi_15661: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_15588: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_15589: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_15030: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_14962: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_15280: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_14964: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_15023: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_14986: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_14978: live/ATIF observed token totals differ; check additional helper or compaction requests
- requests_7505: live/ATIF observed token totals differ; check additional helper or compaction requests
- requests_7502: live/ATIF observed token totals differ; check additional helper or compaction requests
- requests_7433: live/ATIF observed token totals differ; check additional helper or compaction requests
- requests_7427: live/ATIF observed token totals differ; check additional helper or compaction requests
- requests_7315: live/ATIF observed token totals differ; check additional helper or compaction requests
- requests_7328: live/ATIF observed token totals differ; check additional helper or compaction requests
- requests_7309: live/ATIF observed token totals differ; check additional helper or compaction requests
- rich_4070: live/ATIF observed token totals differ; check additional helper or compaction requests
- rich_4079: live/ATIF observed token totals differ; check additional helper or compaction requests
- rich_4076: live/ATIF observed token totals differ; check additional helper or compaction requests
- rich_4075: live/ATIF observed token totals differ; check additional helper or compaction requests
- rich_4006: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_14258: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_14851: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_14873: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_13920: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_14794: live/ATIF observed token totals differ; check additional helper or compaction requests
- Task-type comparisons need labels assigned before testing

## Validation coverage

- helper_invocation_observed: True
- nonempty_patch_observed: True
- verification_exit_observed: True
- solved_task_observed: True
- configuration_snapshot_present: True
