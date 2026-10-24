# Evaluation archive audit

Source: `submission-4-zip_First30Tasks.zip`
Diagnostics: `2.2-result-serialization`

Integrity errors: **0**.

| Task | Resolved | Patch chars | Task seconds | Observed tokens | Usage coverage | Pending tools | Helper calls |
|---|---|---:|---:|---:|---:|---:|---:|
| fastapi_15661 | False | 2755 | 671.22 | 370646 | 38/38 | 0 | 0 |
| fastapi_15588 | False | 1466 | 197.21 | 203758 | 24/24 | 0 | 0 |
| fastapi_15589 | False | 794 | 368.68 | 422956 | 46/46 | 0 | 0 |
| fastapi_15030 | False | 1128 | 272.19 | 300532 | 34/34 | 0 | 0 |
| fastapi_14962 | False | 2866 | 367.42 | 418029 | 43/43 | 0 | 0 |
| fastapi_15280 | False | 1589 | 178.23 | 355844 | 44/44 | 0 | 0 |
| fastapi_14964 | False | 2664 | 240.48 | 235476 | 30/30 | 0 | 0 |
| fastapi_15023 | False | 1598 | 306.74 | 362866 | 42/42 | 0 | 0 |
| fastapi_14986 | False | 1191 | 214.83 | 203951 | 27/27 | 0 | 0 |
| fastapi_14978 | False | 2406 | 445.94 | 415960 | 47/47 | 0 | 0 |
| requests_7505 | False | 4997 | 431.28 | 322983 | 41/41 | 0 | 0 |
| requests_7502 | False | 2254 | 339.10 | 115178 | 15/16 | 0 | 0 |
| requests_7433 | False | 2328 | 440.18 | 346481 | 43/43 | 0 | 0 |
| requests_7427 | False | 1689 | 530.35 | 282393 | 34/34 | 0 | 0 |
| requests_7315 | False | 3040 | 515.45 | 441655 | 53/53 | 0 | 0 |
| requests_7328 | False | 4475 | 573.79 | 271256 | 32/32 | 0 | 0 |
| requests_7309 | False | 2303 | 229.11 | 304328 | 38/38 | 0 | 0 |
| requests_7205 | False | 503 | 365.63 | 340857 | 41/41 | 0 | 0 |
| rich_4070 | False | 4530 | 277.88 | 414210 | 43/43 | 0 | 0 |
| rich_4077 | True | 396 | 274.36 | 384294 | 42/42 | 0 | 0 |
| rich_4079 | False | 0 | 345.59 | 297626 | 46/46 | 0 | 1 |
| rich_4076 | False | 3771 | 331.80 | 393593 | 43/43 | 0 | 0 |
| rich_4075 | False | 220 | 236.67 | 362808 | 44/44 | 0 | 0 |
| rich_4006 | False | 885 | 522.86 | 90496 | 11/12 | 0 | 0 |
| fastapi_14258 | False | 1202 | 333.01 | 326317 | 41/41 | 0 | 0 |
| fastapi_14851 | True | 0 | 341.53 | 400859 | 46/46 | 0 | 0 |
| fastapi_14873 | True | 1681 | 358.93 | 408546 | 45/45 | 0 | 0 |
| fastapi_13920 | False | 3669 | 484.71 | 331479 | 36/36 | 0 | 0 |
| fastapi_14794 | False | 1062 | 263.43 | 147458 | 16/16 | 0 | 0 |
| fastapi_14786 | True | 2014 | 139.70 | 139416 | 20/20 | 0 | 0 |

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
- requests_7502: 1 observed request(s) lack usage; totals are lower bounds
- requests_7433: live/ATIF observed token totals differ; check additional helper or compaction requests
- requests_7427: live/ATIF observed token totals differ; check additional helper or compaction requests
- requests_7315: live/ATIF observed token totals differ; check additional helper or compaction requests
- requests_7328: live/ATIF observed token totals differ; check additional helper or compaction requests
- requests_7309: live/ATIF observed token totals differ; check additional helper or compaction requests
- requests_7205: live/ATIF observed token totals differ; check additional helper or compaction requests
- rich_4070: live/ATIF observed token totals differ; check additional helper or compaction requests
- rich_4077: live/ATIF observed token totals differ; check additional helper or compaction requests
- rich_4076: live/ATIF observed token totals differ; check additional helper or compaction requests
- rich_4075: live/ATIF observed token totals differ; check additional helper or compaction requests
- rich_4006: 1 observed request(s) lack usage; totals are lower bounds
- fastapi_14258: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_14851: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_14873: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_13920: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_14794: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_14786: live/ATIF observed token totals differ; check additional helper or compaction requests
- Task-type comparisons need labels assigned before testing

## Validation coverage

- helper_invocation_observed: True
- nonempty_patch_observed: True
- verification_exit_observed: True
- solved_task_observed: True
- configuration_snapshot_present: True
