# Evaluation archive audit

Source: `submission-6-zip_First30Tasks.zip`
Diagnostics: `2.2-result-serialization`

Integrity errors: **0**.

| Task | Resolved | Patch chars | Task seconds | Observed tokens | Usage coverage | Pending tools | Helper calls |
|---|---|---:|---:|---:|---:|---:|---:|
| fastapi_15661 | False | 0 | 25.26 | 15929 | 3/3 | 0 | 0 |
| fastapi_15588 | False | 571 | 38.49 | 66336 | 8/8 | 0 | 0 |
| fastapi_15589 | False | 773 | 66.51 | 146383 | 17/17 | 0 | 1 |
| fastapi_15030 | False | 5341 | 85.09 | 61159 | 7/7 | 0 | 0 |
| fastapi_14962 | False | 3209 | 48.26 | 75706 | 8/8 | 0 | 0 |
| fastapi_15280 | False | 1245 | 84.67 | 102872 | 12/12 | 0 | 0 |
| fastapi_14964 | False | 1305 | 28.65 | 54941 | 7/7 | 0 | 0 |
| fastapi_15023 | False | 0 | 66.22 | 104880 | 12/12 | 0 | 0 |
| fastapi_14986 | False | 9055 | 206.75 | 189166 | 18/18 | 0 | 1 |
| fastapi_14978 | False | 3035 | 104.02 | 154461 | 16/16 | 0 | 0 |
| requests_7505 | False | 609 | 123.45 | 144943 | 16/16 | 0 | 1 |
| requests_7502 | False | 1276 | 147.72 | 127205 | 19/19 | 0 | 2 |
| requests_7433 | False | 739 | 93.51 | 37831 | 6/6 | 0 | 0 |
| requests_7427 | False | 38053 | 99.76 | 242352 | 28/28 | 0 | 2 |
| requests_7315 | False | 2939 | 110.10 | 173174 | 18/18 | 0 | 0 |
| requests_7328 | False | 0 | 181.18 | 85756 | 13/14 | 0 | 2 |
| requests_7309 | False | 692 | 105.20 | 60664 | 11/11 | 0 | 1 |
| requests_7205 | False | 34741 | 48.11 | 116478 | 13/13 | 0 | 0 |
| rich_4070 | False | 34283 | 105.56 | 190983 | 19/19 | 0 | 0 |
| rich_4077 | False | 884 | 62.80 | 119124 | 17/17 | 0 | 2 |
| rich_4079 | False | 28513 | 163.54 | 203162 | 19/19 | 0 | 0 |
| rich_4076 | False | 793 | 132.87 | 87672 | 9/10 | 0 | 0 |
| rich_4075 | False | 431 | 220.95 | 434958 | 59/59 | 0 | 9 |
| rich_4006 | False | 12548 | 61.15 | 132037 | 15/15 | 0 | 0 |
| fastapi_14258 | False | 0 | 77.77 | 130594 | 13/14 | 0 | 0 |
| fastapi_14851 | False | 0 | 143.19 | 147732 | 14/15 | 0 | 0 |
| fastapi_14873 | False | 0 | 181.99 | 230137 | 25/26 | 0 | 1 |
| fastapi_13920 | False | 1421 | 128.02 | 178756 | 24/24 | 0 | 1 |
| fastapi_14794 | False | 0 | 92.37 | 54287 | 6/7 | 0 | 0 |
| fastapi_14786 | True | 510 | 16.66 | 29773 | 5/5 | 0 | 0 |

## Integrity errors

None recorded.

## Measurement caveats

- fastapi_15589: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_15030: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_15280: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_15023: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_14986: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_14978: live/ATIF observed token totals differ; check additional helper or compaction requests
- requests_7505: live/ATIF observed token totals differ; check additional helper or compaction requests
- requests_7502: live/ATIF observed token totals differ; check additional helper or compaction requests
- requests_7427: live/ATIF observed token totals differ; check additional helper or compaction requests
- requests_7315: live/ATIF observed token totals differ; check additional helper or compaction requests
- requests_7328: 1 observed request(s) lack usage; totals are lower bounds
- requests_7205: live/ATIF observed token totals differ; check additional helper or compaction requests
- rich_4070: live/ATIF observed token totals differ; check additional helper or compaction requests
- rich_4079: live/ATIF observed token totals differ; check additional helper or compaction requests
- rich_4076: live/ATIF observed token totals differ; check additional helper or compaction requests
- rich_4076: 1 observed request(s) lack usage; totals are lower bounds
- rich_4075: live/ATIF observed token totals differ; check additional helper or compaction requests
- rich_4006: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_14258: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_14258: 1 observed request(s) lack usage; totals are lower bounds
- fastapi_14851: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_14851: 1 observed request(s) lack usage; totals are lower bounds
- fastapi_14873: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_14873: 1 observed request(s) lack usage; totals are lower bounds
- fastapi_13920: live/ATIF observed token totals differ; check additional helper or compaction requests
- fastapi_14794: 1 observed request(s) lack usage; totals are lower bounds
- Task-type comparisons need labels assigned before testing

## Validation coverage

- helper_invocation_observed: True
- nonempty_patch_observed: True
- verification_exit_observed: True
- solved_task_observed: True
- configuration_snapshot_present: True
