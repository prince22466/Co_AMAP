# Training Log

## Experiment Summary

| ModelID | Date       | Runner | Machine | data | Model Type          | Key Idea | Validation Score | Model Saved |
| ------- | ---------- | ------ | ------- | ----- | ------------------- | -------- | ---------------- | ----------- |
| 001     | YYYY-MM-DD | codex  | cpu     | data on git repo  | TBD                 | set up baseline | TBD              | TBD         |

---

## Detailed Records

### ModelID: 001

Notebook:

model_training/train_nb/m_001.ipynb

Python file:

model_training/train_nb/m_001.py

Model file:

(not saved in repo; removed to satisfy PR restrictions)

Runner:

codex

Machine:

cpu

Data Scope:

git repo

Model Type:

TBD

Model Structure:

TBD

Key Parameters:

```yaml
TBD
```

Features Used:

Preprocessing:

Validation Score:

Model Saved:

Model Size:

Notes:

---
Preprocessing:

Validation Score:

Model Saved:

Model Size:

Notes:

Next Recommendation:

### ModelID: 001 (executed)

Date:

2026-05-01

Notebook:

model_training/train_nb/m_001.ipynb

Python file:

model_training/train_nb/m_001.py

Model file:

(not saved in repo; removed to satisfy PR restrictions)

Validation prediction file:

(not generated; validation scored in-memory via official scoring script)

Runner:

codex

Machine:

cpu

Data Scope:

full

Model Type:

LogisticRegression (class_weight=balanced) with median/mode imputation and one-hot encoding

Key Parameters:

```yaml
solver: liblinear
max_iter: 1000
class_weight: balanced
threshold_candidates: [0.05, 0.10, ..., 0.95]
best_threshold: 0.95
```

Validation Score:

0.9858326650628174

Model Saved:

no

Model Size:

N/A (artifact intentionally not committed)

Notes:

* Train shape: (14964, 81)
* Validation shape: (3741, 81)
* Train positive rate: 0.0010024057738572574
* Validation positive rate: 0.0010692328254477412
* Schema match: true
* Duplicate columns: 0

Next Recommendation:

Try tree-based model (e.g., HistGradientBoosting with feature pruning) and compare recall-focused thresholding while still reporting official validation score.
