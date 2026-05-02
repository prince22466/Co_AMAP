# Training Log

## Experiment Summary

| ModelID | Date       | Runner | Machine | data | Model Type          | Key Idea | Validation Score | Model Saved |
| ------- | ---------- | ------ | ------- | ----- | ------------------- | -------- | ---------------- | ----------- |
| 001     | 2026-05-01 | codex  | cpu     | full              | LogisticRegression  | baseline with preprocessing + threshold tuning | 0.9858326650628174 | no |

---

## Detailed Records

### ModelID: 001 (executed)

Date:

2026-05-01

Notebook:

model_training/train_nb/m_001.ipynb

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

---

### ModelID: 001 (audit-fix rerun)

Date:

2026-05-02

Notebook:

model_training/train_nb/m_001.ipynb

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
threshold_selection: 5-fold StratifiedKFold on training split only
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

* Addressed audit issues in-place in `m_001` notebook.
* Added required checks: target/factor shapes, schema match, missing values, duplicate columns, numeric/categorical counts.
* Added leakage-pattern scan and row-hash diagnostics for alignment guards.
* Threshold now tuned via train-only cross-validation (validation no longer used for threshold search).
* Added scoring-contract check confirming validation labels used by script match loaded `y_val`.
* Train shape: (14964, 81)
* Validation shape: (3741, 81)
* Train positive rate: 0.0010024057738572574
* Validation positive rate: 0.0010692328254477412
* Scoring target match check: true

Next Recommendation:

Create `m_002` with explicit all-null column handling (drop or constant fill before imputation) to remove repeated warnings and evaluate calibration-aware thresholding.

---

### ModelID: 001 (artifact-install rerun)

Date:

2026-05-02

Notebook:

model_training/train_nb/m_001.ipynb

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
threshold_selection: 5-fold StratifiedKFold on training split only
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

* Installed notebook execution artifacts: `jupyter`, `nbformat`, `nbclient`, `ipykernel`.
* Executed `train_nb/m_001.ipynb` in-place using nbconvert.
* Result checks remained stable: `scoring_target_match=true`, `direct_accuracy_check=0.9858326650628174`.
* Train shape: (14964, 81)
* Validation shape: (3741, 81)
* Missing (train/val): 661150 / 165166
* Leakage flagged columns: []

