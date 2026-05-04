# Training Log

## Experiment Summary

| ModelID | Date       | Runner | Machine | data | Model Type          | Key Idea | Validation Score | Model Saved |
| ------- | ---------- | ------ | ------- | ----- | ------------------- | -------- | ---------------- | ----------- |
| 001     | 2026-05-01 | codex  | cpu     | full              | LogisticRegression  | baseline with preprocessing + threshold tuning | 0.9858326650628174 | no |
| 001     | 2026-05-02 | codex  | cpu     | full              | LogisticRegression  | retrain on updated train/val data + same preprocessing + CV threshold tuning | 0.5 | no |
| 001     | 2026-05-03 | codex  | cpu     | full              | LogisticRegression  | retrain on updated train/val data + same preprocessing + CV threshold tuning | 0.5 | no |

| 002     | 2026-05-03 | codex  | cpu     | full              | LogisticRegression  | fixed threshold at 0.6 (prob > 0.6 => 1) | 0.6730769230769231 | no |

| 003     | 2026-05-03 | codex  | cpu     | full              | DecisionTreeClassifier | fixed threshold at 0.6 (prob > 0.6 => 1) | 0.6634615384615384 | no |

| 003     | 2026-05-03 | codex  | cpu     | full              | DecisionTreeClassifier | append-only notebook-output capture (fixed threshold 0.6) | 0.6538461538461539 | no |

| 004     | 2026-05-03 | codex  | cpu     | full              | RandomForestClassifier | fixed threshold at 0.6 (prob > 0.6 => 1) | 0.5288461538461539 | no |

| 005     | 2026-05-03 | codex  | cpu     | full              | RandomForestClassifier | GridSearchCV over multiple RF parameter sets; select best CV model | 0.5 | no |

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



---

### ModelID: 001 (data-refresh rerun)

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

0.5

Model Saved:

no

Model Size:

N/A (artifact intentionally not committed)

Notes:

* Retrained `m_001` after training/validation data refresh.
* Train shape: (45062, 81)
* Validation shape: (104, 81)
* Train positive rate: 0.004127646353912388
* Validation positive rate: 0.5
* Schema match: true
* Duplicate columns: 0
* Scoring target match check: true

Next Recommendation:

Create `m_002` with robust handling for all-null columns and evaluate whether the refreshed validation split is representative before model comparison.

### ModelID: 001 (data-refresh rerun)

Date:

2026-05-03

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

0.5

Model Saved:

no

Model Size:

N/A (artifact intentionally not committed)

Notes:

* Retrained `m_001` after another training/validation data refresh.
* Train shape: (45062, 74)
* Validation shape: (104, 74)
* Train positive rate: 0.004127646353912388
* Validation positive rate: 0.5
* Schema match: true
* Duplicate columns: 0
* Missing (train/val): 1674885 / 3879
* Scoring target match check: true

Next Recommendation:

Create `m_002` to explicitly drop all-null columns before imputation and compare with this refreshed split.



### ModelID: 002 (executed)

Date:

2026-05-03

Notebook:

model_training/train_nb/m_002.ipynb

Model file:

(not saved in repo; artifact policy)

Validation prediction file:

(not generated; validation scored in-memory via official script)

Runner:

codex

Machine:

cpu

Data Scope:

full

Model Type:

LogisticRegression with median/mode imputation and one-hot encoding

Key Parameters:

```yaml
solver: liblinear
max_iter: 1000
class_weight: balanced
fixed_threshold: 0.6
decision_rule: prob > 0.6 => 1 else 0
```

Validation Score:

0.6730769230769231

Model Saved:

no

Model Size:

N/A

Notes:

* Created as a variation of `m_001`.
* Replaced cross-validated threshold tuning with a fixed threshold of 0.6.
* Executed `model_training/train_nb/m_002.ipynb` successfully via `jupyter nbconvert --execute --inplace`.



### ModelID: 003 (executed)

Date:

2026-05-03

Notebook:

model_training/train_nb/m_003.ipynb

Model file:

(not saved in repo; artifact policy)

Validation prediction file:

(not generated; validation scored in-memory via official script)

Runner:

codex

Machine:

cpu

Data Scope:

full

Model Type:

DecisionTreeClassifier with median/mode imputation and one-hot encoding

Key Parameters:

```yaml
max_depth: 6
min_samples_leaf: 20
class_weight: balanced
fixed_threshold: 0.6
decision_rule: prob > 0.6 => 1 else 0
```

Validation Score:

0.6634615384615384

Model Saved:

no

Model Size:

N/A

Notes:

* Created as new model `m_003` using a decision tree classifier.
* Executed `model_training/train_nb/m_003.ipynb` successfully via `jupyter nbconvert --execute --inplace`.
* Train shape: (45062, 74)
* Validation shape: (104, 74)
* Schema match: true
* Missing (train/val): 1674885 / 3879
* Scoring target match check: true


### ModelID: 002 (audit-fix rerun)

Date:

2026-05-03

Notebook:

model_training/train_nb/m_002.ipynb

Model file:

(not saved in repo; artifact policy)

Validation prediction file:

(not generated; validation scored in-memory via official script)

Runner:

codex

Machine:

cpu

Data Scope:

full

Model Type:

LogisticRegression with explicit all-null-column drop, median/mode imputation, and one-hot encoding

Key Parameters:

```yaml
solver: liblinear
max_iter: 1000
class_weight: balanced
fixed_threshold: 0.6
decision_rule: prob > 0.6 => 1 else 0
all_null_column_policy: drop_train_all_null_and_apply_same_schema_to_validation
index_column_policy: drop_Unnamed_0_when_present
```

Validation Score:

0.6634615384615384

Model Saved:

no

Model Size:

N/A

Notes:

* Audit-fix rerun of `m_002` after notebook robustness updates.
* Executed `model_training/train_nb/m_002.ipynb` successfully via `jupyter nbconvert --execute --inplace`.
* Added secondary diagnostics: confusion matrix, precision/recall/F1, ROC-AUC, PR-AUC, and prediction counts.
* Scoring target match check: true.


### ModelID: 003 (append-only rerun request log)

Date:

2026-05-03

Notebook:

model_training/train_nb/m_003.ipynb

Model file:

(not saved in repo; artifact policy)

Validation prediction file:

(not generated; validation scored in-memory via official script)

Runner:

codex

Machine:

cpu

Data Scope:

full

Model Type:

DecisionTreeClassifier with median/mode imputation and one-hot encoding

Key Parameters:

```yaml
max_depth: 6
min_samples_leaf: 20
class_weight: balanced
fixed_threshold: 0.6
decision_rule: prob > 0.6 => 1 else 0
```

Validation Score:

0.6538461538461539

Model Saved:

no

Model Size:

N/A

Notes:

* Append-only update per request; no prior training-log records were modified.
* Notebook rerun command attempted: `jupyter nbconvert --execute --inplace model_training/train_nb/m_003.ipynb` and `python -m jupyter ...`; both failed because Jupyter is unavailable in this environment.
* Captured the latest stored notebook output values: fixed threshold = 0.6, validation score = 0.653846, direct accuracy check = 0.653846, scoring target match = true.
* Model implementation in notebook pipeline is `DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=42)`.

### ModelID: 004 (executed)

Date:

2026-05-03

Notebook:

model_training/train_nb/m_004.ipynb

Model file:

(not saved in repo; artifact intentionally omitted)

Validation prediction file:

(not generated; validation scored in-memory via official scoring script)

Runner:

codex

Machine:

cpu

Data Scope:

full

Model Type:

RandomForestClassifier with median/mode imputation + one-hot encoding

Key Parameters:

```yaml
n_estimators: 300
max_depth: 10
min_samples_leaf: 10
class_weight: balanced_subsample
random_state: 42
threshold: 0.6
```

Validation Score:

0.5288461538461539

Model Saved:

no

Model Size:

N/A (artifact intentionally not committed)

Notes:

* Train shape: (45062, 38)
* Validation shape: (104, 38)
* Train positive rate: 0.004127646353912388
* Validation positive rate: 0.5
* Schema match: true
* Missing (train/val): 97716 / 239
* Duplicate columns: 0
* Scoring target match check: true

Next Recommendation:

Evaluate threshold sweep and class-weight/grid search for RandomForest, then compare against boosted-tree models on the same full data scope.



### ModelID: 005 (prepared)

Date:

2026-05-03

Notebook:

model_training/train_nb/m_005.ipynb

Runner:

codex

Machine:

cpu

Data Scope:

full

Model Type:

RandomForestClassifier with preprocessing pipeline + GridSearchCV

Key Parameters (grid):

```yaml
rf__n_estimators: [100, 200]
rf__max_depth: [None, 8, 16]
rf__min_samples_split: [2, 10]
rf__min_samples_leaf: [1, 4]
rf__class_weight: [None, balanced]
cv: StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scoring: accuracy
```

Validation Score:

0.5

Model Saved:

no

Notes:

* Notebook includes required checks and official scoring-script integration.
* Best CV params: rf__class_weight=None, rf__max_depth=8, rf__min_samples_leaf=1, rf__n_estimators=100.
* Best CV score: 0.9958723536460876.
* Added all-null column drop before training and keep_empty_features imputers to avoid execution warnings/errors.

Next Recommendation:

After executing `m_005`, compare best RF setting against `m_001` logistic baseline and consider class-imbalance-focused scoring/tuning.
