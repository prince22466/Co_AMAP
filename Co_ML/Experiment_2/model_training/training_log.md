# Training Log

## As of May 13rd 2026, this experiment is finished. best model is RandomForestClassifier as shown below. But most important is working experience with Codex with SKILLs.

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
| 005     | 2026-05-03 | codex  | cpu     | full              | RandomForestClassifier | GridSearchCV with scoring="accuracy" over multiple RF parameter sets; select best CV model | 0.5 | no |
| 006     | 2026-05-04 | codex  | cpu     | full              | RandomForestClassifier | GridSearchCV with scoring="balanced_accuracy" selection for class imbalance | 0.5673076923076923 | no |
| 007     | 2026-05-04 | codex  | cpu     | full              | RandomForestClassifier | GridSearchCV with scoring="average_precision" over RF hyperparameters | 0.5 | no |
| 008     | 2026-05-04 | codex  | cpu     | full              | RandomForestClassifier | GridSearchCV with scoring="f1" over RF hyperparameters | 0.5576923076923077 | no |
| 009     | 2026-05-04 | codex  | cpu     | full              | RandomForestClassifier + LogisticRegression | classifier-agnostic grid with balanced_accuracy (3-fold CV) | 0.7019230769230769 | no |
| 010     | 2026-05-05 | codex  | cpu     | full              | RandomForest-style Bagging(Tree+LeafLogistic) | GridSearchCV on bagged tree+leaf-logistic ensemble (scoring=balanced_accuracy, threshold > 0.6) | 0.5192307692307692 | no |
| 011     | 2026-05-06 | codex  | cpu     | full              | KNeighborsClassifier | deterministic class-balanced training subset + GridSearchCV balanced_accuracy | 0.6634615384615384 | no |

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


### ModelID: 006

Date:

2026-05-04

Notebook:

model_training/train_nb/m_006.ipynb

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
rf__n_estimators: [100, 200, 400]
rf__max_depth: [None, 8, 16, 24]
rf__min_samples_split: [2, 10, 20]
rf__min_samples_leaf: [1, 4, 8]
rf__class_weight: [None, balanced, balanced_subsample]
cv: StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scoring: balanced_accuracy
```

Validation Score:

0.5673076923076923

Model Saved:

no

Notes:

* Created as a new RandomForest grid-search experiment focused on class-imbalance-aware model selection using balanced accuracy.
* Notebook executed end-to-end with no errors.
* Best CV params: rf__class_weight=balanced, rf__max_depth=8, rf__min_samples_leaf=4, rf__n_estimators=100.
* Best CV balanced_accuracy: 0.5480901015654127.
* Validation score from official scoring script: 0.5673076923076923.
* Notebook preserves required data checks, official scoring-script integration, and artifact policy from prior experiments.

Next Recommendation:

Execute `m_006.ipynb`, compare balanced-accuracy-selected parameters against `m_005`, then evaluate recall/precision tradeoff via threshold sweep.

### ModelID: 007

Date:

2026-05-04

Notebook:

model_training/train_nb/m_007.ipynb

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
rf__n_estimators: [100, 300]
rf__max_depth: [8, None]
rf__min_samples_leaf: [1, 4]
rf__class_weight: [None, balanced]
cv: StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
scoring: average_precision
```

Validation Score:

0.5

Model Saved:

no

Notes:

* Notebook executed end-to-end with no errors.
* Best CV params: rf__class_weight=None, rf__max_depth=8, rf__min_samples_leaf=4, rf__n_estimators=300.
* Best CV average_precision: 0.02500060006205744.
* Validation score from official scoring script: 0.5.
* Dropped all-null training columns before fitting to keep preprocessing robust.

Next Recommendation:

Run a threshold sweep and probability calibration (e.g., Platt/isotonic) on top of the selected RF to improve precision-recall behavior under class imbalance.

---

### ModelID: 008 (new experiment)

Date:

2026-05-04

Notebook:

model_training/train_nb/m_008.ipynb

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

RandomForestClassifier with preprocessing pipeline and GridSearchCV

Key Parameters:

```yaml
cv: StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
scoring: f1
param_grid:
  rf__n_estimators: [200, 400]
  rf__max_depth: [8, null]
  rf__min_samples_leaf: [1, 4]
  rf__class_weight: [null, balanced]
best_params:
  rf__class_weight: balanced
  rf__max_depth: 8
  rf__min_samples_leaf: 1
  rf__n_estimators: 400
```

Validation Score:

0.5576923076923077

Model Saved:

no

Model Size:

N/A (artifact intentionally not committed)

Notes:

* All-null training columns are dropped before preprocessing to avoid empty-feature issues.
* Preprocessing uses median imputation for numeric/bool and most-frequent + one-hot for categoricals.
* Official validation scorer loaded from `model_training/help_stuff/validation_score.py`.
* Notebook executed end-to-end successfully.

Next Recommendation:

Try class-imbalance-focused alternatives (e.g., balanced subsampling, threshold tuning after probability calibration) and compare against this `m_008` baseline using the same official score.

---

---

### ModelID: 009 (new experiment)

Date:

2026-05-04

Notebook:

model_training/train_nb/m_009.ipynb

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

Classifier-agnostic pipeline (RandomForestClassifier + LogisticRegression) with GridSearchCV

Key Parameters:

```yaml
cv: StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scoring: balanced_accuracy
param_grid:
  - clf: RandomForestClassifier
    clf__n_estimators: [120]
    clf__max_depth: [3, 5]
    clf__min_samples_leaf: [2]
    clf__class_weight: [balanced]
  - clf: LogisticRegression
    clf__C: [1.0]
    clf__class_weight: [balanced]
```

Validation Score:

0.7019230769230769

Model Saved:

no

Notes:

* Uses official validation scorer at `model_training/help_stuff/validation_score.py`.

Next Recommendation:

Run probability calibration and threshold tuning after selecting the best estimator from this grid.

---

### ModelID: 010 (executed)

Date:

2026-05-05

Notebook:

model_training/train_nb/m_010.ipynb

Model file:

(not saved in repo; per artifact policy)

Validation prediction file:

(not generated)

Runner:

codex

Machine:

cpu

Data Scope:

full

Model Type:

BaggingClassifier (random-forest style ensemble) of shallow DecisionTree partitions with LogisticRegression models trained at leaves

Key Parameters:

```yaml
base_tree_max_depth: 3
base_tree_min_samples_leaf: 120
leaf_logistic_solver: liblinear
leaf_logistic_class_weight: balanced
n_estimators: 20 (base estimator for search)
max_samples: 0.8
grid_n_estimators: [20, 40]
grid_scoring: balanced_accuracy
cv: StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
best_cv_balanced_accuracy: 0.5313727096985049
best_params.clf__n_estimators: 20
max_features: 0.8
threshold: 0.6 (predict 1 if prob > 0.6 else 0)
```

Validation Score:

0.5192307692307692

Model Saved:

no

Model Size:

N/A

Notes:

* Implemented as custom `TreeWithLogisticLeaves` base estimator.
* Ensemble implemented with `BaggingClassifier` to mimic random-forest behavior while allowing logistic experts at leaves.
* Uses median/mode imputation and one-hot encoding preprocessing.
* Notebook executed successfully via nbconvert execution.
* GridSearchCV used with scoring=`balanced_accuracy` over bagging-level parameters.
* Best CV balanced_accuracy: 0.5313727096985049.
* Decision threshold set to: predict 1 if probability > 0.6 else 0.
* Notebook includes required shape/alignment/target-rate checks.

Next Recommendation:

Tune threshold and/or calibrate probabilities for the leaf-logistic ensemble, then compare against prior RF experiments (m_004–m_009).

---

### ModelID: 011 (executed)

Date:

2026-05-06

Notebook:

model_training/train_nb/m_011.ipynb

Model file:

(not saved in repo; artifact intentionally omitted per PR policy)

Validation prediction file:

(not generated; validation scored in-memory via official scoring script)

Runner:

codex

Machine:

cpu

Data Scope:

full

Model Type:

KNeighborsClassifier with deterministic class-balanced training subset

Key Parameters:

```yaml
balanced_training_subset:
  positive_rows: 186
  negative_rows: 186
  negative_to_positive_ratio: 1
  random_state: 42
preprocessing:
  numeric: median imputation + StandardScaler
  categorical: constant missing-value imputation + OneHotEncoder(handle_unknown='ignore')
cv: StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_scoring: balanced_accuracy
param_grid:
  knn__n_neighbors: [3, 5, 7, 11, 15, 21]
  knn__weights: [uniform, distance]
  knn__p: [1, 2]
  knn__metric: [minkowski]
best_cv_balanced_accuracy: 0.75
best_params:
  knn__metric: minkowski
  knn__n_neighbors: 11
  knn__p: 1
  knn__weights: distance
```

Validation Score:

0.6634615384615384

Model Saved:

no

Model Size:

0.32105350494384766 MB (estimated in-memory pickle size; not persisted)

Notes:

* Uses official validation scorer at `model_training/help_stuff/validation_score.py`.
* Notebook executed successfully via nbconvert execution.
* Validation predictions were generated in memory only.
* Confusion matrix on validation data: [[30, 22], [13, 39]].
* Validation prediction distribution: {0: 43, 1: 61}.
* Model binary intentionally omitted from the repository to follow the PR artifact policy.

Next Recommendation:

Try KNN probability-threshold tuning or calibrated distance-weighted KNN variants, selecting thresholds only with training cross-validation before validation scoring.
