# Model Audit Log

Date: 2026-05-02  
Audited notebook: `model_training/train_nb/m_001.ipynb`  
Related files reviewed: `model_training/help_stuff/validation_score.py`, `model_training/training_log.md`

## Executive Summary

- The current notebook has a **critical validation mistake**: it selects the decision threshold by maximizing score on the validation set itself.
- No direct evidence was found of fitting on validation labels or fitting preprocessing on train+validation together.
- Evaluation relies on accuracy-only official scoring, which is risky for this imbalanced target and should be supplemented.
- Train/validation IDs should be explicitly audited each run (overlap and ordering), as scorer compares by position.

## Checklist Results

| Check | Status | Notes |
|---|---|---|
| Validation used for selecting model/threshold | **FAIL** | Threshold grid is searched against validation score. |
| Validation data used for model fit | PASS | `fit` is called with train factor/target only. |
| Validation labels used during training | PASS | No `y_val` passed into training pipeline. |
| Preprocessing fit on train+validation | PASS | Preprocessor is inside sklearn pipeline fitted on train. |
| Target leakage features | WARNING | Dataset appears to include fields potentially post-outcome; requires business-time validation. |
| Duplicate rows/IDs across train/validation | WARNING | Must be checked per run; not enforced in notebook. |
| ID alignment consistency (factor vs target) | WARNING | No explicit assertions in notebook for index equality. |
| Prediction order correctness | WARNING | Official scorer resets index and assumes row order is correct. |
| Metric misuse | WARNING | Accuracy-only scoring under heavy imbalance can mask poor recall. |
| Accidental overwrite of old outputs | PASS | Notebook does not currently auto-overwrite model/prediction artifacts. |

## Severity-Ranked Findings

## Critical

1. **Validation contamination via threshold tuning**  
   The notebook evaluates multiple thresholds on validation predictions and chooses the best threshold using validation score. This uses holdout validation for model-selection decisions and inflates the reported validation estimate.

## Medium

1. **Metric blind spot under imbalance**  
   Accuracy-only metric can hide very poor positive-class recall.

2. **Ordering/alignment risk not guarded**  
   Scoring is positional; no explicit index integrity checks in notebook make silent misalignment possible if upstream data order changes.

## Recommended Remediation

1. Reserve external validation as a pure holdout. Tune threshold on an internal split or cross-validation within training data only.
2. Add secondary diagnostics (precision, recall, F1, confusion matrix, PR-AUC) alongside official score.
3. Add explicit ID/index integrity checks and train-validation overlap checks before fitting/scoring.
4. Add an evaluation policy note in notebook header defining what data may be used for model-selection steps.

## Notebook Modification Statement

- Training notebook was **not modified** in this audit task.
- Findings were documented only in `model_audit/` artifacts.

---

## Follow-up (2026-05-02)

Status of findings for `model_training/train_nb/m_001.ipynb` at follow-up check:

- **Critical finding (validation contamination via threshold tuning): RESOLVED** based on notebook logic review (threshold tuning is performed with training-only cross-validation, not validation-set selection).
- **Medium finding (metric blind spot under imbalance): REMAINS OPEN**.
- **Medium finding (ordering/alignment risk): REMAINS OPEN**.

Follow-up note:
- This section records current status (solved/improved/open) for each prior finding.
- If a newer notebook revision addresses any finding, add a dated follow-up entry with concrete code/run evidence and update only the status lines above (keep original audit narrative unchanged).
- **Validation subset class-imbalance issue: RESOLVED** for the uploaded subset (previously reported as ~1000+ class-0 vs ~5 class-1, now near-balanced).
- Data-provider update: validation target class imbalance is materially improved versus the prior heavily skewed upload (previously reported as ~1000+ class-0 vs ~5 class-1, now near-balanced in the uploaded subset).
- Constraint acknowledged: only a small validation subset could be uploaded to GitHub in this cycle; treat validation metrics as lower-confidence due to limited sample size and reduced comparability to earlier larger-validation runs.

---

## Follow-up (2026-05-03)

Current review of `model_training/train_nb/m_001.ipynb` after the refreshed training data:

- **Validation contamination via threshold tuning: RESOLVED**. Current notebook logic tunes the threshold with 5-fold `StratifiedKFold` on training data only, then fits the final model on `X_train, y_train` and scores validation once.
- **Validation data used in model fitting/selection: PASS**. No direct evidence found that validation factors or validation labels are used in model fitting, preprocessing fitting, or threshold selection.
- **Feature leakage risk: PARTIALLY OPEN**. The notebook currently auto-includes all columns from the factor CSVs. A subset of columns appears to represent post-outcome or removal state and should be removed from prepared training/validation factor data before rerunning the model.

Feature-leakage clarification:

- User confirmed these columns are acceptable and should not be treated as leakage by this audit unless future data-timing evidence says otherwise: `Scheduled Principal Current`, `Total Principal Current`, `Loan Payment History`, `Modification Flag`, `Servicing Activity Indicator`, `Borrower Credit Score Current`, `Co-Borrower Credit Score Current`.
- Columns still flagged for removal from model factors: `Zero Balance Code`, `Zero Balance Effective Date`, `Zero Balance Code Change Date`, `UPB at the Time of Removal`.
- `Repurchase Date` remains a review item because it may represent post-acquisition/removal information; exclude it unless confirmed available at acquisition time.

Recommended next action:

1. Update `data_prepare/dataprepare_training.ipynb` to exclude the zero-balance/removal columns from both train and validation factor outputs.
2. Regenerate `model_training/training_data/train_df_factor.csv` and `model_training/val_data/val_df_factor.csv` with matching schemas.
3. Rerun `model_training/train_nb/m_001.ipynb` and record the refreshed score, explicitly noting that leakage-prone zero-balance/removal fields were removed.

---

## Follow-up (2026-05-03, scoring audit)

Reason for audit:

- `m_001` was retrained on updated train/validation data after leakage-prone zero-balance/removal columns were removed.
- Reported validation score remained `0.5`, which looked suspicious and required checking the official score calculation.

Files/data reviewed:

- `model_training/help_stuff/validation_score.py`
- `model_training/train_nb/m_001.ipynb`
- `model_training/training_data/train_df_factor.csv`
- `model_training/training_data/train_df_target.csv`
- `model_training/val_data/val_df_factor.csv`
- `model_training/val_data/val_df_target.csv`

Scoring conclusion:

- **No current principle mistake was found in the retrained `m_001` workflow.**
- Previously open principle findings are now considered resolved: validation is not used for fitting, preprocessing fitting, threshold selection, or model selection; the official scoring script is being used correctly.
- **Validation score calculation is correct for the current notebook output.**
- `validation_score.py` computes positional accuracy by loading `model_training/val_data/val_df_target.csv`, resetting both indices, checking equal length, and returning `1 - incorrect_count / total_count`.
- Recomputed official score and direct accuracy both equal `0.5`.
- The scorer's target file matches the loaded `y_val`; no stale-target path issue was found.

Reproduction evidence:

```text
train shape: (45062, 74)
validation shape: (104, 74)
train target counts: {0: 44876, 1: 186}
validation target counts: {0: 52, 1: 52}
schema match: true
zero-balance/removal columns present: false

selected threshold: 0.95
official score: 0.5
direct accuracy: 0.5
validation prediction counts: {0: 102, 1: 2}
confusion matrix rows=true [0,1], cols=pred [0,1]: [[51, 1], [51, 1]]
```

Root cause of the surprising `0.5`:

- The score is not a scorer bug.
- The model's train-only threshold tuning optimizes ordinary accuracy on a heavily imbalanced training distribution (`186` positives out of `45062` rows).
- That objective chooses the very high threshold `0.95`, which behaves almost like an all-negative classifier.
- The validation subset is exactly balanced (`52` negatives and `52` positives), so an almost-all-negative classifier naturally scores about `0.5`.
- Threshold diagnostics from the same validation probabilities show that other thresholds would score differently, for example:
  - threshold `0.50`: accuracy `0.6730769230769231`
  - threshold `0.20`: accuracy `0.6153846153846154`
  - threshold `0.95`: accuracy `0.5`
- These alternative threshold scores are diagnostic only. Selecting a threshold from validation results would reintroduce validation-set model-selection leakage.

Additional audit findings:

- The remaining notes below are engineering/modeling follow-ups, not principle violations.
- Current `m_001.ipynb` file output appears stale relative to regenerated CSVs: notebook output still records `(45062, 81)` / `(104, 81)`, while current CSVs are `(45062, 74)` / `(104, 74)`. The training log records the refreshed 74-column run, but the notebook artifact itself should be rerun/saved if it is intended to document the latest experiment state.
- There are many all-null factor columns after the data refresh:
  - all-null in train: `34` columns
  - all-null in validation: `35` columns
  - non-null in train but all-null in validation: `Repurchase Make Whole Proceeds Flag`
- These all-null columns are part of the observed data condition and are not a data-preparation principle mistake. They are skipped by `SimpleImputer`, create repeated warnings, and are not the direct cause of the exact `0.5` score. Future model pipelines should handle them explicitly as normal model-building hygiene.

Recommended next action:

1. Keep `validation_score.py` unchanged as the official accuracy scorer.
2. In the next model experiment, do not tune threshold with plain accuracy on the naturally imbalanced training folds if the validation/evaluation subset is balanced. Use a documented training-only threshold policy, such as balanced internal folds, class-balanced accuracy, or a fixed threshold chosen before validation.
3. Add secondary diagnostics to the notebook: confusion matrix, prediction class counts, ROC-AUC, PR-AUC, precision, recall, and F1.
4. Add explicit all-null column handling before model fitting.
5. Rerun and save `m_001.ipynb` if it is meant to represent the latest 74-column data run.

---

## Follow-up (2026-05-03, m_002 audit)

Reason for audit:

- `m_002` was created as a LogisticRegression variation with a fixed threshold of `0.6`.
- Prior recommendations for `m_002` mentioned explicit all-null column handling, so the notebook was checked for validation correctness and experiment hygiene.

Files/data reviewed:

- `model_training/train_nb/m_002.ipynb`
- `model_training/help_stuff/validation_score.py`
- `model_training/training_log.md`
- `model_training/training_data/train_df_factor.csv`
- `model_training/training_data/train_df_target.csv`
- `model_training/val_data/val_df_factor.csv`
- `model_training/val_data/val_df_target.csv`

Audit conclusion:

- **No principle validation or leakage mistake was found in `m_002`.**
- `m_002` trains on training data only.
- Validation scoring uses the official `help_stuff/validation_score.py` script.
- The scorer's validation target matches the loaded `y_val`.
- The reported validation score is reproducible: `0.6730769230769231`.
- No forbidden `m_002` model or validation prediction artifacts were found.

Reproduction evidence:

```text
train shape: (45062, 74)
validation shape: (104, 74)
schema match: true
train positive rate: 0.004127646353912388
validation positive rate: 0.5
fixed threshold: 0.6
official score: 0.6730769230769231
direct accuracy: 0.6730769230769231
validation prediction counts: {0: 66, 1: 38}
scoring target match: true
```

Improvement findings:

- `m_002` was expected/recommended to handle all-null columns, but the notebook still passes all numeric columns directly into `SimpleImputer(strategy="median")`.
- The executed notebook still emits `SimpleImputer` warnings for all-null numeric features.
- Notebook structure is thinner than `help_stuff/notebook_template.ipynb`; it omits explicit artifact-policy and training-log-entry sections.
- Result reporting uses the misleading field name `top_cv_thresholds` even though `m_002` uses a fixed threshold and does not perform CV threshold tuning.
- CSV loading emits a mixed-type `DtypeWarning`; use `low_memory=False` or explicit dtype handling for cleaner reproducibility.

Recommended next action:

1. Treat the current `m_002` score as valid for its fixed-threshold experiment.
2. In the next experiment or an audit-fix rerun, explicitly drop or constant-fill all-null columns before imputation.
3. Clean reporting labels so fixed-threshold output is not labeled as CV threshold output.
4. Align the notebook sections more closely with `help_stuff/notebook_template.ipynb`.

---

## Follow-up (2026-05-03, m_003 audit)

Reason for audit:

- `m_003` was created as a `DecisionTreeClassifier` experiment with fixed threshold `0.6`.
- The notebook was checked for validation correctness, leakage risk, row/ID integrity, scorer usage, and experiment hygiene.

Files/data reviewed:

- `model_training/train_nb/m_003.ipynb`
- `model_training/help_stuff/validation_score.py`
- `model_training/training_log.md`
- `model_training/training_data/train_df_factor.csv`
- `model_training/training_data/train_df_target.csv`
- `model_training/val_data/val_df_factor.csv`
- `model_training/val_data/val_df_target.csv`

Executive summary:

- **No critical validation contamination was found in `m_003`.**
- `m_003` fits preprocessing and the decision tree on training data only.
- Validation is used once for evaluation with a predeclared fixed threshold of `0.6`.
- The official score is reproducible and matches direct accuracy: `0.6634615384615384`.
- Main open issue: the notebook includes `Unnamed: 0` as a predictor, and the trained tree gives it non-trivial feature importance (`0.0647`). This is likely a persisted CSV index and should be removed or explicitly justified before trusting model behavior.

Checklist results:

| Check | Status | Notes |
|---|---|---|
| Validation used for selecting model/threshold | PASS | Threshold is fixed at `0.6`; no validation threshold search was found. |
| Validation data used for model fit | PASS | `model.fit(X_train, y_train)` only. |
| Validation labels used during training | PASS | No `y_val` passed into model fitting, preprocessing fitting, or threshold selection. |
| Preprocessing fit on train+validation | PASS | Preprocessor is inside sklearn pipeline fitted on training data only. |
| Official scorer usage | PASS | Uses `help_stuff.validation_score.prediction_score`; scorer target matches loaded `y_val`. |
| Target directly included in features | PASS | No `Prepaied_3m`, target, or label column found in factor data. |
| Post-outcome zero-balance/removal fields | PASS | Previously flagged zero-balance/removal columns are not present. |
| Row/ID overlap checks | WARNING | Factor files contain no obvious loan ID column, so train/validation ID overlap cannot be verified from current inputs. |
| Factor/target row alignment | WARNING | Notebook checks shapes but has no factor-target key assertion; scoring remains positional. |
| CSV index feature handling | WARNING | `Unnamed: 0` is used as a numeric predictor and appears in the tree's top features. |
| All-null feature handling | WARNING | Many all-null numeric columns are passed to `SimpleImputer`, causing repeated sklearn warnings. |
| Artifact policy | PASS | No committed `m_003` model, validation prediction CSV, or feature-importance CSV was found. |
| Notebook run evidence | WARNING | Existing notebook outputs show a completed run, but local re-execution via nbconvert could not be performed because `jupyter-nbconvert` is not installed in this environment. |

Reproduction evidence:

```text
train shape: (45062, 74)
validation shape: (104, 74)
train target counts: {0: 44876, 1: 186}
validation target counts: {0: 52, 1: 52}
schema match: true
duplicate columns: 0
missing train / validation: 1674885 / 3879
all-null columns in train / validation: 34 / 35
train-not-val all-null column: Repurchase Make Whole Proceeds Flag
fixed threshold: 0.6
official score: 0.6634615384615384
direct accuracy: 0.6634615384615384
scoring target match: true
validation prediction counts: {0: 57, 1: 47}
confusion matrix rows=true [0,1], cols=pred [0,1]: [[37, 15], [20, 32]]
top feature importance for persisted index: num__Unnamed: 0 = 0.06471847461215792
```

Severity-ranked findings:

## Medium

1. **Persisted CSV index is being used as a model feature**

   `Unnamed: 0` is included in `numeric_cols`, passed through preprocessing, and used by the decision tree. Reproduced feature importances show `num__Unnamed: 0` as the fifth-largest feature (`0.0647`). This can encode row order or upstream data-preparation artifacts rather than loan characteristics. It does not prove target leakage by itself, but it weakens the validity of model interpretation and may inflate or destabilize validation behavior.

2. **Row/ID integrity cannot be fully audited from current factor files**

   The factor data contains no obvious loan identifier column. The notebook checks row counts and schema, but cannot verify train/validation ID overlap or factor-target alignment by key. Since the official scorer compares predictions to labels by position, silent upstream row-order mistakes would not be caught.

## Low

1. **All-null feature handling remains noisy and implicit**

   `SimpleImputer(strategy="median")` skips many all-null numeric features, producing repeated warnings. This is not a validation principle violation, but the pipeline should explicitly drop or constant-fill all-null columns for cleaner reproducibility.

2. **Notebook copy/paste labels are stale**

   Section 4 says "Fit logistic regression" even though `m_003` uses `DecisionTreeClassifier`, and the result field is still named `top_cv_thresholds` even though no CV threshold tuning is performed.

3. **CSV load emits mixed-type warning**

   `Repurchase Make Whole Proceeds Flag` triggers a mixed-type warning under default `pd.read_csv`. Use `low_memory=False` or explicit dtype handling.

4. **Notebook structure is thinner than the canonical template**

   The notebook has the required core flow, but it omits explicit artifact-policy and training-log-entry sections from the template style.

Recommended remediation:

1. Remove `Unnamed: 0` from model factors in data preparation or drop it explicitly in the notebook before preprocessing, then rerun as either an audit-fix rerun or a new experiment if the modeling behavior changes materially.
2. Add explicit all-null column handling before `SimpleImputer`.
3. If a stable loan identifier exists upstream, preserve it for audit checks but exclude it from model features; use it to verify train/validation overlap and factor-target alignment.
4. Rename fixed-threshold reporting fields and text so the notebook does not imply CV threshold tuning or logistic regression.
5. Re-execute `m_003.ipynb` top-to-bottom in an environment with `jupyter-nbconvert` installed, or document the direct-code reproduction as the audit evidence.

Notebook modification statement:

- Training notebook code was **not modified** in this audit task.
- This audit appended documentation only to `model_audit/model_audit_log.md`.
