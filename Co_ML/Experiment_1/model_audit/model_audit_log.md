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
