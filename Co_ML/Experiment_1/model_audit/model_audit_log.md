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
