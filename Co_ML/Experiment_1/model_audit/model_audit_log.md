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
