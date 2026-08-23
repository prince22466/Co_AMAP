# SKILL: Model Training Notebook Audit

## Purpose

Use this skill to audit model training notebooks for principle errors that can invalidate validation scores or create hidden leakage.

## Scope

- Primary target: notebooks in `model_training/train_nb/`.
- Do **not** overwrite existing training notebooks during audit.
- Write findings into `model_audit/model_audit_log.md`.

## Audit Checklist

### 1) Data split and usage correctness

- Confirm train factor + train target are used for all **training-stage operations**: model fitting, calibration, threshold tuning, feature selection, and hyperparameter search.
- Confirm validation factor/target are **not** used in those training-stage operations (validation should be for unbiased evaluation unless explicitly marked as a non-holdout dev set).
- Confirm validation is not used for hyperparameter/model selection unless explicitly documented as non-holdout.
- Flag any step where validation influences threshold/model choice.

### 2) Leakage checks

- Check for target included in feature matrix directly/indirectly.
- Check for post-outcome fields used as predictors.
- Check for preprocessing fit on train+validation together.
- Check for target encoding or aggregations fitted with validation labels.

### 3) Row/ID integrity

- Check factor/target row alignment assumptions.
- Check index/ID consistency between factor and target files.
- Check train-vs-validation ID overlap and duplicate IDs.
- Check prediction ordering assumptions before scoring.

### 4) Metric and evaluation correctness

- Verify official scorer usage where required.
- Detect metric misuse under class imbalance (e.g., only accuracy).
- Verify threshold tuning strategy does not contaminate holdout score.

### 5) Output and reproducibility hygiene

- Check for accidental overwrite of prior artifacts or logs.
- Confirm model ID naming and artifact naming consistency.
- Confirm notebook can run top-to-bottom without hidden state.

## Severity guidance

- **Critical**: invalidates holdout estimate (e.g., val used for model selection).
- **High**: strong leakage risk or wrong label-feature alignment.
- **Medium**: metric misuse or insufficient diagnostics.
- **Low**: process/documentation hygiene issues.

## Required audit output format

In `model_audit/model_audit_log.md`, include:

1. Date + audited files.
2. Executive summary.
3. Checklist results with PASS/FAIL/WARNING.
4. Severity-ranked findings.
5. Concrete remediation actions.
6. Explicit statement whether training notebook code was modified (should be no for pure audits).
