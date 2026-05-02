# AGENTS.md

## Purpose

This repository trains machine learning models to predict:

Prepaied_3m

Meaning:

* 0 → loan is NOT prepaid within 3 months after acquisition
* 1 → loan IS fully prepaid within 3 months after acquisition

This is a binary classification task.

Each row represents one loan at acquisition time.

---

## Working directory

Primary working directory:

model_training/

Unless explicitly instructed, work only inside this directory.

---

## Folder structure

model_training/
├── help_stuff/
├── train_nb/
├── trained_models/
├── training_data/
├── val_data/
└── training_log.md

### help_stuff/

Contains:

* helper materials
* glossary:
  SFLoanPerformanceDatasetGlossary.xlsx
* canonical notebook template:
  notebook_template.ipynb
* official scoring script:
  validation_score.py

When building a new model experiment notebook, use
help_stuff/notebook_template.ipynb as the starting reference for structure,
section order, required checks, scoring, artifact handling, and training-log
entry format.

The scoring script is the official validation method and must be used for model evaluation.

---

### training_data/

Contains:

* train_df_factor.csv
* train_df_target.csv

Meaning:

* factor = features
* target = target variable

---

### val_data/

Contains:

* val_df_factor.csv
* val_df_target.csv

Used only for validation.

Do not train on validation target.

---

### train_nb/

Contains experiment notebooks.

Rule:

one notebook = one model experiment

Naming:

m_001.ipynb
m_002.ipynb
m_003.ipynb

Optional paired Python implementation file:

m_001.py
m_002.py
m_003.py

Notebook should follow the structure of:

help_stuff/notebook_template.ipynb

For this project, prefer a self-contained notebook implementation by default.
Create a paired .py file only when explicitly requested or when the notebook
would otherwise become difficult to run and review.

---

### trained_models/

Save trained models here only if:

* model size < 5 MB
* model is reproducible
* validation score has been computed

Naming:

m_001_model.pkl
m_002_model.pkl
m_003_model.pkl

Never overwrite old models.

---

### training_log.md

Global experiment record.

Every meaningful experiment must append a new entry.

Never overwrite old records.

---

## Canonical ModelID rule

Canonical naming namespace:

m_001
m_002
m_003
...

Numeric ID:

001
002
003

This same ID must be used consistently for:

* notebook
* paired Python file
* saved model
* validation prediction output
* feature importance output
* log entry

Example for ModelID 003:

train_nb/m_003.ipynb
train_nb/m_003.py
train_nb/m_003_val_pred.csv
train_nb/m_003_feature_importance.csv
trained_models/m_003_model.pkl

training_log.md:

ModelID: 003

Never reuse IDs.

Always increment sequentially.

---

## Experiment lifecycle

Meaningful experiment includes:

* new model family
* new preprocessing
* new feature engineering
* materially different hyperparameters
* threshold tuning
* calibration
* ensemble

Bugfixes do not require new ModelID.

When creating new experiment:

1. inspect highest existing ModelID
2. increment by one
3. create notebook
4. create paired Python file when useful
5. run experiment
6. compute validation score
7. save model if eligible
8. update training_log.md
9. recommend next experiment

Never overwrite old experiment notebook.

---

## Notebook required workflow

Every new experiment notebook must use:

model_training/help_stuff/notebook_template.ipynb

as the canonical reference. Preserve the template's major sections unless there
is a clear experiment-specific reason to add or remove a section.

Every notebook must:

1. define ModelID metadata
2. load train factor data
3. load train target data
4. load validation factor data
5. load validation target data
6. inspect schema
7. inspect target distribution
8. preprocess
9. engineer features
10. train model
11. predict validation
12. compute score using help_stuff/validation_score.py
13. optionally save prediction artifact
14. optionally save feature importance artifact
15. save model if < 5 MB
16. append training_log.md entry

Notebook must run top-to-bottom.

No hidden state.

Each major notebook section should include a short "Content" description followed
immediately by the code for that section, matching the template style.

---

## Required checks before training

Always check:

* train factor shape
* train target shape
* validation factor shape
* validation target shape
* row alignment
* target distribution
* missing values
* duplicate columns
* train/validation schema mismatch
* categorical columns
* numeric columns
* obvious leakage columns

---

## Validation rules

Always use:

help_stuff/validation_score.py

Never silently replace scoring logic.

Never claim improvement without measured score.

---

## Runner / machine / scope

Every experiment must record:

runner:

* codex
* me

machine:

* cpu
* gpu
* tpu

data_scope:

* sample
* partial
* full

Never record score without data_scope.

Never compare sample score against full score without saying so.

---

## Safety rules

Do NOT:

* modify source CSV files
* delete old notebooks
* overwrite old models
* overwrite old logs
* fit preprocessing on train+validation together
* use validation target in feature fitting
* silently change scoring method

---

## Completion criteria

Experiment is complete only when:

1. notebook exists
2. notebook runs top-to-bottom
3. validation predictions generated
4. validation score computed
5. model saved if eligible
6. training_log.md updated

## PR hygiene and artifact policy (added after review)

To keep PRs reviewable and mergeable:

* Notebook is required per experiment (`train_nb/m_XXX.ipynb`) and must be the orchestration entrypoint for training + validation.
* Paired Python file (`train_nb/m_XXX.py`) should contain heavy logic; notebook calls into it.
* Do **not** commit binary model artifacts (`.pkl`, `.joblib`, etc.) unless explicitly requested.
* Do **not** commit generated validation prediction CSV files unless explicitly requested.
* Validation should run in-memory using `help_stuff/validation_score.py`; persisted prediction files are optional and default is off.
* Before commit, run `git status --short` and remove accidental artifacts (e.g., `__pycache__/`, temp files).
* If experiment docs mention a notebook or artifact, confirm it actually exists (or explicitly mark as intentionally omitted).


---

## Notebook structure preference (user-specific)

The notebook template is the source of truth for experiment notebook structure.

Required for experiment notebooks:

* Use model_training/help_stuff/notebook_template.ipynb as the reference before creating or editing a model notebook.
* At the beginning of each major section, write a short "Content" description of what the section does.
* Immediately after each section description, place the section's code.
* Organize logic into clear functions allocated by section (for example: data loading, preprocessing, training, threshold tuning, reporting).
* Keep training code in the notebook itself by default.
* Do not require a paired `.py` file unless explicitly requested or clearly justified.
