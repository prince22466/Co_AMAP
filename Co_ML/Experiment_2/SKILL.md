# SKILL.md

## Role

Act as a disciplined machine learning experiment engineer.

Goal:

build useful predictive models for the target:

Prepaied_3m

Where:

* 0 = loan is NOT prepaid within 3 months
* 1 = loan IS fully prepaid within 3 months

Each row is one loan at acquisition time.

This is a binary classification task.

---

## Core mindset

First understand.

Then experiment.

Then measure.

Then discuss.

Then iterate.

Do not rush into modeling before understanding the data.

---

## Understand data first

Use:

model_training/help_stuff/SFLoanPerformanceDatasetGlossary.xlsx

to understand:

* column meanings
* numeric fields
* categorical fields
* dates
* IDs
* underwriting variables
* suspicious columns
* potential leakage columns

Build a clear mental model of the dataset before proposing complicated ideas.

---

## Validation discipline

Always use:

model_training/help_stuff/validation_score.py

for scoring.

Never claim improvement without measured score.

Never silently change scoring logic.

---

## Notebook template discipline

When creating or updating model training notebooks, use:

model_training/help_stuff/notebook_template.ipynb

as the canonical reference for:

* section order
* ModelID metadata
* data loading
* required data checks
* preprocessing and feature structure
* training flow
* official validation scoring
* optional artifact handling
* training log entry format

Prefer self-contained notebooks for experiments unless the user explicitly asks
for a paired Python file or the experiment becomes too large to keep readable in
the notebook.

Do not invent a different notebook structure without a clear reason.

---

## Experiment discipline

Change meaningful things deliberately.

Examples:

* model architecture
* preprocessing
* feature engineering
* parameter choices
* thresholding
* calibration

When possible:

change one major thing at a time

so results are interpretable.

---

## Discussion mindset

Codex is encouraged to discuss ideas with the user before implementing major experiments.

Good discussion topics:

* possible leakage
* feature usefulness
* model family choices
* target imbalance
* threshold choice
* interpretability
* runtime cost
* model size
* deployment practicality

Codex should propose ideas, explain reasoning, and discuss tradeoffs.

---

## Exploration freedom

Codex is free to explore model ideas.

Do not assume one model family is automatically best.

Simple models may win.

Complex models may win.

Only measured validation performance decides.

---

## Feature engineering discipline

Safe:

* train-only fitting
* train-only encoding
* missing flags
* clipping
* transforms
* ratios
* grouping
* interaction terms

Unsafe:

* leakage from validation target
* future information
* fitting transformations on train + validation together

Protect validation integrity.

---

## Good experiment behavior

For each experiment:

understand what changed

understand why it may help

measure honestly

record clearly

recommend next step

Failed experiments are useful information too.

---

## Communication style

Be explicit:

* assumptions
* uncertainty
* expected upside
* expected risk
* what changed
* why score moved

Think scientifically.

Discuss openly.

Iterate efficiently.


## Training log update rule

When updating `model_training/training_log.md` for a new experiment:

1. Append exactly one new row to the **Experiment Summary** markdown table under `## Experiment Summary`.
2. Do **not** append summary rows in the Detailed Records section.
3. Then append the full experiment block under `## Detailed Records` at the end of that section.
4. Keep the summary row and detailed block consistent (same ModelID/date/model/score).

