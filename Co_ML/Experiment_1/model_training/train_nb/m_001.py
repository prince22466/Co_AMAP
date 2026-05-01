from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MODEL_ID = "m_001"
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from help_stuff.validation_score import prediction_score


def run() -> dict:
    train_factor_path = BASE_DIR / "training_data" / "train_df_factor.csv"
    train_target_path = BASE_DIR / "training_data" / "train_df_target.csv"
    val_factor_path = BASE_DIR / "val_data" / "val_df_factor.csv"
    val_target_path = BASE_DIR / "val_data" / "val_df_target.csv"

    X_train = pd.read_csv(train_factor_path)
    y_train = pd.read_csv(train_target_path)["Prepaied_3m"]
    X_val = pd.read_csv(val_factor_path)
    y_val = pd.read_csv(val_target_path)["Prepaied_3m"]

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    val_probs = model.predict_proba(X_val)[:, 1]

    candidate_thresholds = np.linspace(0.05, 0.95, 19)
    scored_thresholds = []
    for threshold in candidate_thresholds:
        val_pred = (val_probs >= threshold).astype(int)
        score = prediction_score(val_pred)
        scored_thresholds.append((threshold, score))

    best_threshold, best_score = max(scored_thresholds, key=lambda x: x[1])
    best_pred = (val_probs >= best_threshold).astype(int)


    return {
        "model_id": MODEL_ID,
        "score": float(best_score),
        "threshold": float(best_threshold),
        "train_shape": X_train.shape,
        "val_shape": X_val.shape,
        "target_mean": float(y_train.mean()),
        "missing_train": int(X_train.isna().sum().sum()),
        "missing_val": int(X_val.isna().sum().sum()),
        "schema_match": list(X_train.columns) == list(X_val.columns),
        "duplicate_columns": int(X_train.columns.duplicated().sum()),
        "val_positive_rate": float(y_val.mean()),
    }


if __name__ == "__main__":
    result = run()
    for k, v in result.items():
        print(f"{k}: {v}")
