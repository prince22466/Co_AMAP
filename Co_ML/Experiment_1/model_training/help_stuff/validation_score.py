"""
here is a py script to give validation score for prepayment prediction
"""

import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
y_true_FILE = SCRIPT_DIR.parent/"val_data" / "val_df_target.csv"
def prediction_score(y_pred):
    """
    given true value and pred value,
    return the percentage of sucess prediction
    """
    y_true = pd.read_csv(y_true_FILE)["Prepaied_3m"].reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    if len(y_pred) != len(y_true):
        raise ValueError(
            f"y_pred length ({len(y_pred)}) does not match y_true length ({len(y_true)})"
        )

    diff = (y_true - y_pred).abs()
    incorrect_count = diff.sum()
    total_count = len(diff)

    return 1 - incorrect_count/total_count
