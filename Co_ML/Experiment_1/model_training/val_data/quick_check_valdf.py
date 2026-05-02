import pandas as pd
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "val_df_target.csv"


df = pd.read_csv(INPUT_FILE)

print("total count of 0 and 1: ",df.shape[0])
print("count of 1: ",df['Prepaied_3m'].sum())