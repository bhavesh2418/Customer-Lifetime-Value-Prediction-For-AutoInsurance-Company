import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_PATH, TARGET_COL


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert TARGET_COL in df.columns, f"Target column '{TARGET_COL}' not found."
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} cols from {path.name}")
    return df


def validate_data(df: pd.DataFrame) -> None:
    print("\n-- Shape:", df.shape)
    print("-- Dtypes:\n", df.dtypes.value_counts())
    missing = df.isnull().sum()
    if missing.any():
        print("-- Missing values:\n", missing[missing > 0])
    else:
        print("-- No missing values")
    dups = df.duplicated().sum()
    print(f"-- Duplicate rows: {dups}")
