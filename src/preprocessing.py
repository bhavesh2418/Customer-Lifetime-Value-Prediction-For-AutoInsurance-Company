import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import (
    TARGET_COL, TARGET_LOG_COL, CATEGORICAL_COLS,
    NUMERICAL_COLS, DROP_COLS, PROCESSED_DATA_PATH
)


def drop_unused(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns])


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            skew = df[col].skew()
            fill_val = df[col].median() if abs(skew) > 0.5 else df[col].mean()
            df[col] = df[col].fillna(fill_val)
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_encode = [c for c in CATEGORICAL_COLS if c in df.columns and c != TARGET_COL]
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Log-transform right-skewed target
    df[TARGET_LOG_COL] = np.log1p(df[TARGET_COL])

    # Interaction features
    if "Monthly Premium Auto" in df.columns and "Number of Policies" in df.columns:
        df["Premium_x_Policies"] = df["Monthly Premium Auto"] * df["Number of Policies"]

    if "Months Since Policy Inception" in df.columns and "Months Since Last Claim" in df.columns:
        df["Policy_Claim_Gap"] = df["Months Since Policy Inception"] - df["Months Since Last Claim"]

    if "Total Claim Amount" in df.columns and "Monthly Premium Auto" in df.columns:
        df["Claim_to_Premium_Ratio"] = df["Total Claim Amount"] / (df["Monthly Premium Auto"] + 1)

    return df


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def run_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_unused(df)
    df = impute_missing(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed data -> {PROCESSED_DATA_PATH}")
    return df
