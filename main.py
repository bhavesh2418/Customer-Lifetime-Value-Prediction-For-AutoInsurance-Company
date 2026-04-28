"""
Full pipeline runner for Customer Lifetime Value Prediction.
Run:  python main.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH, REPORTS_DIR, TARGET_LOG_COL,
    NUMERICAL_COLS, RANDOM_STATE, TEST_SIZE
)
from src.data_loader import load_raw_data, validate_data
from src.preprocessing import run_preprocessing, scale_features
from src.model import train_all_models, tune_best_model, save_model
from src.visualize import (
    plot_missing_values, plot_target_distribution,
    plot_numerical_distributions, plot_correlation_heatmap,
    plot_model_comparison, plot_feature_importance,
    plot_actual_vs_predicted, plot_residuals
)


def main():
    print("=" * 60)
    print("CUSTOMER LIFETIME VALUE PREDICTION PIPELINE")
    print("=" * 60)

    # 1. Load & validate
    print("\n[1] Loading data...")
    df_raw = load_raw_data(RAW_DATA_PATH)
    validate_data(df_raw)

    # 2. EDA plots on raw data
    print("\n[2] Generating EDA plots...")
    plot_missing_values(df_raw)
    plot_target_distribution(df_raw, "Customer Lifetime Value")
    num_cols_present = [c for c in NUMERICAL_COLS if c in df_raw.columns]
    plot_numerical_distributions(df_raw, num_cols_present)
    plot_correlation_heatmap(df_raw, num_cols_present + ["Customer Lifetime Value"])

    # 3. Preprocess
    print("\n[3] Preprocessing...")
    df = run_preprocessing(df_raw.copy())

    # 4. Feature / target split
    print("\n[4] Splitting features and target...")
    feature_cols = [c for c in df.columns if c not in ["Customer Lifetime Value", TARGET_LOG_COL]]
    X = df[feature_cols]
    y = df[TARGET_LOG_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train_sc, X_test_sc, _ = scale_features(X_train, X_test)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    # 5. Train all models
    print("\n[5] Training all models...")
    results_df = train_all_models(X_train_sc, X_test_sc, y_train, y_test)
    print("\nModel Results:")
    print(results_df.to_string())
    results_path = REPORTS_DIR / "model_results.csv"
    results_df.to_csv(results_path)
    print(f"\nSaved results -> {results_path}")

    # 6. Plots
    plot_model_comparison(results_df)

    # 7. Tune best model (Random Forest)
    print("\n[6] Tuning best model (Random Forest)...")
    best_model = tune_best_model(X_train_sc, y_train, model_name="Random Forest")
    y_pred = best_model.predict(X_test_sc)

    plot_feature_importance(best_model, list(X.columns))
    plot_actual_vs_predicted(y_test, y_pred)
    plot_residuals(y_test, y_pred)

    # 8. Save model
    save_model(best_model, "best_random_forest.pkl")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
