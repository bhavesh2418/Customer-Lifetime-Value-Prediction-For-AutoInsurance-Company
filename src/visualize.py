import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import IMAGES_DIR, PLOT_STYLE, FIG_DPI

plt.style.use(PLOT_STYLE)


def _save(fig: plt.Figure, filename: str) -> None:
    path = IMAGES_DIR / filename
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def plot_missing_values(df: pd.DataFrame) -> None:
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values to plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    missing.plot(kind="bar", ax=ax, color="coral")
    ax.set_title("Missing Values per Feature")
    ax.set_ylabel("Count")
    _save(fig, "missing_values.png")


def plot_target_distribution(df: pd.DataFrame, target_col: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[target_col], kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title(f"{target_col} (raw)")
    sns.histplot(np.log1p(df[target_col]), kde=True, ax=axes[1], color="seagreen")
    axes[1].set_title(f"log1p({target_col})")
    fig.suptitle("Target Distribution", fontsize=14)
    _save(fig, "target_distribution.png")


def plot_numerical_distributions(df: pd.DataFrame, num_cols: list) -> None:
    cols = [c for c in num_cols if c in df.columns]
    n = len(cols)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="mediumpurple")
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Numerical Feature Distributions", fontsize=14)
    plt.tight_layout()
    _save(fig, "numerical_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame, num_cols: list) -> None:
    cols = [c for c in num_cols if c in df.columns]
    fig, ax = plt.subplots(figsize=(12, 8))
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap")
    _save(fig, "correlation_heatmap.png")


def plot_clv_by_category(df: pd.DataFrame, cat_col: str, target_col: str) -> None:
    if cat_col not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df.groupby(cat_col)[target_col].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x=cat_col, y=target_col, order=order, ax=ax, palette="Set2")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(f"{target_col} by {cat_col}")
    plt.tight_layout()
    _save(fig, f"clv_by_{cat_col.lower().replace(' ', '_')}.png")


def plot_model_comparison(results_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    results_df["R2"].sort_values().plot(kind="barh", ax=ax, color="teal")
    ax.set_title("Model Comparison -- R2 Score")
    ax.set_xlabel("R2")
    ax.axvline(0.9, color="red", linestyle="--", label="R2=0.9")
    ax.legend()
    _save(fig, "model_comparison_r2.png")


def plot_feature_importance(model, feature_names: list, top_n: int = 20) -> None:
    if not hasattr(model, "feature_importances_"):
        print("Model has no feature_importances_")
        return
    imp = pd.Series(model.feature_importances_, index=feature_names).nlargest(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    imp.sort_values().plot(kind="barh", ax=ax, color="darkorange")
    ax.set_title(f"Top {top_n} Feature Importances")
    _save(fig, "feature_importance.png")


def plot_actual_vs_predicted(y_test, y_pred) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, color="royalblue", edgecolors="none", s=20)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")
    ax.set_xlabel("Actual CLV (log)")
    ax.set_ylabel("Predicted CLV (log)")
    ax.set_title("Actual vs Predicted CLV")
    ax.legend()
    _save(fig, "actual_vs_predicted.png")


def plot_residuals(y_test, y_pred) -> None:
    residuals = np.array(y_test) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.4, color="slategray", s=20)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residual Plot")
    sns.histplot(residuals, kde=True, ax=axes[1], color="slategray")
    axes[1].set_title("Residual Distribution")
    plt.tight_layout()
    _save(fig, "residuals.png")


def plot_pca_variance(explained_variance_ratio: np.ndarray) -> None:
    cumulative = np.cumsum(explained_variance_ratio)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio,
           alpha=0.7, color="steelblue", label="Individual")
    ax.plot(range(1, len(cumulative) + 1), cumulative, "ro-", label="Cumulative")
    ax.axhline(0.90, color="green", linestyle="--", label="90% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Explained Variance")
    ax.legend()
    _save(fig, "pca_variance.png")
