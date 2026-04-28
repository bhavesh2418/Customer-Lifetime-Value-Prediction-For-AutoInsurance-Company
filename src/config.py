import os
from pathlib import Path

# ── Root paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
IMAGES_DIR = ROOT_DIR / "images"
REPORTS_DIR = ROOT_DIR / "reports"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_FILENAME = "AutoInsurance.csv"
RAW_DATA_PATH = DATA_RAW_DIR / DATASET_FILENAME
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "processed_data.csv"

# ── Target ───────────────────────────────────────────────────────────────────
TARGET_COL = "Customer Lifetime Value"
TARGET_LOG_COL = "CLV_log"

# ── Feature groups ───────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "State", "Response", "Coverage", "Education",
    "EmploymentStatus", "Gender", "Location Code",
    "Marital Status", "Policy Type", "Policy",
    "Renew Offer Type", "Sales Channel", "Vehicle Class", "Vehicle Size"
]

NUMERICAL_COLS = [
    "Income", "Monthly Premium Auto", "Months Since Last Claim",
    "Months Since Policy Inception", "Number of Open Complaints",
    "Number of Policies", "Total Claim Amount"
]

DATE_COLS = ["Effective To Date"]   # parsed to extract month number
DROP_COLS = ["Customer"]            # ID column -- no predictive value

# ── Model hyperparameters ─────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

RF_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

XGB_PARAMS = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
}

# ── Plot style ────────────────────────────────────────────────────────────────
PLOT_STYLE = "seaborn-v0_8-whitegrid"
FIG_DPI = 150
