"""Generate the project process PDF report using fpdf2."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fpdf import FPDF
import pandas as pd

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "reports"
IMAGES_DIR = ROOT / "images"
OUTPUT_PDF = REPORTS_DIR / "CLV_AutoInsurance_Process_Report.pdf"


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, "Customer Lifetime Value Prediction -- Project Process Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")

    def h1(self, text):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 90, 150)
        self.ln(4)
        self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 90, 150)
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)
        self.set_text_color(0, 0, 0)

    def h2(self, text):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(50, 50, 50)
        self.ln(3)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        self.set_text_color(0, 0, 0)

    def h3(self, text):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(80, 80, 80)
        self.ln(2)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

    def body(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.cell(6, 6, "-", new_x="RIGHT", new_y="LAST")
        self.multi_cell(0, 6, text)

    def table_row(self, cols, widths, bold=False, fill=False):
        style = "B" if bold else ""
        self.set_font("Helvetica", style, 9)
        if fill:
            self.set_fill_color(220, 235, 255)
        for col, w in zip(cols, widths):
            self.cell(w, 7, str(col), border=1, fill=fill)
        self.ln()
        self.set_fill_color(255, 255, 255)

    def add_image_if_exists(self, filename, w=170, caption=""):
        path = IMAGES_DIR / filename
        if path.exists():
            self.ln(2)
            try:
                x = (self.w - w) / 2
                self.image(str(path), x=x, w=w)
                if caption:
                    self.set_font("Helvetica", "I", 8)
                    self.set_text_color(100, 100, 100)
                    self.cell(0, 5, caption, align="C", new_x="LMARGIN", new_y="NEXT")
                    self.set_text_color(0, 0, 0)
                self.ln(3)
            except Exception as e:
                self.body(f"[Image error: {e}]")


pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# ── Cover ──────────────────────────────────────────────────────────────────────
pdf.set_font("Helvetica", "B", 22)
pdf.set_text_color(30, 90, 150)
pdf.ln(15)
pdf.cell(0, 14, "Customer Lifetime Value", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 14, "Prediction for Auto Insurance", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 13)
pdf.set_text_color(80, 80, 80)
pdf.ln(4)
pdf.cell(0, 9, "A Complete Data Science Portfolio Project", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(6)
pdf.set_font("Helvetica", "", 10)
pdf.cell(0, 7, "Dataset: Vehicle Insurance Customer Data (Kaggle)", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "Model: Random Forest Regressor | Best R2: 0.9115", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "Author: bhavesh makwana", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_text_color(0, 0, 0)

# ── 1. Project Overview ───────────────────────────────────────────────────────
pdf.add_page()
pdf.h1("1. Project Title & Overview")
pdf.body(
    "This project predicts Customer Lifetime Value (CLV) for auto insurance customers using "
    "supervised regression. CLV quantifies the total revenue expected from a single customer "
    "over their relationship with the company.\n\n"
    "Accurate CLV prediction enables insurance companies to:\n"
    "- Target retention campaigns at high-value customers\n"
    "- Optimize premium pricing per risk segment\n"
    "- Allocate marketing spend more efficiently\n\n"
    "ML Problem Type: Regression (continuous target -- CLV in dollars)\n"
    "Dataset Size: 9,134 rows, 24 columns\n"
    "Best Model: Tuned Random Forest (R2=0.9115, RMSE=0.198 on log scale)"
)

# ── 2. Dataset Description ────────────────────────────────────────────────────
pdf.h1("2. Dataset Description")
pdf.body("Source: https://www.kaggle.com/datasets/ranja7/vehicle-insurance-customer-data\n"
         "File: AutoInsurance.csv | Size: 1.59 MB | Rows: 9,134 | Columns: 24\n"
         "Missing values: None | Duplicates: 0")
pdf.ln(2)
headers = ["Feature", "Type", "Description"]
widths = [60, 30, 90]
pdf.table_row(headers, widths, bold=True, fill=True)
rows = [
    ("Customer Lifetime Value", "Float", "TARGET -- total CLV in dollars"),
    ("Customer", "String", "ID -- dropped before modeling"),
    ("State", "Categorical", "Customer state"),
    ("Response", "Categorical", "Response to last marketing campaign"),
    ("Coverage", "Categorical", "Insurance coverage level"),
    ("Education", "Categorical", "Customer education level"),
    ("Effective To Date", "Date", "Policy effective date (month extracted)"),
    ("EmploymentStatus", "Categorical", "Employment status"),
    ("Gender", "Categorical", "Customer gender"),
    ("Income", "Numerical", "Annual income ($)"),
    ("Location Code", "Categorical", "Urban / Suburban / Rural"),
    ("Marital Status", "Categorical", "Marital status"),
    ("Monthly Premium Auto", "Numerical", "Monthly auto premium paid ($)"),
    ("Months Since Last Claim", "Numerical", "Recency of last claim"),
    ("Months Since Policy Inception", "Numerical", "Policy age in months"),
    ("Number of Open Complaints", "Numerical", "Active complaint count"),
    ("Number of Policies", "Numerical", "Number of policies held"),
    ("Policy Type / Policy", "Categorical", "Policy type and sub-type"),
    ("Renew Offer Type", "Categorical", "Type of renewal offer"),
    ("Sales Channel", "Categorical", "How policy was sold"),
    ("Total Claim Amount", "Numerical", "Total historical claims ($)"),
    ("Vehicle Class / Size", "Categorical", "Vehicle classification and size"),
]
for i, r in enumerate(rows):
    pdf.table_row(r, widths, fill=(i % 2 == 0))

# ── 3. Workflow Diagram ───────────────────────────────────────────────────────
pdf.add_page()
pdf.h1("3. Workflow Diagram")
pdf.set_font("Courier", "", 10)
workflow = """
  Raw CSV (AutoInsurance.csv, 9134 rows)
           |
           v
  [Phase 0] Project Setup
  - Folder structure, src modules, .gitignore, GitHub repo
           |
           v
  [Phase 1] Download Dataset
  - Kaggle API -> data/raw/AutoInsurance.csv
           |
           v
  [01_Data_Preparation.ipynb]  EDA
  - Shape, dtypes, missing values, duplicates
  - Target distribution (skewness: 3.99 raw -> 0.05 log)
  - Numerical distributions, categorical value counts
  - Correlation heatmap, CLV by Coverage / Vehicle Class
           |
           v
  [02_Feature_Engineering.ipynb]  Feature Engineering
  - Parse Effective To Date -> Effective_Month
  - Outlier analysis (IQR) -- kept all (legitimate CLV extremes)
  - Log-transform target: CLV_log = log1p(CLV)
  - 4 interaction features engineered
  - One-hot encode 14 categorical columns
  - Shape after encoding: 9134 rows x 80+ features
           |
           v
  [03_Feature_Selection.ipynb]  Feature Selection
  - LASSO (LassoCV) -- non-zero coefficient features
  - RFE (GradientBoosting) -- top 30 by recursive elimination
  - Consensus = LASSO union RFE + engineered features
           |
           v
  [04_Model_Training.ipynb]  Model Training
  - 11 models trained and compared
  - Best: Random Forest (R2=0.9106)
  - Results saved to reports/model_results.csv
           |
           v
  [05_Model_Evaluation.ipynb]  Model Evaluation
  - GridSearchCV tuning: R2 improved to 0.9115
  - SHAP explainability: Monthly Premium is top predictor
  - Residual analysis: mean~0, normally distributed
  - Model saved: models/best_random_forest.pkl
           |
           v
  [main.py]  Full Pipeline Runner
           |
           v
  reports/  +  images/  (committed to GitHub)
"""
for line in workflow.strip().split("\n"):
    pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")

# ── 4. Phase-by-phase ─────────────────────────────────────────────────────────
pdf.add_page()
pdf.h1("4. Phase-by-Phase Explanation")

pdf.h2("Phase 0: Project Setup")
pdf.body(
    "Created a production-quality folder structure with separate directories for raw data, "
    "processed data, notebooks, source modules, models, images, reports, and scripts.\n\n"
    "Core source modules written:\n"
    "- src/config.py: all paths and hyperparameters in one place\n"
    "- src/data_loader.py: load and validate raw CSV\n"
    "- src/preprocessing.py: imputation, date parsing, feature engineering, encoding\n"
    "- src/model.py: train, evaluate, tune, and save all models\n"
    "- src/visualize.py: all plot functions, saving to images/\n\n"
    "GitHub repo created via Python urllib (avoids curl JSON issues on Windows).\n"
    "Verified: .env is gitignored before first push."
)

pdf.h2("Phase 1 (EDA): 01_Data_Preparation.ipynb")
pdf.body(
    "Key findings:\n"
    "- Dataset: 9,134 rows, 24 columns, no missing values, no duplicates\n"
    "- Target (CLV) range: $1,898 to $83,325 (mean $8,005)\n"
    "- CLV is heavily right-skewed (skewness=3.99) -- log transform required\n"
    "- After log transform: skewness drops to 0.05 (near-normal)\n"
    "- Monthly Premium Auto has the highest positive correlation with CLV\n"
    "- Extended Coverage customers have the highest median CLV\n"
    "- Luxury Car and Sports Car vehicle classes show highest CLV\n"
    "- Employed customers show higher CLV than unemployed"
)
pdf.add_image_if_exists("target_distribution.png", w=170, caption="Figure 1: CLV raw vs log-transformed distribution")
pdf.add_image_if_exists("correlation_heatmap.png", w=170, caption="Figure 2: Feature correlation heatmap")

pdf.add_page()
pdf.h2("Phase 2 (Feature Engineering): 02_Feature_Engineering.ipynb")
pdf.body(
    "Decisions made:\n"
    "- Effective To Date: extracted month number (seasonality signal) and dropped raw date\n"
    "- Outliers: IQR analysis run on all numerical features. Decision: KEEP all outliers.\n"
    "  Reason: insurance CLV legitimately contains high-value customers; capping would\n"
    "  destroy the signal we are trying to predict.\n"
    "- Log-transform applied to target (CLV_log = log1p(CLV))\n"
    "- 4 interaction features created (see Feature Engineering table)\n"
    "- 14 categorical columns one-hot encoded (drop_first=True to avoid multicollinearity)\n"
    "- Final shape: 9,134 rows x 80+ features"
)
pdf.add_image_if_exists("engineered_features_vs_clv.png", w=170, caption="Figure 3: Engineered features vs log(CLV)")

pdf.add_page()
pdf.h2("Phase 3 (Feature Selection): 03_Feature_Selection.ipynb")
pdf.body(
    "Why feature selection: with 80+ features after one-hot encoding, multicollinearity "
    "and noise features degrade generalization and slow training.\n\n"
    "LASSO (LassoCV): uses L1 regularization to shrink irrelevant feature coefficients to zero.\n"
    "RFE (GradientBoostingRegressor): recursively removes weakest features, keeping top 30.\n"
    "Consensus: union of both sets + all engineered features.\n\n"
    "Result: consensus set retains the most predictive features while removing noise."
)
pdf.add_image_if_exists("lasso_feature_importance.png", w=160, caption="Figure 4: LASSO feature importances")
pdf.add_image_if_exists("feature_selection_comparison.png", w=130, caption="Figure 5: R2 before vs after feature selection")

pdf.add_page()
pdf.h2("Phase 4 (Model Training): 04_Model_Training.ipynb")
pdf.body(
    "11 regression models trained on log-transformed CLV with 5-fold cross-validation.\n"
    "Train/Test split: 80%/20% stratified by random seed 42.\n"
    "All features standardized with StandardScaler before model fitting."
)
pdf.add_image_if_exists("model_comparison_r2.png", w=170, caption="Figure 6: Model comparison by R2 and RMSE")

pdf.add_page()
pdf.h2("Phase 5 (Model Evaluation): 05_Model_Evaluation.ipynb")
pdf.body(
    "GridSearchCV tuning on Random Forest:\n"
    "- Grid: n_estimators [100,200], max_depth [None,10,20], "
    "min_samples_split [2,5], min_samples_leaf [1,2]\n"
    "- Best CV R2 improved from 0.9093 to 0.9115\n\n"
    "SHAP explainability (TreeExplainer on 200-sample subset):\n"
    "- Monthly Premium Auto: dominant predictor\n"
    "- Number of Policies: second strongest\n"
    "- Total Claim Amount: third\n"
    "- Premium_x_Policies (engineered): strong additive signal\n\n"
    "Residuals: mean ~0, normally distributed -- model assumptions satisfied."
)
pdf.add_image_if_exists("actual_vs_predicted.png", w=130, caption="Figure 7: Actual vs Predicted CLV (log scale)")
pdf.add_image_if_exists("residuals.png", w=170, caption="Figure 8: Residual plot and distribution")
pdf.add_image_if_exists("shap_summary.png", w=150, caption="Figure 9: SHAP feature importance summary")

# ── 5. Feature Engineering Summary ───────────────────────────────────────────
pdf.add_page()
pdf.h1("5. Feature Engineering Summary")
headers = ["Feature", "Formula", "Rationale"]
widths = [55, 65, 70]
pdf.table_row(headers, widths, bold=True, fill=True)
fe_rows = [
    ("CLV_log", "log1p(CLV)", "Normalize right-skewed target for linear model fit"),
    ("Effective_Month", "month(Effective To Date)", "Seasonality signal for renewal behavior"),
    ("Premium_x_Policies", "Monthly Premium x Num Policies", "Total monthly premium commitment"),
    ("Policy_Claim_Gap", "Months Inception - Months Last Claim", "Loyal customer who hasn't claimed recently"),
    ("Claim_to_Premium_Ratio", "Total Claims / (Premium + 1)", "Claims efficiency per premium dollar"),
    ("Income_per_Policy", "Income / (Num Policies + 1)", "Affordability normalized by portfolio size"),
]
for i, r in enumerate(fe_rows):
    pdf.table_row(r, widths, fill=(i % 2 == 0))

# ── 6. Feature Selection Summary ─────────────────────────────────────────────
pdf.ln(6)
pdf.h1("6. Feature Selection Summary")
headers = ["Method", "Criterion", "Features Selected"]
widths = [55, 80, 55]
pdf.table_row(headers, widths, bold=True, fill=True)
fs_rows = [
    ("LASSO (LassoCV)", "Non-zero L1 regularized coefficient", "Variable"),
    ("RFE (GradientBoosting)", "Recursive elimination, top 30 kept", "30"),
    ("Consensus (Union)", "LASSO union RFE + engineered features", "Consensus set"),
]
for i, r in enumerate(fs_rows):
    pdf.table_row(r, widths, fill=(i % 2 == 0))

# ── 7. Model Results ─────────────────────────────────────────────────────────
pdf.add_page()
pdf.h1("7. Model Results -- Full Comparison Table")

try:
    results_df = pd.read_csv(ROOT / "reports" / "model_results.csv")
    headers = ["Model", "R2", "RMSE", "MAE", "CV R2 Mean", "CV R2 Std"]
    widths = [50, 22, 22, 22, 28, 26]
    pdf.table_row(headers, widths, bold=True, fill=True)
    for i, row in results_df.iterrows():
        pdf.table_row([
            str(row.get("Model", row.iloc[0])),
            str(row.get("R2", "")),
            str(row.get("RMSE", "")),
            str(row.get("MAE", "")),
            str(row.get("CV_R2_Mean", "")),
            str(row.get("CV_R2_Std", "")),
        ], widths, fill=(i % 2 == 0))
except Exception as e:
    pdf.body(f"Error loading results: {e}")

pdf.ln(5)
pdf.h2("Final Tuned Model -- Random Forest (GridSearchCV)")
pdf.body(
    "R2:   0.9115   (91.15% of CLV variance explained)\n"
    "RMSE: 0.1980   (on log scale -- very low prediction error)\n"
    "MAE:  0.0913   (average absolute error in log-CLV units)\n\n"
    "Comparison with reference repo (adiag321):\n"
    "- Reference best R2: 0.91 (Random Forest, untuned)\n"
    "- Our best R2: 0.9115 (Random Forest, GridSearchCV tuned)\n"
    "- Additional improvements: XGBoost comparison, SHAP explainability,\n"
    "  RFE+LASSO feature selection, engineered interaction features"
)
pdf.add_image_if_exists("feature_importance.png", w=150, caption="Figure 10: Top 25 feature importances")

# ── 8. Key Insights ───────────────────────────────────────────────────────────
pdf.add_page()
pdf.h1("8. Key Insights")
insights = [
    "Monthly Premium Auto is the single strongest predictor of CLV -- customers paying higher premiums generate more lifetime value.",
    "Number of Policies is the second strongest predictor -- multi-policy customers have significantly higher CLV.",
    "The engineered feature Premium_x_Policies (interaction term) amplifies both signals and ranks in top 5 SHAP importance.",
    "CLV is heavily right-skewed (skewness=3.99) -- log transformation is essential for all linear models and improves tree-based models too.",
    "Outliers were intentionally kept: insurance CLV legitimately contains extreme high-value customers; capping destroys predictive signal.",
    "Extended Coverage customers show ~2x higher median CLV than Basic coverage -- coverage type is a strong segmentation variable.",
    "Luxury Car and Sports Car vehicle classes correlate with highest CLV -- likely due to higher premiums and more policies.",
    "Linear models (R2~0.34) fail completely on this dataset -- the CLV-feature relationship is highly non-linear.",
    "Random Forest achieves R2=0.9115 with near-zero mean residuals -- model assumptions are well satisfied.",
    "SHAP analysis confirms the model is learning economically meaningful patterns, not spurious correlations.",
]
for i, insight in enumerate(insights, 1):
    pdf.bullet(f"{i}. {insight}\n")

# ── 9. Business Recommendations ───────────────────────────────────────────────
pdf.add_page()
pdf.h1("9. Business Recommendations")
recs = [
    ("Premium-Based Segmentation",
     "Segment customers into High/Medium/Low CLV tiers using predicted CLV. "
     "Allocate retention budgets proportionally -- high-CLV customers justify more expensive retention offers."),
    ("Multi-Policy Upselling",
     "Number of Policies is the second strongest CLV predictor. Proactively offer cross-sell campaigns "
     "(home + auto bundles) to single-policy customers to increase their CLV."),
    ("Coverage Upgrade Campaigns",
     "Extended Coverage customers have ~2x higher CLV than Basic. Design upgrade incentives "
     "for Basic coverage customers who have the income and policy history to support Extended."),
    ("Luxury Vehicle Loyalty Programs",
     "Luxury Car and Sports Car customers show highest CLV. Create exclusive loyalty programs "
     "for this segment to improve retention and reduce churn."),
    ("Claims-Based Risk Scoring",
     "Claim_to_Premium_Ratio (engineered feature) distinguishes efficient vs. costly customers. "
     "Use this ratio in underwriting to price renewals more accurately."),
    ("Seasonal Campaign Timing",
     "Effective_Month (from policy start date) shows seasonal patterns. Schedule renewal "
     "campaigns 30-60 days before peak effective months to maximize conversion."),
]
for title, body in recs:
    pdf.h3(f"- {title}")
    pdf.body(body)
    pdf.ln(1)

# ── 10. File Index ────────────────────────────────────────────────────────────
pdf.add_page()
pdf.h1("10. File Index")
headers = ["File / Folder", "Purpose"]
widths = [90, 100]
pdf.table_row(headers, widths, bold=True, fill=True)
file_index = [
    ("data/raw/AutoInsurance.csv", "Original Kaggle dataset (gitignored)"),
    ("data/processed/processed_data.csv", "Cleaned, encoded, engineered data (gitignored)"),
    ("data/processed/consensus_features.json", "LASSO+RFE consensus feature list"),
    ("notebooks/01_Data_Preparation.ipynb", "EDA -- distributions, correlations, key findings"),
    ("notebooks/02_Feature_Engineering.ipynb", "Feature engineering, encoding, outlier analysis"),
    ("notebooks/03_Feature_Selection.ipynb", "LASSO + RFE feature selection, comparison"),
    ("notebooks/04_Model_Training.ipynb", "Train 11 models, compare R2/RMSE/MAE/CV"),
    ("notebooks/05_Model_Evaluation.ipynb", "GridSearchCV tuning, SHAP, residuals"),
    ("src/config.py", "All paths, constants, and hyperparameters"),
    ("src/data_loader.py", "Load and validate raw CSV data"),
    ("src/preprocessing.py", "Imputation, encoding, date parsing, feature engineering"),
    ("src/model.py", "Train, evaluate, tune, save/load all models"),
    ("src/visualize.py", "All plot functions -- save to images/"),
    ("scripts/download_data.py", "Kaggle API dataset download"),
    ("scripts/build_notebooks.py", "Programmatically generate all .ipynb files"),
    ("scripts/generate_pdf.py", "Generate this PDF process report"),
    ("models/best_random_forest.pkl", "Tuned Random Forest model (gitignored)"),
    ("models/scaler.pkl", "StandardScaler fitted on training data (gitignored)"),
    ("images/*.png", "All 17 plots -- committed for README display"),
    ("reports/model_results.csv", "Full model comparison table with all metrics"),
    ("reports/final_model_metrics.csv", "Tuned model final metrics"),
    ("reports/CLV_AutoInsurance_Process_Report.pdf", "This document"),
    ("main.py", "Full end-to-end pipeline runner"),
    ("requirements.txt", "Pinned Python dependencies"),
    (".gitignore", "Excludes .env, data, models from git"),
    ("README.md", "GitHub README with badges, results, inline images"),
]
for i, r in enumerate(file_index):
    pdf.table_row(r, widths, fill=(i % 2 == 0))

pdf.output(str(OUTPUT_PDF))
print(f"PDF generated -> {OUTPUT_PDF}")
print(f"Size: {OUTPUT_PDF.stat().st_size:,} bytes")
