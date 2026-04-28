import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor
)
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.config import MODELS_DIR, RANDOM_STATE, CV_FOLDS, RF_PARAMS, XGB_PARAMS


def get_base_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE),
        "ElasticNet": ElasticNet(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostRegressor(random_state=RANDOM_STATE),
        "SVR": SVR(),
        "XGBoost": XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
    }


def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="r2")
    return {
        "R2": round(r2_score(y_test, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        "CV_R2_Mean": round(cv_scores.mean(), 4),
        "CV_R2_Std": round(cv_scores.std(), 4),
    }


def train_all_models(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    models = get_base_models()
    results = []
    for name, model in models.items():
        print(f"  Training {name}...")
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        metrics["Model"] = name
        results.append(metrics)
    df_results = pd.DataFrame(results).set_index("Model").sort_values("R2", ascending=False)
    return df_results


def tune_best_model(X_train, y_train, model_name: str = "Random Forest"):
    if model_name == "Random Forest":
        base = RandomForestRegressor(random_state=RANDOM_STATE)
        params = RF_PARAMS
    else:
        base = XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
        params = XGB_PARAMS

    grid = GridSearchCV(base, params, cv=CV_FOLDS, scoring="r2", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV R2: {grid.best_score_:.4f}")
    return grid.best_estimator_


def save_model(model, filename: str) -> Path:
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    print(f"Model saved -> {path}")
    return path


def load_model(filename: str):
    path = MODELS_DIR / filename
    return joblib.load(path)
