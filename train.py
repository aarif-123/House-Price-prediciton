import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdvancedTraining")

def _numeric_reference(x: pd.Series, bins: int = 10) -> dict:
    """Build a lightweight numeric distribution reference for drift checks."""
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    if x.empty:
        return {"bins": [], "proportions": []}

    # Quantile bins handle skewed features better than equal-width.
    edges = np.unique(np.quantile(x.to_numpy(), np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        # Fallback to min/max if there isn't enough variance.
        edges = np.unique(np.array([x.min(), x.max()], dtype=float))
        if edges.size == 1:
            edges = np.array([edges[0], edges[0] + 1.0], dtype=float)

    counts, hist_edges = np.histogram(x.to_numpy(), bins=edges)
    total = int(counts.sum())
    proportions = (counts / total).tolist() if total > 0 else [0.0 for _ in counts]
    return {"bins": hist_edges.tolist(), "proportions": proportions}


def _categorical_reference(x: pd.Series, top_k: int = 25) -> dict:
    """Build a categorical frequency reference with an __OTHER__ bucket."""
    x = x.astype("string").fillna("__MISSING__")
    vc = x.value_counts(dropna=False)
    top = vc.head(top_k)
    other = int(vc.iloc[top_k:].sum()) if len(vc) > top_k else 0
    counts = {k: int(v) for k, v in top.to_dict().items()}
    if other > 0:
        counts["__OTHER__"] = other
    total = int(vc.sum())
    proportions = {k: (v / total) for k, v in counts.items()} if total > 0 else {k: 0.0 for k in counts}
    return {"proportions": proportions}


def train_advanced_model():
    # 1. Load Data
    logger.info("Loading training data...")
    BASE_DIR = Path(__file__).resolve().parent
    df = pd.read_csv(BASE_DIR / "train.csv")

    # 2. Data Cleaning (From Model Notebook)
    # Remove large houses that are outliers
    df = df[df['GrLivArea'] < 4000]

    # 3. Feature Selection
    # We use these features as they are the most impactful (based on correlation)
    FEATURES = [
        "OverallQual", "GrLivArea", "GarageCars", 
        "TotalBsmtSF", "FullBath", "YearBuilt", 
        "Neighborhood", "MSZoning"
    ]
    TARGET = "SalePrice"

    X = df[FEATURES]
    # Transform target using Log (Makes distribution normal)
    y = np.log1p(df[TARGET])

    # 4. Building the Preprocessing Pipeline
    num_cols = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"]
    cat_cols = ["Neighborhood", "MSZoning"]

    # Numeric: Fill gaps with median
    num_transformer = SimpleImputer(strategy="median")

    # Categorical: Fill missing and encode to numbers
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])

    # 5. Advanced Model (XGBoost)
    # Using parameters optimized for this dataset
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])

    # 6. Training
    logger.info("Starting Advanced XGBoost Training...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model_pipeline.fit(X_train, y_train)

    # Baseline metrics on a held-out split (in original USD scale)
    y_val_pred_log = model_pipeline.predict(X_val)
    y_val_true = np.expm1(y_val.to_numpy())
    y_val_pred = np.expm1(y_val_pred_log)

    mae = float(mean_absolute_error(y_val_true, y_val_pred))
    rmse = float(np.sqrt(mean_squared_error(y_val_true, y_val_pred)))
    mape = float(np.mean(np.abs((y_val_true - y_val_pred) / np.maximum(y_val_true, 1.0))))

    # 7. Save Artifacts
    logger.info("Saving optimized model...")
    artifacts_dir = BASE_DIR / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save the whole pipeline
    joblib.dump(model_pipeline, artifacts_dir / "model.joblib")

    # Save metadata (Crucial: note that target is log-transformed)
    metadata = {
        "feature_columns": FEATURES,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "model_version": "2.0-XGBoost-Log",
        "target_transform": "log1p",
        "baseline_metrics": {
            "split": "train_test_split",
            "test_size": 0.2,
            "random_state": 42,
            "mae_usd": mae,
            "rmse_usd": rmse,
            "mape": mape,
        },
        "reference_distributions": {
            "numeric": {c: _numeric_reference(X[c]) for c in num_cols},
            "categorical": {c: _categorical_reference(X[c]) for c in cat_cols},
        },
        "metrics": {"status": "trained_successfully"},
    }
    with open(artifacts_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info("Success! End-to-End Model V2.0 is ready.")

if __name__ == "__main__":
    train_advanced_model()
