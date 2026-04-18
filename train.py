import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from feature_engineering import FeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdvancedTraining")

def _numeric_reference(x: pd.Series, bins: int = 10) -> dict:
    """Build a lightweight numeric distribution reference for drift checks."""
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    if x.empty:
        return {"bins": [], "proportions": []}

    edges = np.unique(np.quantile(x.to_numpy(), np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
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
    df = pd.read_csv(BASE_DIR / "train.csv", engine='python')

    # 2. Outlier Handling & Data Augmentation
    # Remove extreme outliers in Area
    df = df[df['GrLivArea'] < 4500]
    # Cap SalePrice at 99th percentile to avoid extreme target skew
    # y_cap = np.percentile(df['SalePrice'], 99.5)
    # df['SalePrice'] = df['SalePrice'].clip(upper=y_cap)

    # 3. Feature Selection
    # Base features needed for FeatureEngineer + final model features
    BASE_FEATURES = [
        "OverallQual", "GrLivArea", "GarageCars", 
        "TotalBsmtSF", "FullBath", "YearBuilt", 
        "Neighborhood", "MSZoning", "YrSold"
    ]
    if "HalfBath" in df.columns: BASE_FEATURES.append("HalfBath")
    
    TARGET = "SalePrice"

    X = df[BASE_FEATURES]
    y = np.log1p(df[TARGET])

    # 4. Building the Preprocessing Pipeline
    num_cols = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt", "HouseAge", "TotalSF", "Qual_Area_Interact"]
    cat_cols = ["Neighborhood", "MSZoning"]

    # Preprocessing for numerical data
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Preprocessing for categorical data
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ], remainder="drop")

    # Full Pipeline
    model_pipeline = Pipeline(steps=[
        ("engineer", FeatureEngineer()),
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=450,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        ))
    ])

    # 5. Model Training (Vercel-friendly runtime footprint)
    logger.info("Starting RandomForest training (V3.1)...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    best_model = model_pipeline
    best_model.fit(X_train, y_train)

    # 6. Evaluation
    y_val_pred_log = best_model.predict(X_val)
    y_val_true = np.expm1(y_val.to_numpy())
    y_val_pred = np.expm1(y_val_pred_log)

    mae = float(mean_absolute_error(y_val_true, y_val_pred))
    rmse = float(np.sqrt(mean_squared_error(y_val_true, y_val_pred)))
    mape = float(np.mean(np.abs((y_val_true - y_val_pred) / np.maximum(y_val_true, 1.0))))

    # 7. Save Artifacts
    logger.info("Saving optimized model...")
    artifacts_dir = BASE_DIR / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    joblib.dump(best_model, artifacts_dir / "model.joblib")

    # Important: FeatureEngineer adds new columns, we need to track what main.py should send
    # main.py sends the BASE_FEATURES, and the pipeline handles the rest.
    metadata = {
        "feature_columns": BASE_FEATURES,
        "numeric_features": num_cols, # Used for PSI report
        "categorical_features": cat_cols,
        "model_version": "3.1-RandomForest-Eng-Tuned",
        "target_transform": "log1p",
        "baseline_metrics": {
            "mae_usd": mae,
            "rmse_usd": rmse,
            "mape": mape,
        },
        "reference_distributions": {
            # We calculate reference on the ENGINEERED features for the drift report
            "numeric": {c: _numeric_reference(best_model.named_steps["engineer"].transform(X)[c]) for c in num_cols},
            "categorical": {c: _categorical_reference(X[c]) for c in cat_cols},
        },
        "metrics": {"status": "trained_successfully"},
    }
    with open(artifacts_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Success! Model V3.1 is ready. MAE: ${mae:,.2f}")

if __name__ == "__main__":
    train_advanced_model()
