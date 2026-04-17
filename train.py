import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdvancedTraining")

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for House Price feature engineering."""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        # 1. House Age
        # Assume YearBuilt is present. If YrSold is missing, use modern year (e.g. 2011)
        # Note: In the official dataset, the last year of sale is 2010.
        yr_sold = X['YrSold'] if 'YrSold' in X.columns else 2010
        X['HouseAge'] = yr_sold - X['YearBuilt']
        
        # 2. Total Area Interaction
        X['TotalSF'] = X['GrLivArea'] + X['TotalBsmtSF'].fillna(0)
        
        # 3. Quality and Area Interaction (Very important for price)
        X['Qual_Area_Interact'] = X['OverallQual'] * X['GrLivArea']
        
        # 4. Bathrooms Total
        if 'FullBath' in X.columns and 'HalfBath' in X.columns:
            X['TotalBath'] = X['FullBath'] + (0.5 * X['HalfBath'])
        else:
            X['TotalBath'] = X['FullBath'] if 'FullBath' in X.columns else 1

        # Drop columns that are no longer needed if we want to be strict, 
        # but here we keep them and let the model decide or ColumnTransformer handle it.
        return X

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
        ("regressor", XGBRegressor(random_state=42))
    ])

    # 5. Model Training (Optimized for speed/completion)
    logger.info("Starting XGBoost Training (V3.0)...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # Use the pre-defined pipeline
    best_model = model_pipeline

    # Check for GPU support (Modern XGBoost 2.0+ and Legacy)
    gpu_params = {'tree_method': 'hist'} # Default to CPU
    try:
        import xgboost as xgb
        v = [int(i) for i in xgb.__version__.split('.')[:2]]
        if v[0] >= 2:
            # Modern XGBoost: tree_method='hist', device='cuda'
            try:
                test_model = xgb.XGBRegressor(tree_method='hist', device='cuda')
                test_model.fit(np.array([[0]]), np.array([0]))
                gpu_params = {'tree_method': 'hist', 'device': 'cuda'}
                logger.info("GPU (CUDA) detected! Using GPU acceleration.")
            except Exception:
                # Fallback to legacy gpu_hist just in case
                test_model = xgb.XGBRegressor(tree_method='gpu_hist')
                test_model.fit(np.array([[0]]), np.array([0]))
                gpu_params = {'tree_method': 'gpu_hist'}
                logger.info("Legacy GPU (gpu_hist) detected!")
        else:
            # Legacy XGBoost
            test_model = xgb.XGBRegressor(tree_method='gpu_hist')
            test_model.fit(np.array([[0]]), np.array([0]))
            gpu_params = {'tree_method': 'gpu_hist'}
            logger.info("GPU (gpu_hist) detected!")
    except Exception as e:
        logger.info(f"GPU not available, falling back to CPU. (Reason: {str(e)[:100]})")
        gpu_params = {'tree_method': 'hist'}

    best_model.set_params(
        regressor__n_estimators=1000,
        regressor__learning_rate=0.05,
        regressor__max_depth=5,
        regressor__subsample=0.8,
        regressor__colsample_bytree=0.8,
        **{f"regressor__{k}": v for k, v in gpu_params.items()}
    )
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
        "model_version": "3.0-XGBoost-Eng-Tuned",
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

    logger.info(f"Success! Model V3.0 is ready. MAE: ${mae:,.2f}")

if __name__ == "__main__":
    train_advanced_model()
