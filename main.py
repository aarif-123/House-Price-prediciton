import logging
import os
from pathlib import Path
from typing import List
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram
import time
import math

# Define custom Prometheus metrics for model performance
PREDICTION_COUNTER = Counter(
    "model_predictions_total", 
    "Total number of house price predictions made",
    ["model_version"]
)
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds", 
    "Time taken for model inference",
    ["model_version"]
)
PREDICTION_VALUE = Histogram(
    "model_prediction_value_usd", 
    "Distribution of predicted house prices",
    ["model_version"],
    buckets=[100000, 150000, 200000, 250000, 300000, 350000, 400000, 500000, 750000, 1000000]
)

FEEDBACK_COUNTER = Counter(
    "model_feedback_total",
    "Total number of ground-truth feedback events received",
    ["model_version", "status"],
)
PREDICTION_ABS_ERROR = Histogram(
    "model_prediction_abs_error_usd",
    "Absolute error between prediction and actual (USD)",
    ["model_version"],
    buckets=[5000, 10000, 25000, 50000, 75000, 100000, 150000, 200000, 300000],
)
PREDICTION_SQ_ERROR = Histogram(
    "model_prediction_sq_error_usd2",
    "Squared error between prediction and actual (USD^2)",
    ["model_version"],
    buckets=[
        25_000_000,
        100_000_000,
        625_000_000,
        2_500_000_000,
        5_625_000_000,
        10_000_000_000,
        22_500_000_000,
        40_000_000_000,
        90_000_000_000,
    ],
)
PREDICTION_APE = Histogram(
    "model_prediction_abs_percentage_error",
    "Absolute percentage error between prediction and actual",
    ["model_version"],
    buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
)
PENDING_FEEDBACK = Gauge(
    "model_predictions_pending_feedback",
    "Number of predictions awaiting ground-truth feedback",
    ["model_version"],
)

FEATURE_GRLIVAREA = Histogram(
    "feature_grlivarea",
    "Observed GrLivArea values in inference requests",
    buckets=[500, 1000, 1500, 2000, 2500, 3000, 4000, 5000],
)
FEATURE_TOTALBSMTSF = Histogram(
    "feature_totalbsmtsf",
    "Observed TotalBsmtSF values in inference requests",
    buckets=[0, 250, 500, 750, 1000, 1500, 2000, 3000, 4000],
)
FEATURE_YEARBUILT = Histogram(
    "feature_yearbuilt",
    "Observed YearBuilt values in inference requests",
    buckets=[1900, 1920, 1940, 1960, 1980, 1990, 2000, 2010, 2020, 2030],
)

# 1. Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_app")

# 2. Define Data Models (Inputs/Outputs)
class HouseData(BaseModel):
    OverallQual: int
    GrLivArea: int
    GarageCars: int
    TotalBsmtSF: int
    FullBath: int
    YearBuilt: int
    Neighborhood: str
    MSZoning: str

class PredictionRequest(BaseModel):
    rows: List[HouseData]

class PredictionResponse(BaseModel):
    predictions: List[float]
    prediction_ids: List[str]
    model_version: str

class FeedbackRow(BaseModel):
    prediction_id: str
    actual_price: float

class FeedbackRequest(BaseModel):
    rows: List[FeedbackRow]

class FeedbackResponse(BaseModel):
    received: int
    matched: int
    model_version: str

class DriftRow(HouseData):
    actual_price: float | None = None

class DriftRequest(BaseModel):
    rows: List[DriftRow]

class DriftResponse(BaseModel):
    model_version: str
    baseline_metrics: dict
    drift: dict
    batch_metrics: dict

# 3. Initialize FastAPI
app = FastAPI(title="House Price Predictor")

# Setup Prometheus Instrumentator
Instrumentator().instrument(app).expose(app)

# 4. Load Model and Metadata
# Use absolute paths to prevent working-directory errors
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"
META_PATH = BASE_DIR / "artifacts" / "metadata.json"

if not MODEL_PATH.exists():
    logger.error(f"Model file not found at {MODEL_PATH}! Please run training first.")
    model = None
    metadata = {}
else:
    model = joblib.load(MODEL_PATH)
    import json
    metadata = json.loads(META_PATH.read_text())

_PREDICTION_STORE_TTL_SECONDS = int(os.getenv("PREDICTION_STORE_TTL_SECONDS", "86400"))
_prediction_store: dict[str, dict] = {}

def _gc_prediction_store(now: float) -> None:
    if not _prediction_store:
        return
    cutoff = now - _PREDICTION_STORE_TTL_SECONDS
    stale_ids = [pid for pid, rec in _prediction_store.items() if rec.get("ts", 0) < cutoff]
    for pid in stale_ids:
        _prediction_store.pop(pid, None)

from fastapi.staticfiles import StaticFiles

def _psi(expected: list[float], actual: list[float], eps: float = 1e-6) -> float:
    """
    Population Stability Index (PSI).
    expected/actual are proportions per bin (must align by bin index).
    """
    if not expected or not actual or len(expected) != len(actual):
        return float("nan")
    psi = 0.0
    for e, a in zip(expected, actual):
        e = max(float(e), eps)
        a = max(float(a), eps)
        psi += (a - e) * math.log(a / e)
    return float(psi)


def _numeric_actual_proportions(values: pd.Series, edges: list[float]) -> list[float]:
    values = pd.to_numeric(values, errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty or not edges or len(edges) < 2:
        return []
    counts, _ = np.histogram(values.to_numpy(), bins=np.array(edges, dtype=float))
    total = int(counts.sum())
    return (counts / total).tolist() if total > 0 else [0.0 for _ in counts]


def _categorical_actual_proportions(values: pd.Series, expected_keys: set[str]) -> dict[str, float]:
    values = values.astype("string").fillna("__MISSING__")
    total = int(len(values))
    if total == 0:
        return {k: 0.0 for k in expected_keys}

    vc = values.value_counts(dropna=False)
    out: dict[str, float] = {k: 0.0 for k in expected_keys}
    other = 0
    for k, v in vc.to_dict().items():
        if k in expected_keys:
            out[k] = float(v) / total
        else:
            other += int(v)

    if "__OTHER__" in expected_keys:
        out["__OTHER__"] = float(other) / total
    return out

# 5. API Endpoints
app.mount("/static", StaticFiles(directory=BASE_DIR, html=True), name="static")

@app.get("/")
def home():
    """Serves the main webpage."""
    return FileResponse(BASE_DIR / "index.html")

@app.get("/health")
def health():
    """Checks if the system is running."""
    return {"status": "ok", "model_version": metadata.get("model_version", "1.0")}

@app.get("/model/info")
def model_info():
    """Returns non-sensitive model metadata helpful for monitoring & drift checks."""
    return {
        "model_version": metadata.get("model_version", "unknown"),
        "feature_columns": metadata.get("feature_columns", []),
        "numeric_features": metadata.get("numeric_features", []),
        "categorical_features": metadata.get("categorical_features", []),
        "baseline_metrics": metadata.get("baseline_metrics", {}),
        "has_reference_distributions": bool(metadata.get("reference_distributions")),
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Takes house details and returns estimated prices."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    model_ver = metadata.get("model_version", "2.0")
    
    try:
        # Measure inference latency
        start_time = time.perf_counter()
        
        # Convert input to a DataFrame (what the model expects)
        input_data = pd.DataFrame([row.dict() for row in request.rows])

        # Lightweight drift signals (no high-cardinality labels)
        for r in request.rows:
            FEATURE_GRLIVAREA.observe(float(r.GrLivArea))
            FEATURE_TOTALBSMTSF.observe(float(r.TotalBsmtSF))
            FEATURE_YEARBUILT.observe(float(r.YearBuilt))
        
        # Ensure we have all columns in the correct order
        input_data = input_data[metadata["feature_columns"]]
        
        # Run prediction
        raw_preds = model.predict(input_data)
        
        # Reverse Log transform if the model was trained on log scale
        if metadata.get("target_transform") == "log1p":
            final_preds = np.expm1(raw_preds)
        else:
            final_preds = raw_preds
        
        # End latency measurement
        latency = time.perf_counter() - start_time

        now = time.time()
        _gc_prediction_store(now)

        prediction_ids: List[str] = []
        for pred in final_preds:
            pid = str(uuid4())
            prediction_ids.append(pid)
            _prediction_store[pid] = {"ts": now, "pred": float(pred), "model_version": model_ver}

        # Record metrics
        PREDICTION_COUNTER.labels(model_version=model_ver).inc(len(final_preds))
        PREDICTION_LATENCY.labels(model_version=model_ver).observe(latency)
        for p in final_preds:
            PREDICTION_VALUE.labels(model_version=model_ver).observe(float(p))
        PENDING_FEEDBACK.labels(model_version=model_ver).set(
            sum(1 for rec in _prediction_store.values() if rec.get("model_version") == model_ver)
        )
            
        # Return results (rounded to 0 decimals for simplicity)
        return {
            "predictions": [round(float(p)) for p in final_preds],
            "prediction_ids": prediction_ids,
            "model_version": model_ver
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest):
    """Accepts ground-truth prices for previous predictions to compute online error metrics."""
    model_ver = metadata.get("model_version", "2.0")
    now = time.time()
    _gc_prediction_store(now)

    received = len(request.rows)
    matched = 0

    for row in request.rows:
        rec = _prediction_store.pop(row.prediction_id, None)
        if not rec:
            FEEDBACK_COUNTER.labels(model_version=model_ver, status="missing_prediction_id").inc()
            continue

        try:
            pred = float(rec["pred"])
            actual = float(row.actual_price)
        except Exception:
            FEEDBACK_COUNTER.labels(model_version=model_ver, status="invalid_payload").inc()
            continue

        matched += 1
        err = pred - actual
        abs_err = abs(err)
        sq_err = err * err
        ape = abs_err / actual if actual != 0 else float("inf")

        PREDICTION_ABS_ERROR.labels(model_version=model_ver).observe(abs_err)
        PREDICTION_SQ_ERROR.labels(model_version=model_ver).observe(sq_err)
        if np.isfinite(ape):
            PREDICTION_APE.labels(model_version=model_ver).observe(ape)
        FEEDBACK_COUNTER.labels(model_version=model_ver, status="ok").inc()

    PENDING_FEEDBACK.labels(model_version=model_ver).set(
        sum(1 for rec in _prediction_store.values() if rec.get("model_version") == model_ver)
    )

    return {"received": received, "matched": matched, "model_version": model_ver}


@app.post("/drift/report", response_model=DriftResponse)
def drift_report(request: DriftRequest):
    """
    Computes feature drift versus training reference and (optionally) accuracy on a labeled batch.

    Notes:
    - "Accuracy" for regression is reported as MAE/RMSE/MAPE (lower is better).
    - If you deploy on Vercel, this endpoint is stateless; drift over time should be persisted externally
      (e.g., Vercel Postgres/KV) if you want history.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    model_ver = metadata.get("model_version", "2.0")
    baseline = metadata.get("baseline_metrics", {})
    ref = metadata.get("reference_distributions") or {}

    df = pd.DataFrame([row.dict() for row in request.rows])
    drift: dict = {"numeric_psi": {}, "categorical_tvd": {}, "missing_rate": {}}

    # Missing rate per feature (good early warning for data quality drift)
    for c in metadata.get("feature_columns", []):
        if c in df.columns:
            drift["missing_rate"][c] = float(pd.isna(df[c]).mean())

    # Numeric drift (PSI)
    for c in metadata.get("numeric_features", []):
        ref_c = ((ref.get("numeric") or {}).get(c) or {})
        edges = ref_c.get("bins") or []
        expected = ref_c.get("proportions") or []
        actual = _numeric_actual_proportions(df.get(c, pd.Series(dtype=float)), edges)
        drift["numeric_psi"][c] = _psi(expected, actual) if actual else None

    # Categorical drift (Total Variation Distance vs reference proportions)
    for c in metadata.get("categorical_features", []):
        ref_c = ((ref.get("categorical") or {}).get(c) or {})
        expected_map = ref_c.get("proportions") or {}
        if not expected_map:
            drift["categorical_tvd"][c] = None
            continue
        expected_keys = set(expected_map.keys())
        actual_map = _categorical_actual_proportions(df.get(c, pd.Series(dtype="string")), expected_keys)
        tvd = 0.0
        for k in expected_keys:
            tvd += abs(float(actual_map.get(k, 0.0)) - float(expected_map.get(k, 0.0)))
        drift["categorical_tvd"][c] = float(0.5 * tvd)

    # Optional batch metrics if labels provided
    batch_metrics: dict = {"has_labels": False}
    if "actual_price" in df.columns and df["actual_price"].notna().any():
        labeled = df[df["actual_price"].notna()].copy()
        if not labeled.empty:
            batch_metrics["has_labels"] = True

            X = labeled[metadata.get("feature_columns", [])]
            y_true = labeled["actual_price"].astype(float).to_numpy()

            raw_preds = model.predict(X)
            y_pred = np.expm1(raw_preds) if metadata.get("target_transform") == "log1p" else raw_preds

            err = y_pred - y_true
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err * err)))
            mape = float(np.mean(np.abs(err) / np.maximum(y_true, 1.0)))

            batch_metrics.update(
                {
                    "count": int(len(labeled)),
                    "mae_usd": mae,
                    "rmse_usd": rmse,
                    "mape": mape,
                }
            )

            if baseline:
                # Deltas (positive means worse for these metrics)
                batch_metrics["delta_vs_baseline"] = {
                    "mae_usd": mae - float(baseline.get("mae_usd", 0.0)),
                    "rmse_usd": rmse - float(baseline.get("rmse_usd", 0.0)),
                    "mape": mape - float(baseline.get("mape", 0.0)),
                }

    return {
        "model_version": model_ver,
        "baseline_metrics": baseline,
        "drift": drift,
        "batch_metrics": batch_metrics,
    }

# Running the app:
# uvicorn main:app --reload
