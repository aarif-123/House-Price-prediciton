# Vercel Deployment Log (House Price Prediction)

This document records the exact deployment flow we executed, including failures, fixes, and final validation.

## Project
- Local path: `C:\Users\Mohmmed Aarif\projects\vizuara-rag\rag-chat\HousePriceprediciton`
- Platform: Vercel (Python serverless function)
- Final live URL: `https://housepriceprediction-rouge.vercel.app`

## 1. Initial Deployment Attempt
Command:
```powershell
npx vercel deploy --prod --yes --name housepriceprediction
```

Result:
- Project linked successfully.
- Build failed with bundle size error:
  - `Total bundle size (888.41 MB) exceeds Lambda ephemeral storage limit (500 MB)`

Root cause:
- Runtime dependencies and deployed files were too large (not suitable for Vercel Python Lambda limits).

## 2. Reduce Bundle Size
Changes made:
- Switched training model from XGBoost to RandomForest (lighter runtime footprint).
- Removed `xgboost` from runtime dependencies.
- Added `.vercelignore` to exclude non-runtime files:
  - notebooks, csv files, Docker files, grafana folder, etc.

Files updated:
- `train.py`
- `requirements.txt`
- `.vercelignore`

Command:
```powershell
python train.py
```

Result:
- New model artifacts generated under `artifacts/`.

## 3. Redeploy After Slimming
Command:
```powershell
npx vercel deploy --prod --yes
```

Result:
- Deployment succeeded and alias assigned:
  - `https://housepriceprediction-rouge.vercel.app`
- But runtime returned `FUNCTION_INVOCATION_FAILED` (500).

## 4. Fix Runtime Startup Robustness
Issue observed:
- Function started but model metadata/artifacts were not loaded reliably.

Fixes:
- Added safe model/metadata loading with graceful fallback in `main.py`.
- Added robust artifact path resolution (multiple candidate directories).
- Added explicit `includeFiles` in `vercel.json` for:
  - `artifacts/**`
  - `index.html`
  - `house_bg.png`

Files updated:
- `main.py`
- `vercel.json`

## 5. Fix Model Unpickling in Serverless
Issue observed:
- Model loaded inconsistently due custom transformer class serialization context.

Fix:
- Moved custom transformer to dedicated module:
  - `feature_engineering.py`
- Updated `train.py` to import from that module.
- Included module in Vercel bundle via `vercel.json`.
- Retrained model artifacts.

Files updated:
- `feature_engineering.py` (new)
- `train.py`
- `vercel.json`

Commands:
```powershell
python train.py
npx vercel deploy --prod --yes
```

## 6. Fix scikit-learn Version Mismatch
Issue observed from live `/predict` response:
- `AttributeError("'SimpleImputer' object has no attribute '_fill_dtype'")`

Root cause:
- Model was trained with a different scikit-learn version than deployed runtime.

Fix:
- Pinned scikit-learn to local training version in `requirements.txt`:
  - `scikit-learn==1.7.2`
- Retrained model and redeployed.

Files updated:
- `requirements.txt`
- `artifacts/model.joblib`
- `artifacts/metadata.json`

Commands:
```powershell
python -c "import sklearn; print(sklearn.__version__)"
python train.py
npx vercel deploy --prod --yes
```

## 7. Final Verification (Successful)
Command checks run:
```powershell
Invoke-RestMethod https://housepriceprediction-rouge.vercel.app/health
Invoke-RestMethod https://housepriceprediction-rouge.vercel.app/model/info
curl.exe -s -i -X POST "https://housepriceprediction-rouge.vercel.app/predict" -H "Content-Type: application/json" --data-binary "@payload.json"
```

Final results:
- `/health` -> `200 OK`
- `/model/info` -> model loaded with:
  - `model_version: 3.1-RandomForest-Eng-Tuned`
  - baseline metrics present
  - drift reference distributions present
- `/predict` -> `200 OK` with valid prediction response

Example successful response:
```json
{
  "predictions": [201676.0],
  "prediction_ids": ["7706773d-1f9b-44eb-8ca0-c903808f5b6a"],
  "model_version": "3.1-RandomForest-Eng-Tuned"
}
```

## Current Production URLs
- Alias URL: `https://housepriceprediction-rouge.vercel.app`
- Latest deployment URL used in final fix cycle:
  - `https://housepriceprediction-ded0149f6-aarif-123s-projects.vercel.app`
