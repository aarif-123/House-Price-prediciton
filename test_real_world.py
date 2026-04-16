import joblib
import pandas as pd
import json
from pathlib import Path
import numpy as np

# Load the trained model and metadata
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"
META_PATH = BASE_DIR / "artifacts" / "metadata.json"

if not MODEL_PATH.exists():
    print("Model not found. Please train first.")
    exit(1)

model = joblib.load(MODEL_PATH)
with open(META_PATH, "r") as f:
    metadata = json.load(f)

# Real world scenarios (we have to map them to Ames, Iowa dataset categories)
# Features needed: OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath, YearBuilt, Neighborhood, MSZoning

real_world_houses = [
    {
        "description": "Brand New Suburban Family Home (High Quality)",
        "data": {
            "OverallQual": 8,            # 8/10 quality
            "GrLivArea": 2500,           # 2500 sqft living area
            "GarageCars": 3,             # 3 car garage
            "TotalBsmtSF": 1200,         # 1200 sqft basement
            "FullBath": 3,               # 3 full bathrooms
            "YearBuilt": 2023,           # Very recent
            "Neighborhood": "CollgCr",   # Popular suburban neighborhood
            "MSZoning": "RL"             # Residential Low Density
        }
    },
    {
        "description": "Older Historic Townhome (Fixer-Upper)",
        "data": {
            "OverallQual": 5,            # Average/Fair quality
            "GrLivArea": 1200,           # 1200 sqft
            "GarageCars": 1,             # 1 car garage
            "TotalBsmtSF": 600,          # Small basement
            "FullBath": 1,               # 1 bathroom
            "YearBuilt": 1940,           # Older home
            "Neighborhood": "OldTown",   # Historic district
            "MSZoning": "RM"             # Residential Medium Density
        }
    },
    {
        "description": "Massive Luxury Mansion",
        "data": {
            "OverallQual": 10,           # Perfect quality
            "GrLivArea": 3900,           # 3900 sqft 
            "GarageCars": 4,             # 4 car garage
            "TotalBsmtSF": 2000,         # Massive basement
            "FullBath": 4,               # 4 bathrooms
            "YearBuilt": 2018,           # Modern Luxury
            "Neighborhood": "NridgHt",   # Most expensive neighborhood
            "MSZoning": "RL"             # Residential Low Density
        }
    }
]

print("--- TESTING HOUSE PRICE PREDICTION MODEL OVER REAL WORLD SCENARIOS ---\n")

for item in real_world_houses:
    print(f"Scenario: {item['description']}")
    
    # Convert to DataFrame as expected by the model
    input_df = pd.DataFrame([item["data"]])[metadata["feature_columns"]]
    
    # Predict
    raw_pred = model.predict(input_df)
    
    if metadata.get("target_transform") == "log1p":
        final_price = np.expm1(raw_pred)[0]
    else:
        final_price = raw_pred[0]
        
    # Formatting features for display
    print(f"  - Size: {item['data']['GrLivArea']} sqft, Built: {item['data']['YearBuilt']}, Quality: {item['data']['OverallQual']}/10")
    print(f"  - Location: {item['data']['Neighborhood']} ({item['data']['MSZoning']})")
    print(f"  => PREDICTED FAIR MARKET VALUE: ${final_price:,.2f}\n")
