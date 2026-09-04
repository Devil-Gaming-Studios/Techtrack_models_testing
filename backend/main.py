from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="EV Range Prediction API")

# Allow the React dev server (and any frontend) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's actual origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILES = {
    "Ridge Regression": "ridge_range_model.pkl",
    "Decision Tree": "decision_tree_range_model.pkl",
    "Random Forest": "random_forest_range_model.pkl",
    "Gradient Boosting": "gradient_boosting_range_model.pkl",
    "XGBoost": "xgboost_range_model.pkl",
    "LightGBM": "lightgbm_range_model.pkl",
}

CLUSTER_FILES = {
    "kmeans": "kmeans_model.pkl",
    "scaler": "scaler_for_clustering.pkl",
    "cluster_avg_range": "cluster_avg_range.pkl",
}

models = {}
cluster_assets = {}


@app.on_event("startup")
def load_all():
    for name, filename in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)

    for name, filename in CLUSTER_FILES.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            cluster_assets[name] = joblib.load(path)

    print(f"Loaded models: {list(models.keys())}")
    print(f"Loaded cluster assets: {list(cluster_assets.keys())}")


class EVSpecs(BaseModel):
    battery_capacity_kWh: float
    torque_nm: float
    top_speed_kmh: float
    acceleration_0_100_s: float
    fast_charging_power_kw_dc: float
    towing_capacity_kg: float
    efficiency_wh_per_km: float  # placeholder input; not a meaningful predictor
    length_mm: float
    width_mm: float
    height_mm: float
    seats: int
    cargo_volume_l: float
    drivetrain: str
    segment: str
    car_body_type: str


NUMERIC_COLS_FOR_CLUSTERING = [
    "top_speed_kmh", "battery_capacity_kWh", "torque_nm",
    "efficiency_wh_per_km", "acceleration_0_100_s",
    "fast_charging_power_kw_dc", "towing_capacity_kg",
    "cargo_volume_l", "seats", "length_mm", "width_mm", "height_mm",
]


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(models.keys())}


@app.post("/predict")
def predict(specs: EVSpecs):
    if not models:
        raise HTTPException(status_code=503, detail="No models loaded on the server.")

    input_df = pd.DataFrame([specs.dict()])

    predictions = {}
    for name, pipe in models.items():
        try:
            pred = float(pipe.predict(input_df)[0])
            predictions[name] = round(pred, 1)
        except Exception as e:
            predictions[name] = f"Error: {e}"

    numeric_preds = [v for v in predictions.values() if isinstance(v, (int, float))]
    avg_prediction = round(sum(numeric_preds) / len(numeric_preds), 1) if numeric_preds else None

    cluster_result = None
    if all(k in cluster_assets for k in ["kmeans", "scaler", "cluster_avg_range"]):
        try:
            numeric_input = input_df[NUMERIC_COLS_FOR_CLUSTERING]
            scaled_input = cluster_assets["scaler"].transform(numeric_input)
            cluster_label = int(cluster_assets["kmeans"].predict(scaled_input)[0])
            cluster_avg = cluster_assets["cluster_avg_range"].get(cluster_label)
            cluster_result = {
                "cluster": cluster_label,
                "average_range_km": round(cluster_avg, 1) if cluster_avg is not None else None,
            }
        except Exception as e:
            cluster_result = {"error": str(e)}

    return {
        "predictions": predictions,
        "average_prediction_km": avg_prediction,
        "cluster_estimate": cluster_result,
    }
