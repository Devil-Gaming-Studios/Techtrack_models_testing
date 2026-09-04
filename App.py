import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="EV Range Predictor", layout="centered")
st.title("🔋 EV Range Prediction — Model Comparison")
st.write("Enter EV specifications below and compare predictions across all trained models.")

# ------------------------------------------------------------------
# Load all trained pipelines (each is a full sklearn Pipeline that
# already includes preprocessing, so raw input works directly)
# ------------------------------------------------------------------
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

@st.cache_resource
def load_models():
    models = {}
    for name, path in MODEL_FILES.items():
        try:
            models[name] = joblib.load(path)
        except FileNotFoundError:
            pass  # skip models whose .pkl isn't present in this folder
    return models

models = load_models()

@st.cache_resource
def load_cluster_assets():
    assets = {}
    for name, path in CLUSTER_FILES.items():
        try:
            assets[name] = joblib.load(path)
        except FileNotFoundError:
            assets[name] = None
    return assets

cluster_assets = load_cluster_assets()
has_clustering = all(v is not None for v in cluster_assets.values())

if not models:
    st.error(
        "No trained model files found. Place the .pkl files "
        "(e.g. ridge_range_model.pkl, random_forest_range_model.pkl, ...) "
        "in the same folder as this app.py."
    )
    st.stop()

st.success(f"Loaded models: {', '.join(models.keys())}")

# ------------------------------------------------------------------
# Input form — EDIT these fields/options to exactly match the
# columns your pipelines were trained on (names must match X columns)
# ------------------------------------------------------------------
st.header("Enter EV Specifications")

col1, col2 = st.columns(2)

with col1:
    battery_capacity_kWh = st.number_input("Battery Capacity (kWh)", 10.0, 200.0, 60.0)
    torque_nm = st.number_input("Torque (Nm)", 50.0, 1500.0, 300.0)
    top_speed_kmh = st.number_input("Top Speed (km/h)", 80.0, 350.0, 180.0)
    acceleration_0_100_s = st.number_input("0-100 km/h Acceleration (s)", 2.0, 20.0, 7.5)
    fast_charging_power_kw_dc = st.number_input("Fast Charging Power (kW DC)", 10.0, 350.0, 100.0)
    towing_capacity_kg = st.number_input("Towing Capacity (kg)", 0.0, 3500.0, 0.0)
    efficiency_wh_per_km = st.number_input(
        "Efficiency (Wh/km)", 50.0, 400.0, 150.0,
        help="Included only because the saved model expects this column; not used as a meaningful predictor."
    )

with col2:
    length_mm = st.number_input("Length (mm)", 2500.0, 6000.0, 4500.0)
    width_mm = st.number_input("Width (mm)", 1400.0, 2200.0, 1850.0)
    height_mm = st.number_input("Height (mm)", 1200.0, 2200.0, 1550.0)
    seats = st.number_input("Seats", 2, 9, 5)
    cargo_volume_l = st.number_input("Cargo Volume (L)", 0.0, 3000.0, 450.0)
    drivetrain = st.selectbox("Drivetrain", ["FWD", "RWD", "AWD"])

segment = st.selectbox(
    "Segment",
    ["A - Mini", "B - Compact", "C - Mid-size", "D - Large", "E - Premium",
     "F - Luxury", "SUV", "Crossover", "Pickup", "Van", "Sports", "Other"]
)
car_body_type = st.selectbox(
    "Body Type",
    ["Hatchback", "Sedan", "SUV", "Crossover", "Coupe", "Pickup", "Van", "Wagon"]
)

# Build a single-row DataFrame matching training feature columns exactly.
# IMPORTANT: column names below must match the X used when the pipelines
# were fit (edit/add/remove as needed for your actual feature set).
input_df = pd.DataFrame([{
    "battery_capacity_kWh": battery_capacity_kWh,
    "torque_nm": torque_nm,
    "top_speed_kmh": top_speed_kmh,
    "acceleration_0_100_s": acceleration_0_100_s,
    "fast_charging_power_kw_dc": fast_charging_power_kw_dc,
    "towing_capacity_kg": towing_capacity_kg,
    "efficiency_wh_per_km": efficiency_wh_per_km,
    "length_mm": length_mm,
    "width_mm": width_mm,
    "height_mm": height_mm,
    "seats": seats,
    "cargo_volume_l": cargo_volume_l,
    "drivetrain": drivetrain,
    "segment": segment,
    "car_body_type": car_body_type,
}])

st.divider()

if st.button("Predict Range", type="primary"):
    results = []
    for name, pipe in models.items():
        try:
            pred = pipe.predict(input_df)[0]
            results.append({"Model": name, "Predicted Range (km)": round(float(pred), 1)})
        except Exception as e:
            results.append({"Model": name, "Predicted Range (km)": f"Error: {e}"})

    results_df = pd.DataFrame(results)
    st.subheader("Predictions")
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    numeric_results = results_df[
        pd.to_numeric(results_df["Predicted Range (km)"], errors="coerce").notna()
    ]
    if not numeric_results.empty:
        st.bar_chart(numeric_results.set_index("Model")["Predicted Range (km)"])
        avg_pred = pd.to_numeric(numeric_results["Predicted Range (km)"]).mean()
        st.metric("Average Predicted Range Across Models", f"{avg_pred:.1f} km")

    st.divider()
    st.subheader("Cluster-based Estimate")

    if has_clustering:
        try:
            numeric_cols = [
                "top_speed_kmh", "battery_capacity_kWh", "torque_nm",
                "efficiency_wh_per_km", "acceleration_0_100_s",
                "fast_charging_power_kw_dc", "towing_capacity_kg",
                "cargo_volume_l", "seats", "length_mm", "width_mm", "height_mm",
            ]
            numeric_input = input_df[numeric_cols]

            scaled_input = cluster_assets["scaler"].transform(numeric_input)
            cluster_label = cluster_assets["kmeans"].predict(scaled_input)[0]
            avg_range_map = cluster_assets["cluster_avg_range"]
            cluster_avg = avg_range_map.get(cluster_label)

            st.write(f"This input falls into **Cluster {cluster_label}**.")
            if cluster_avg is not None:
                st.metric(f"Average Range of Cluster {cluster_label}", f"{cluster_avg:.1f} km")
            else:
                st.warning("No average range recorded for this cluster.")
        except Exception as e:
            st.warning(f"Could not compute cluster estimate: {e}")
    else:
        st.info(
            "Cluster-based estimate unavailable — missing one of: "
            "kmeans_model.pkl, scaler_for_clustering.pkl, cluster_avg_range.pkl "
            "in the app folder."
        )