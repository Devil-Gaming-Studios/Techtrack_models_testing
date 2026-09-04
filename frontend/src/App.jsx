import { useState } from "react";

const API_URL = "https://techtrack-models-testing.onrender.com/"; // change to your deployed backend URL

const DEFAULTS = {
  battery_capacity_kWh: 60,
  torque_nm: 300,
  top_speed_kmh: 180,
  acceleration_0_100_s: 7.5,
  fast_charging_power_kw_dc: 100,
  towing_capacity_kg: 0,
  efficiency_wh_per_km: 150,
  length_mm: 4500,
  width_mm: 1850,
  height_mm: 1550,
  seats: 5,
  cargo_volume_l: 450,
  drivetrain: "FWD",
  segment: "C - Mid-size",
  car_body_type: "Hatchback",
};

const NUMERIC_FIELDS = [
  ["battery_capacity_kWh", "Battery Capacity (kWh)"],
  ["torque_nm", "Torque (Nm)"],
  ["top_speed_kmh", "Top Speed (km/h)"],
  ["acceleration_0_100_s", "0-100 km/h Accel (s)"],
  ["fast_charging_power_kw_dc", "Fast Charging Power (kW DC)"],
  ["towing_capacity_kg", "Towing Capacity (kg)"],
  ["efficiency_wh_per_km", "Efficiency (Wh/km)"],
  ["length_mm", "Length (mm)"],
  ["width_mm", "Width (mm)"],
  ["height_mm", "Height (mm)"],
  ["seats", "Seats"],
  ["cargo_volume_l", "Cargo Volume (L)"],
];

const DRIVETRAIN_OPTIONS = ["FWD", "RWD", "AWD"];
const SEGMENT_OPTIONS = [
  "A - Mini", "B - Compact", "C - Mid-size", "D - Large", "E - Premium",
  "F - Luxury", "SUV", "Crossover", "Pickup", "Van", "Sports", "Other",
];
const BODY_TYPE_OPTIONS = [
  "Hatchback", "Sedan", "SUV", "Crossover", "Coupe", "Pickup", "Van", "Wagon",
];

export default function App() {
  const [form, setForm] = useState(DEFAULTS);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload = {
        ...form,
        seats: Number(form.seats),
      };
      NUMERIC_FIELDS.forEach(([key]) => {
        if (key !== "seats") payload[key] = Number(form[key]);
      });

      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed (${res.status})`);
      }

      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 720, margin: "0 auto", padding: 24, fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ marginBottom: 4 }}>🔋 EV Range Predictor</h1>
      <p style={{ color: "#666", marginTop: 0 }}>
        Enter EV specifications and compare predictions across all trained models.
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 24 }}>
        {NUMERIC_FIELDS.map(([key, label]) => (
          <label key={key} style={{ display: "flex", flexDirection: "column", fontSize: 13 }}>
            {label}
            <input
              type="number"
              value={form[key]}
              onChange={(e) => handleChange(key, e.target.value)}
              style={{ padding: 8, borderRadius: 6, border: "1px solid #ccc", marginTop: 4 }}
            />
          </label>
        ))}

        <label style={{ display: "flex", flexDirection: "column", fontSize: 13 }}>
          Drivetrain
          <select
            value={form.drivetrain}
            onChange={(e) => handleChange("drivetrain", e.target.value)}
            style={{ padding: 8, borderRadius: 6, border: "1px solid #ccc", marginTop: 4 }}
          >
            {DRIVETRAIN_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        </label>

        <label style={{ display: "flex", flexDirection: "column", fontSize: 13 }}>
          Segment
          <select
            value={form.segment}
            onChange={(e) => handleChange("segment", e.target.value)}
            style={{ padding: 8, borderRadius: 6, border: "1px solid #ccc", marginTop: 4 }}
          >
            {SEGMENT_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        </label>

        <label style={{ display: "flex", flexDirection: "column", fontSize: 13 }}>
          Body Type
          <select
            value={form.car_body_type}
            onChange={(e) => handleChange("car_body_type", e.target.value)}
            style={{ padding: 8, borderRadius: 6, border: "1px solid #ccc", marginTop: 4 }}
          >
            {BODY_TYPE_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
          </select>
        </label>
      </div>

      <button
        onClick={handleSubmit}
        disabled={loading}
        style={{
          marginTop: 24, padding: "10px 20px", borderRadius: 8, border: "none",
          background: "#2563eb", color: "white", fontWeight: 600, cursor: "pointer",
        }}
      >
        {loading ? "Predicting..." : "Predict Range"}
      </button>

      {error && (
        <p style={{ color: "crimson", marginTop: 16 }}>Error: {error}</p>
      )}

      {result && (
        <div style={{ marginTop: 32 }}>
          <h2>Predictions</h2>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                <th style={{ textAlign: "left", borderBottom: "1px solid #ddd", padding: 8 }}>Model</th>
                <th style={{ textAlign: "right", borderBottom: "1px solid #ddd", padding: 8 }}>Predicted Range (km)</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(result.predictions).map(([model, pred]) => (
                <tr key={model}>
                  <td style={{ padding: 8, borderBottom: "1px solid #f0f0f0" }}>{model}</td>
                  <td style={{ padding: 8, borderBottom: "1px solid #f0f0f0", textAlign: "right" }}>{pred}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {result.average_prediction_km != null && (
            <p style={{ marginTop: 16, fontWeight: 600 }}>
              Average predicted range: {result.average_prediction_km} km
            </p>
          )}

          {result.cluster_estimate && !result.cluster_estimate.error && (
            <div style={{ marginTop: 16, padding: 12, background: "#f5f7fa", borderRadius: 8 }}>
              <p style={{ margin: 0 }}>
                Falls into <strong>Cluster {result.cluster_estimate.cluster}</strong> —
                average range for this cluster: <strong>{result.cluster_estimate.average_range_km} km</strong>
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
