"""
app.py  -  Used Car Price Prediction
=========================================
ONE file for both backend (Flask API) and frontend (Streamlit UI).

  Run Flask API  :  python app.py
  Run Streamlit  :  streamlit run app.py

Flask API endpoints (port 5000):
  GET  /api/health
  GET  /api/model-info
  GET  /api/dataset-info
  POST /api/predict
  POST /api/batch-predict
"""

import os
import re
import sys

# ── Detect whether we are running under Streamlit ─────────────────────────────
# When Streamlit executes this file it injects its own argv; we detect that
# so we can branch: Streamlit path vs Flask path.
_IS_STREAMLIT = "streamlit" in sys.modules or any(
    "streamlit" in a for a in sys.argv
)

# ==============================================================================
# SHARED HELPERS  (used by both Flask and Streamlit)
# ==============================================================================
import re
import numpy as np
import pandas as pd
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "used_car_rf_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "..", "models", "model_meta.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "..", "test-data.csv")


def _extract_numeric(val):
    """Pull the first float from a string like '1200 CC' -> 1200.0"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    m = re.search(r"([\d.]+)", str(val))
    return float(m.group(1)) if m else None


def _extract_mileage(val):
    """Parse mileage string; convert km/kg -> kmpl (*1.4)."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    val = str(val).strip()
    m = re.match(r"([\d.]+)\s*(kmpl|km/kg)?", val, re.I)
    if not m:
        return None
    num = float(m.group(1))
    if (m.group(2) or "").strip().lower() == "km/kg":
        num *= 1.4
    return num


def build_input_df(data: dict) -> pd.DataFrame:
    """Convert a user-submitted dict into the DataFrame the model expects."""
    year = int(data.get("year", 2018))
    row = {
        "Brand":             data.get("brand", "Maruti"),
        "Location":          data.get("location", "Mumbai"),
        "Fuel_Type":         data.get("fuel_type", "Petrol"),
        "Transmission":      data.get("transmission", "Manual"),
        "Owner_Type":        data.get("owner_type", "First"),
        "Car_Age":           2024 - year,
        "Kilometers_Driven": int(data.get("kilometers_driven", 50000)),
        "Mileage":           _extract_mileage(str(data.get("mileage", "18 kmpl"))),
        "Engine":            _extract_numeric(str(data.get("engine", "1200 CC"))),
        "Power":             _extract_numeric(str(data.get("power", "80 bhp"))),
        "Seats":             float(data.get("seats", 5)),
    }
    return pd.DataFrame([row])


def load_model():
    """Load and return (model, meta). Returns (None, {}) if not trained yet."""
    try:
        m    = joblib.load(MODEL_PATH)
        meta = joblib.load(META_PATH)
        return m, meta
    except Exception as e:
        print(f"[WARN] Could not load model: {e}  — run train_model.py first.")
        return None, {}


# ==============================================================================
# FLASK  BACKEND
# ==============================================================================
if not _IS_STREAMLIT:
    from flask import Flask, request, jsonify
    from flask_cors import CORS

    flask_app = Flask(__name__)
    CORS(flask_app)

    print("Loading model ...")
    _model, _meta = load_model()
    if _model:
        print("  Model loaded OK")

    # ── Routes ─────────────────────────────────────────────────────────────────
    @flask_app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({
            "status":       "ok",
            "model_loaded": _model is not None,
            "message":      "Used Car Price Prediction API is running",
        })

    @flask_app.route("/api/model-info", methods=["GET"])
    def model_info():
        if not _meta:
            return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503
        return jsonify({
            "model_type": "RandomForestRegressor",
            "total_rows": _meta.get("total_rows"),
            "train_rows": _meta.get("train_rows"),
            "test_rows":  _meta.get("test_rows"),
            "features":   _meta.get("features"),
            "metrics": {
                "MAE_Lakh":   _meta.get("mae"),
                "RMSE_Lakh":  _meta.get("rmse"),
                "R2_score":   _meta.get("r2"),
                "CV_R2_mean": _meta.get("cv_r2_mean"),
                "CV_R2_std":  _meta.get("cv_r2_std"),
            },
            "options": {
                "brands":        _meta.get("brands"),
                "locations":     _meta.get("locations"),
                "fuel_types":    _meta.get("fuel_types"),
                "transmissions": _meta.get("transmissions"),
                "owner_types":   _meta.get("owner_types"),
                "year_range":    [_meta.get("year_min"), _meta.get("year_max")],
                "km_range":      [_meta.get("km_min"),   _meta.get("km_max")],
            },
        })

    @flask_app.route("/api/dataset-info", methods=["GET"])
    def dataset_info():
        try:
            df = pd.read_csv(DATA_PATH)
            df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True)
            return jsonify({
                "total_rows":        int(len(df)),
                "columns":           df.columns.tolist(),
                "fuel_type_dist":    df["Fuel_Type"].value_counts().to_dict(),
                "transmission_dist": df["Transmission"].value_counts().to_dict(),
                "owner_type_dist":   df["Owner_Type"].value_counts().to_dict(),
                "location_dist":     df["Location"].value_counts().to_dict(),
                "year_stats": {
                    "min":  int(df["Year"].min()),
                    "max":  int(df["Year"].max()),
                    "mean": round(float(df["Year"].mean()), 1),
                },
                "km_stats": {
                    "min":  int(df["Kilometers_Driven"].min()),
                    "max":  int(df["Kilometers_Driven"].max()),
                    "mean": round(float(df["Kilometers_Driven"].mean()), 0),
                },
                "null_counts": df.isnull().sum().to_dict(),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @flask_app.route("/api/predict", methods=["POST"])
    def predict():
        if _model is None:
            return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503
        try:
            data = request.get_json(force=True)
            if not data:
                return jsonify({"error": "Request body must be JSON"}), 400
            price = round(max(0.1, float(_model.predict(build_input_df(data))[0])), 2)
            return jsonify({
                "status":          "success",
                "predicted_price": price,
                "unit":            "Lakh INR",
                "input_received":  data,
                "price_range": {
                    "low":  round(price * 0.90, 2),
                    "high": round(price * 1.10, 2),
                },
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @flask_app.route("/api/batch-predict", methods=["POST"])
    def batch_predict():
        if _model is None:
            return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503
        try:
            data = request.get_json(force=True)
            if not isinstance(data, list):
                return jsonify({"error": "Expected a JSON array of car objects"}), 400
            results = []
            for i, item in enumerate(data):
                try:
                    price = round(max(0.1, float(_model.predict(build_input_df(item))[0])), 2)
                    results.append({"index": i, "predicted_price": price,
                                    "unit": "Lakh INR", "input": item})
                except Exception as e:
                    results.append({"index": i, "error": str(e), "input": item})
            return jsonify({"status": "success", "count": len(results), "predictions": results})
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    # ── Entry point ─────────────────────────────────────────────────────────────
    if __name__ == "__main__":
        print("Starting Flask server on http://127.0.0.1:5000 ...")
        flask_app.run(host="127.0.0.1", port=5000, debug=False)


# ==============================================================================
# STREAMLIT  FRONTEND
# ==============================================================================
else:
    import streamlit as st
    import requests
    import matplotlib.pyplot as plt

    API_BASE = "http://127.0.0.1:5000/api"

    # ── Page config ─────────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Used Car Price Predictor",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ───────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    /* ════════════════════════════════════════════════════════════════
       GLOBAL
    ════════════════════════════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif !important;
    }
    .stApp {
        background: #f1f5f9;
    }
    .main .block-container {
        padding: 1.2rem 2.8rem 2.5rem 2.8rem;
        max-width: 1380px;
    }

    /* ════════════════════════════════════════════════════════════════
       SIDEBAR
    ════════════════════════════════════════════════════════════════ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(175deg, #0f172a 0%, #1e3a8a 60%, #1d4ed8 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }

    /* Sidebar text: headings, paragraphs, markdown → light */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown * {
        color: #e2e8f0 !important;
    }

    /* Widget labels (the text above each widget) */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
    section[data-testid="stSidebar"] [data-testid="InputInstructions"],
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label {
        color: #e2e8f0 !important;
    }

    /* Widget input boxes: white bg, dark text */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] select {
        color: #0f172a !important;
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }

    /* Selectbox: the control box showing selected value */
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-baseweb="select"] > div > div,
    section[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stSelectboxValue"],
    section[data-testid="stSidebar"] [data-baseweb="select"] span {
        color: #0f172a !important;
        background: #ffffff !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="select"] {
        background: #ffffff !important;
        border-radius: 8px !important;
    }

    /* Number input: value display & stepper buttons */
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
        color: #0f172a !important;
        background: #ffffff !important;
    }
    section[data-testid="stSidebar"] [data-testid="stNumberInputStepDown"],
    section[data-testid="stSidebar"] [data-testid="stNumberInputStepUp"] {
        color: #0f172a !important;
        background: #f1f5f9 !important;
        border-color: #cbd5e1 !important;
    }

    /* Slider: tick labels and current value tooltip */
    section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBar"],
    section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMin"],
    section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMax"],
    section[data-testid="stSidebar"] .stSlider span,
    section[data-testid="stSidebar"] .stSlider div[class*="sliderThumb"] {
        color: #e2e8f0 !important;
    }
    /* Slider track fill */
    section[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
        background: #60a5fa !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="slider"] div[class*="track"] {
        background: rgba(255,255,255,0.2) !important;
    }

    /* Sidebar heading underline */
    section[data-testid="stSidebar"] h2 {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        border-bottom: 2px solid rgba(96,165,250,0.6);
        padding-bottom: 0.4rem;
        margin-bottom: 0.6rem !important;
    }
    /* Divider inside sidebar */
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15) !important;
        margin: 0.7rem 0 !important;
    }

    /* ════════════════════════════════════════════════════════════════
       MAIN HEADER BANNER
    ════════════════════════════════════════════════════════════════ */
    .app-header {
        display: flex;
        align-items: center;
        gap: 1.2rem;
        padding: 1rem 1.6rem;
        background: linear-gradient(135deg, #1e40af, #4f46e5);
        border-radius: 16px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(30,64,175,0.3);
    }
    .app-header .icon { font-size: 2.8rem; line-height: 1; }
    .app-header h1 {
        margin: 0;
        color: #ffffff;
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .app-header p {
        margin: 0.2rem 0 0 0;
        color: #bfdbfe;
        font-size: 0.92rem;
        font-weight: 500;
    }

    /* ════════════════════════════════════════════════════════════════
       TABS (ULTIMATE VISIBILITY FIX - ZERO OPACITY FADING)
    ════════════════════════════════════════════════════════════════ */
    
    /* 1. Background for the Tab Container */
    div[data-testid="stTabs"] > div > div:first-child,
    .stTabs [data-baseweb="tab-list"] {
        background: #dde3ed !important;
        padding: 6px !important;
        border-radius: 12px !important;
        border: 1px solid #c8d1df !important;
        gap: 6px !important;
    }

    /* 2. Style ALL Tab Buttons (Default Unselected State) */
    div[data-testid="stTabs"] button {
        background: transparent !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 1.4rem !important;
        opacity: 1 !important; /* Force completely opaque */
        visibility: visible !important;
        transition: all 0.2s ease !important;
    }

    /* Force Pitch-Black Bold Text on UNSELECTED Tabs */
    div[data-testid="stTabs"] button p,
    div[data-testid="stTabs"] button span,
    div[data-testid="stTabs"] button div,
    div[data-testid="stTabs"] button * {
        color: #000000 !important;  /* Absolute Black for extreme visibility */
        font-weight: 900 !important; /* Extra Bold */
        font-size: 1.05rem !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    /* 3. Style SELECTED Tab State */
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background: #2563eb !important; /* Vivid Blue Background */
        box-shadow: 0 4px 12px rgba(37,99,235,0.4) !important;
        opacity: 1 !important;
    }

    /* Force Pure White Bold Text on SELECTED Tabs */
    div[data-testid="stTabs"] button[aria-selected="true"] p,
    div[data-testid="stTabs"] button[aria-selected="true"] span,
    div[data-testid="stTabs"] button[aria-selected="true"] div,
    div[data-testid="stTabs"] button[aria-selected="true"] * {
        color: #ffffff !important; /* Absolute White */
        font-weight: 900 !important;
    }

    /* 4. Hover States */
    div[data-testid="stTabs"] button:hover {
        background: rgba(37,99,235,0.15) !important;
        opacity: 1 !important;
    }
    div[data-testid="stTabs"] button:hover * {
        color: #1d4ed8 !important; /* Turn deep blue on hover */
    }

    div[data-testid="stTabs"] button[aria-selected="true"]:hover {
        background: #1e40af !important; /* Darker blue on selected hover */
    }
    div[data-testid="stTabs"] button[aria-selected="true"]:hover * {
        color: #ffffff !important;
    }

    /* Add Padding below the tab menu */
    [data-testid="stTabBase"], 
    .stTabs [data-baseweb="tab-panel"] {
        padding: 1.6rem 0 0.5rem 0 !important;
    }

    /* ════════════════════════════════════════════════════════════════
       SECTION HEADERS
    ════════════════════════════════════════════════════════════════ */
    .section-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0f172a;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        border-left: 4px solid #2563eb;
        padding: 0.3rem 0 0.3rem 0.75rem;
        background: linear-gradient(90deg, #eff6ff, transparent);
        border-radius: 0 6px 6px 0;
        margin: 1.6rem 0 1rem 0;
    }

    /* ════════════════════════════════════════════════════════════════
       PRICE RESULT BOX
    ════════════════════════════════════════════════════════════════ */
    .price-box {
        background: linear-gradient(135deg, #1e40af 0%, #4f46e5 50%, #7c3aed 100%);
        color: #ffffff;
        border-radius: 20px;
        padding: 2.4rem 2.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(30,64,175,0.4);
        margin-bottom: 1.2rem;
        position: relative;
        overflow: hidden;
    }
    .price-box::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 180px; height: 180px;
        background: rgba(255,255,255,0.06);
        border-radius: 50%;
    }
    .price-box .label {
        font-size: 0.88rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #bfdbfe;
        margin: 0 0 0.4rem 0;
    }
    .price-box h1 {
        font-size: 3.4rem;
        margin: 0 0 0.4rem 0;
        font-weight: 900;
        letter-spacing: -0.02em;
        color: #ffffff;
        line-height: 1;
    }
    .price-box .range {
        font-size: 1rem;
        color: #bfdbfe;
        margin: 0;
        font-weight: 500;
    }

    /* ════════════════════════════════════════════════════════════════
       METRIC CARDS  (Tab 2)
    ════════════════════════════════════════════════════════════════ */
    .metric-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.4rem 1.6rem 1.2rem 1.6rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border-top: 4px solid #2563eb;
        position: relative;
    }
    .metric-card .mc-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0 0 0.5rem 0;
    }
    .metric-card .mc-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #2563eb;
        margin: 0;
        line-height: 1;
    }
    .metric-card .mc-sub {
        font-size: 0.78rem;
        color: #94a3b8;
        margin: 0.3rem 0 0 0;
        font-weight: 500;
    }

    /* ════════════════════════════════════════════════════════════════
       INFO CARDS  (wrapper for chart + stat panels)
    ════════════════════════════════════════════════════════════════ */
    .info-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
    }

    /* ════════════════════════════════════════════════════════════════
       ST.METRIC WIDGETS
    ════════════════════════════════════════════════════════════════ */
    [data-testid="stMetric"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem 1.2rem !important;
        box-shadow: 0 1px 5px rgba(0,0,0,0.07);
        border-top: 3px solid #2563eb;
    }
    [data-testid="stMetricLabel"] > div {
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        color: #64748b !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
    }
    [data-testid="stMetricValue"] > div {
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
    }

    /* ════════════════════════════════════════════════════════════════
       FEATURE PILLS
    ════════════════════════════════════════════════════════════════ */
    .feature-pill {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        border: 1.5px solid #bfdbfe;
        border-radius: 20px;
        padding: 0.28rem 0.9rem;
        font-size: 0.83rem;
        font-weight: 600;
        margin: 0.18rem 0.1rem;
        letter-spacing: 0.01em;
    }

    /* ════════════════════════════════════════════════════════════════
       MARKDOWN TABLES
    ════════════════════════════════════════════════════════════════ */
    .stMarkdown table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.91rem;
        margin: 0.4rem 0 1rem 0;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .stMarkdown th {
        background: #2563eb;
        color: #ffffff !important;
        padding: 0.6rem 1rem;
        text-align: left;
        font-weight: 700;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stMarkdown td {
        padding: 0.52rem 1rem;
        border-bottom: 1px solid #e2e8f0;
        color: #1e293b;
        font-size: 0.91rem;
    }
    .stMarkdown tr:last-child td { border-bottom: none; }
    .stMarkdown tr:nth-child(even) td { background: #f8fafc; }
    .stMarkdown tr:hover td { background: #eff6ff !important; }

    /* ════════════════════════════════════════════════════════════════
       CODE BLOCKS  (inline in tables)
    ════════════════════════════════════════════════════════════════ */
    code {
        background: #eff6ff;
        color: #1d4ed8;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: 600;
    }

    /* ════════════════════════════════════════════════════════════════
       ALERTS / BANNERS
    ════════════════════════════════════════════════════════════════ */
    [data-testid="stAlert"] {
        border-radius: 12px !important;
        font-size: 0.94rem !important;
        font-weight: 500;
        border-width: 1.5px !important;
    }

    /* ════════════════════════════════════════════════════════════════
       SPINNER
    ════════════════════════════════════════════════════════════════ */
    [data-testid="stSpinner"] p {
        color: #2563eb !important;
        font-weight: 600;
    }

    /* ════════════════════════════════════════════════════════════════
       HIDE STREAMLIT CHROME
    ════════════════════════════════════════════════════════════════ */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    [data-testid="stDecoration"] { display: none; }

    /* ════════════════════════════════════════════════════════════════
       PERSISTENT SIDEBAR TOGGLE BUTTONS
    ════════════════════════════════════════════════════════════════ */
    header[data-testid="stHeader"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {
        opacity: 1 !important;
        visibility: visible !important;
        z-index: 999999 !important;
    }

    [data-testid="collapsedControl"] button,
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stExpandSidebarButton"] {
        background: #1e40af !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 10px rgba(30,64,175,0.35) !important;
        border: none !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    [data-testid="collapsedControl"] button:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover,
    [data-testid="stExpandSidebarButton"]:hover {
        background: #1d4ed8 !important;
    }

    [data-testid="collapsedControl"] svg,
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stExpandSidebarButton"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] button {
        opacity: 1 !important;
        visibility: visible !important;
    }

    [data-testid="stSidebarCollapseButton"] button {
        background: rgba(255,255,255,0.15) !important;
        border-radius: 8px !important;
        border: none !important;
    }

    [data-testid="stSidebarCollapseButton"] button:hover {
        background: rgba(255,255,255,0.25) !important;
    }

    [data-testid="stSidebarCollapseButton"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Cached API helpers ───────────────────────────────────────────────────────
    @st.cache_data(ttl=300)
    def fetch_model_info():
        try:
            r = requests.get(f"{API_BASE}/model-info", timeout=5)
            return r.json() if r.ok else None
        except Exception:
            return None

    @st.cache_data(ttl=300)
    def fetch_dataset_info():
        try:
            r = requests.get(f"{API_BASE}/dataset-info", timeout=5)
            return r.json() if r.ok else None
        except Exception:
            return None

    def check_backend():
        try:
            return requests.get(f"{API_BASE}/health", timeout=3).ok
        except Exception:
            return False

    def do_predict(payload: dict):
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
        return r.json()

    # ── Header ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
        <span class="icon">🚗</span>
        <div>
            <h1>Used Car Price Predictor</h1>
            <p>Powered by Random Forest &nbsp;&bull;&nbsp; IBM BOB AI/ML Project &nbsp;&bull;&nbsp; Flask + Streamlit</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    backend_ok = check_backend()
    if backend_ok:
        st.success("Backend connected  •  http://127.0.0.1:5000")
    else:
        st.error("Backend offline — open a terminal and run:  python backend/app.py")

    # ── Sidebar – input form ─────────────────────────────────────────────────────
    model_info = fetch_model_info()
    with st.sidebar:
        st.markdown("## 🚗 Car Details")
        st.markdown("<p style='color:#94a3b8;font-size:0.83rem;margin:0 0 0.5rem 0;'>Fill in all fields and click Predict Price.</p>", unsafe_allow_html=True)
        st.markdown("---")

        opts      = model_info.get("options", {}) if model_info else {}
        brands    = opts.get("brands",        ["Maruti","Hyundai","Honda","Toyota","Ford"])
        locations = opts.get("locations",     ["Mumbai","Delhi","Bangalore","Chennai","Pune"])
        fuels     = opts.get("fuel_types",    ["Petrol","Diesel","CNG","LPG"])
        trans     = opts.get("transmissions", ["Manual","Automatic"])
        owners    = opts.get("owner_types",   ["First","Second","Third","Fourth & Above"])

        brand        = st.selectbox("Car Brand",           brands)
        location     = st.selectbox("Location",            locations)
        year         = st.slider("Manufacturing Year",     1996, 2019, 2016)
        km_driven    = st.number_input("Kilometers Driven", min_value=1000,
                                        max_value=350000, value=45000, step=1000)
        fuel_type    = st.selectbox("Fuel Type",           fuels)
        transmission = st.selectbox("Transmission",        trans)
        owner_type   = st.selectbox("Owner Type",          owners)
        mileage      = st.number_input("Mileage (kmpl)",   min_value=5.0,
                                        max_value=35.0, value=18.0, step=0.5)
        engine       = st.number_input("Engine (CC)",      min_value=600,
                                        max_value=5000, value=1200, step=50)
        power        = st.number_input("Power (bhp)",      min_value=30.0,
                                        max_value=600.0, value=80.0, step=5.0)
        seats        = st.selectbox("Seats",               [2,4,5,6,7,8,9], index=2)

        st.markdown("---")
        predict_btn = st.button("Predict Price", use_container_width=True, type="primary")

    # ── Tabs ─────────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯  Prediction", "📊  Model Metrics", "🗂  Dataset Overview", "ℹ  About"
    ])

    # ── Tab 1: Prediction ────────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">Price Prediction</div>',
                    unsafe_allow_html=True)

        if predict_btn or st.session_state.get("last_prediction"):
            if predict_btn:
                if not backend_ok:
                    st.error("⚠️ Backend is offline. Please start the Flask server first.")
                else:
                    payload = {
                        "brand":             brand,
                        "location":          location,
                        "year":              year,
                        "kilometers_driven": km_driven,
                        "fuel_type":         fuel_type,
                        "transmission":      transmission,
                        "owner_type":        owner_type,
                        "mileage":           f"{mileage} kmpl",
                        "engine":            f"{engine} CC",
                        "power":             f"{power} bhp",
                        "seats":             seats,
                    }
                    with st.spinner("Calling prediction API…"):
                        result = do_predict(payload)
                    st.session_state["last_prediction"] = result
                    st.session_state["last_payload"]    = payload

            result  = st.session_state.get("last_prediction", {})
            payload = st.session_state.get("last_payload", {})

            if "predicted_price" in result:
                price = result["predicted_price"]
                low   = result["price_range"]["low"]
                high  = result["price_range"]["high"]

                st.markdown(f"""
                <div class="price-box">
                    <p class="label">&#128200; Estimated Market Price</p>
                    <h1>&#8377;&nbsp;{price:.2f} <span style='font-size:1.4rem;font-weight:600;opacity:0.75'>Lakh</span></h1>
                    <p class="range">Confidence range &nbsp;&#8377; {low:.2f} L &nbsp;&#8211;&nbsp; &#8377; {high:.2f} L</p>
                </div>
                """, unsafe_allow_html=True)

                # Car details metrics
                st.markdown('<div class="section-header">Car Details Summary</div>',
                            unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🏷 Brand",        payload.get("brand", "-"))
                c2.metric("📅 Year",          payload.get("year", "-"))
                c3.metric("⛽ Fuel Type",     payload.get("fuel_type", "-"))
                c4.metric("⚙ Transmission",  payload.get("transmission", "-"))

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("🛣 KM Driven",    f"{payload.get('kilometers_driven', 0):,}")
                c6.metric("🔧 Engine",        payload.get("engine", "-"))
                c7.metric("⚡ Power",         payload.get("power", "-"))
                c8.metric("👤 Owner Type",    payload.get("owner_type", "-"))

                # Price range chart
                st.markdown('<div class="section-header">Price Range Visualisation</div>',
                            unsafe_allow_html=True)
                y_pos = np.arange(3)
                fig, ax = plt.subplots(figsize=(9, 3.0), facecolor="#ffffff")
                ax.set_facecolor("#f8fafc")
                bar_colors = ["#93c5fd", "#2563eb", "#93c5fd"]
                bars = ax.barh(y_pos, [low, price, high],
                               color=bar_colors, height=0.5, zorder=3)
                # Add value labels
                for idx, v in enumerate([low, price, high]):
                    ax.text(v + high * 0.015, idx,
                            f"\u20b9 {v:.2f} L",
                            va="center", fontsize=12,
                            fontweight="bold" if idx == 1 else "normal",
                            color="#1e293b")
                ax.set_yticks(y_pos)
                ax.set_yticklabels(["Low\nEstimate", "Predicted\nPrice", "High\nEstimate"],
                                   fontsize=11, fontweight="bold", color="#1e293b")
                ax.set_xlabel("Price (Lakh INR)", fontsize=10, color="#64748b", labelpad=8)
                ax.set_title("Predicted Price Range", fontsize=13,
                             fontweight="bold", color="#0f172a", pad=14)
                ax.set_xlim(0, high * 1.32)
                ax.tick_params(axis="x", labelsize=9, colors="#64748b")
                ax.grid(axis="x", color="#e2e8f0", linestyle="--", linewidth=0.8, zorder=0)
                ax.spines[["top", "right", "left"]].set_visible(False)
                ax.spines["bottom"].set_color("#e2e8f0")
                plt.tight_layout(pad=1.4)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            elif "error" in result:
                st.error(f"❌ Prediction failed: {result['error']}")
        else:
            st.info("👈  Fill in the car details in the sidebar and click **Predict Price**.")

    # ── Tab 2: Model Metrics ─────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">Model Performance Metrics</div>',
                    unsafe_allow_html=True)
        if model_info:
            metrics = model_info.get("metrics", {})

            # Four metric cards
            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(f"""
            <div class="metric-card" style="border-top-color:#2563eb">
                <p class="mc-label">&#x1F4C8; R&sup2; Score</p>
                <p class="mc-value" style="color:#2563eb">{metrics.get('R2_score', '-')}</p>
                <p class="mc-sub">Higher is better &nbsp;(max 1.0)</p>
            </div>""", unsafe_allow_html=True)
            m2.markdown(f"""
            <div class="metric-card" style="border-top-color:#7c3aed">
                <p class="mc-label">&#x1F4B0; MAE (Lakh)</p>
                <p class="mc-value" style="color:#7c3aed">{metrics.get('MAE_Lakh', '-')}</p>
                <p class="mc-sub">Mean Abs. Error</p>
            </div>""", unsafe_allow_html=True)
            m3.markdown(f"""
            <div class="metric-card" style="border-top-color:#059669">
                <p class="mc-label">&#x1F4CF; RMSE (Lakh)</p>
                <p class="mc-value" style="color:#059669">{metrics.get('RMSE_Lakh', '-')}</p>
                <p class="mc-sub">Root Mean Sq. Error</p>
            </div>""", unsafe_allow_html=True)
            m4.markdown(f"""
            <div class="metric-card" style="border-top-color:#d97706">
                <p class="mc-label">&#x1F504; CV R&sup2; (5-fold)</p>
                <p class="mc-value" style="color:#d97706">{metrics.get('CV_R2_mean', '-')}</p>
                <p class="mc-sub">Cross-validated R&sup2;</p>
            </div>""", unsafe_allow_html=True)

            # Charts
            st.markdown('<div class="section-header">Train / Test Split & Metrics Chart</div>',
                        unsafe_allow_html=True)
            sc1, sc2 = st.columns(2)
            with sc1:
                fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor="#ffffff")
                train_n = model_info.get("train_rows", 977)
                test_n  = model_info.get("test_rows", 244)
                wedges, texts, autotexts = ax.pie(
                    [train_n, test_n],
                    labels=["Train", "Test"],
                    colors=["#2563eb", "#93c5fd"],
                    autopct="%1.1f%%", startangle=90,
                    wedgeprops={"edgecolor": "white", "linewidth": 3},
                    textprops={"fontsize": 12, "color": "#0f172a", "fontweight": "600"},
                    pctdistance=0.72,
                )
                for at in autotexts:
                    at.set_fontsize(11)
                    at.set_fontweight("bold")
                    at.set_color("#ffffff")
                ax.set_title(
                    f"Dataset Split\nTotal: {model_info.get('total_rows', '')} rows",
                    fontsize=11, color="#0f172a", fontweight="bold", pad=12
                )
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with sc2:
                m_names = ["R²\nScore", "MAE\n(Lakh)", "RMSE\n(Lakh)", "CV-R²\n(5-fold)"]
                m_vals  = [
                    metrics.get("R2_score", 0),
                    metrics.get("MAE_Lakh", 0),
                    metrics.get("RMSE_Lakh", 0),
                    metrics.get("CV_R2_mean", 0),
                ]
                bar_colors = ["#2563eb", "#7c3aed", "#059669", "#d97706"]
                fig2, ax2 = plt.subplots(figsize=(5.5, 4.5), facecolor="#ffffff")
                ax2.set_facecolor("#f8fafc")
                bars = ax2.bar(m_names, m_vals, color=bar_colors, width=0.52,
                               zorder=3, edgecolor="white", linewidth=1.5)
                for bar, v in zip(bars, m_vals):
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2,
                        v + max(m_vals) * 0.025,
                        str(round(v, 3)),
                        ha="center", va="bottom",
                        fontsize=11, fontweight="bold", color="#0f172a"
                    )
                ax2.set_title("Performance Metrics", fontsize=12, fontweight="bold",
                              color="#0f172a", pad=12)
                ax2.set_ylim(0, max(m_vals) * 1.38)
                ax2.tick_params(axis="x", labelsize=9.5, colors="#0f172a", pad=4)
                ax2.tick_params(axis="y", labelsize=8.5, colors="#64748b")
                ax2.grid(axis="y", color="#e2e8f0", linestyle="--",
                         linewidth=0.8, zorder=0)
                ax2.spines[["top", "right"]].set_visible(False)
                ax2.spines[["left", "bottom"]].set_color("#e2e8f0")
                plt.tight_layout(pad=1.4)
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

            st.markdown('<div class="section-header">Features Used by the Model</div>',
                        unsafe_allow_html=True)
            feats = model_info.get("features", [])
            st.markdown(
                " ".join(
                    f"<span class='feature-pill'>&#9881; {f}</span>"
                    for f in feats
                ),
                unsafe_allow_html=True,
            )
        else:
            st.warning("⚠️ Model info unavailable. Make sure the Flask backend is running.")

    # ── Tab 3: Dataset Overview ──────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Dataset Overview</div>',
                    unsafe_allow_html=True)
        ds = fetch_dataset_info()
        if ds:
            d1, d2, d3 = st.columns(3)
            d1.metric("📋 Total Records",   ds.get("total_rows", "-"))
            d2.metric("🗃 Total Features",  len(ds.get("columns", [])))
            d3.metric("📍 Cities Covered",  len(ds.get("location_dist", {})))

            st.markdown('<div class="section-header">Distribution Charts</div>',
                        unsafe_allow_html=True)
            ca, cb = st.columns(2)
            # Shared chart style helper
            _palette = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626"]

            with ca:
                fuel_dist = ds.get("fuel_type_dist", {})
                fig3, ax3 = plt.subplots(figsize=(5.2, 4.5), facecolor="#ffffff")
                wedges, texts, autotexts = ax3.pie(
                    list(fuel_dist.values()),
                    labels=list(fuel_dist.keys()),
                    autopct="%1.1f%%", startangle=90,
                    colors=_palette[:len(fuel_dist)],
                    wedgeprops={"edgecolor": "white", "linewidth": 2.5},
                    textprops={"fontsize": 11, "color": "#0f172a", "fontweight": "600"},
                    pctdistance=0.72,
                )
                for at in autotexts:
                    at.set_fontsize(10)
                    at.set_fontweight("bold")
                    at.set_color("#ffffff")
                ax3.set_title("Fuel Type Distribution", fontsize=12,
                              fontweight="bold", color="#0f172a", pad=12)
                plt.tight_layout()
                st.pyplot(fig3, use_container_width=True)
                plt.close(fig3)

            with cb:
                loc_dist   = ds.get("location_dist", {})
                sorted_loc = dict(sorted(loc_dist.items(),
                                         key=lambda x: x[1], reverse=True))
                fig4, ax4  = plt.subplots(figsize=(5.2, 4.5), facecolor="#ffffff")
                ax4.set_facecolor("#f8fafc")
                bars4 = ax4.barh(list(sorted_loc.keys()),
                                  list(sorted_loc.values()),
                                  color="#2563eb", alpha=0.88, zorder=3)
                ax4.set_title("Cars per City", fontsize=12,
                              fontweight="bold", color="#0f172a", pad=12)
                ax4.spines[["top", "right"]].set_visible(False)
                ax4.spines[["left", "bottom"]].set_color("#e2e8f0")
                ax4.set_xlabel("Number of Cars", color="#64748b", fontsize=9.5)
                ax4.tick_params(axis="y", labelsize=9.5, colors="#0f172a")
                ax4.tick_params(axis="x", labelsize=9, colors="#64748b")
                ax4.grid(axis="x", color="#e2e8f0", linestyle="--",
                         linewidth=0.7, zorder=0)
                for val, bar in zip(sorted_loc.values(), ax4.patches):
                    ax4.text(bar.get_width() + max(sorted_loc.values()) * 0.015,
                             bar.get_y() + bar.get_height() / 2,
                             str(val), va="center", fontsize=9.5,
                             fontweight="600", color="#0f172a")
                plt.tight_layout(pad=1.4)
                st.pyplot(fig4, use_container_width=True)
                plt.close(fig4)

            cc, cd = st.columns(2)
            with cc:
                tr_dist = ds.get("transmission_dist", {})
                fig5, ax5 = plt.subplots(figsize=(4.5, 3.6), facecolor="#ffffff")
                ax5.set_facecolor("#f8fafc")
                ax5.bar(list(tr_dist.keys()), list(tr_dist.values()),
                        color=["#2563eb", "#93c5fd"], width=0.45,
                        zorder=3, edgecolor="white", linewidth=1.5)
                for idx, v in enumerate(tr_dist.values()):
                    ax5.text(idx, v + max(tr_dist.values()) * 0.025,
                             str(v), ha="center", fontsize=12,
                             fontweight="bold", color="#0f172a")
                ax5.set_title("Transmission Distribution", fontsize=11,
                              fontweight="bold", color="#0f172a", pad=12)
                ax5.spines[["top", "right"]].set_visible(False)
                ax5.spines[["left", "bottom"]].set_color("#e2e8f0")
                ax5.tick_params(axis="x", labelsize=11, colors="#0f172a")
                ax5.tick_params(axis="y", labelsize=9, colors="#64748b")
                ax5.grid(axis="y", color="#e2e8f0", linestyle="--",
                         linewidth=0.7, zorder=0)
                plt.tight_layout(pad=1.4)
                st.pyplot(fig5, use_container_width=True)
                plt.close(fig5)

            with cd:
                ow_dist = ds.get("owner_type_dist", {})
                fig6, ax6 = plt.subplots(figsize=(4.5, 3.6), facecolor="#ffffff")
                ax6.set_facecolor("#f8fafc")
                ax6.bar(list(ow_dist.keys()), list(ow_dist.values()),
                        color=_palette[:len(ow_dist)],
                        zorder=3, edgecolor="white", linewidth=1.5)
                for idx, v in enumerate(ow_dist.values()):
                    ax6.text(idx, v + max(ow_dist.values()) * 0.025,
                             str(v), ha="center", fontsize=10,
                             fontweight="bold", color="#0f172a")
                ax6.set_title("Owner Type Distribution", fontsize=11,
                              fontweight="bold", color="#0f172a", pad=12)
                ax6.spines[["top", "right"]].set_visible(False)
                ax6.spines[["left", "bottom"]].set_color("#e2e8f0")
                ax6.tick_params(axis="x", labelsize=8.5, colors="#0f172a",
                                rotation=12)
                ax6.tick_params(axis="y", labelsize=9, colors="#64748b")
                ax6.grid(axis="y", color="#e2e8f0", linestyle="--",
                         linewidth=0.7, zorder=0)
                plt.tight_layout(pad=1.4)
                st.pyplot(fig6, use_container_width=True)
                plt.close(fig6)

            # Numeric stats
            yr = ds.get("year_stats", {})
            km = ds.get("km_stats",   {})
            st.markdown('<div class="section-header">Numeric Ranges</div>',
                        unsafe_allow_html=True)
            nr1, nr2, nr3 = st.columns(3)
            nr1.metric("📅 Year Min",  yr.get("min", "-"))
            nr2.metric("📅 Year Max",  yr.get("max", "-"))
            nr3.metric("📅 Year Avg",  yr.get("mean", "-"))
            nr4, nr5, nr6 = st.columns(3)
            nr4.metric("🛣 KM Min",  f"{km.get('min', 0):,}" if km.get("min") else "-")
            nr5.metric("🛣 KM Max",  f"{km.get('max', 0):,}")
            nr6.metric("🛣 KM Avg",  f"{int(km.get('mean', 0)):,}")
        else:
            st.warning("⚠️ Dataset info unavailable. Make sure the Flask backend is running.")

    # ── Tab 4: About ─────────────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">About This Project</div>',
                    unsafe_allow_html=True)

        # Hero card
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1e40af,#4f46e5);
                    border-radius:16px;padding:1.6rem 2rem;margin-bottom:1.4rem;
                    box-shadow:0 4px 20px rgba(30,64,175,0.25);'>
            <h3 style='margin:0 0 0.5rem 0;color:#ffffff;font-size:1.3rem;font-weight:800;'>
                &#x1F697; Used Car Price Prediction System
            </h3>
            <p style='margin:0;color:#bfdbfe;font-size:0.93rem;font-weight:500;line-height:1.6;'>
                <strong style="color:#ffffff;">Project:</strong> IBM BOB AI/ML Project
                &nbsp;&bull;&nbsp;
                <strong style="color:#ffffff;">Model:</strong> Random Forest Regressor
                &nbsp;&bull;&nbsp;
                <strong style="color:#ffffff;">Dataset:</strong> 1,234 records &bull; 11 Indian cities
                &nbsp;&bull;&nbsp;
                <strong style="color:#ffffff;">Stack:</strong> Python &bull; Flask &bull; Streamlit &bull; scikit-learn
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2, gap="large")
        with col_a:
            st.markdown('<div class="section-header">System Architecture</div>',
                        unsafe_allow_html=True)
            st.markdown("""
| Component | Technology |
|-----------|------------|
| ML Model  | Random Forest Regressor (scikit-learn) |
| Backend   | Flask REST API — `python app.py` |
| Frontend  | Streamlit — `streamlit run app.py` |
| Data Processing | Pandas + NumPy |
| Visualisation | Matplotlib |
| Model Storage | joblib (.pkl) |
            """)

            st.markdown('<div class="section-header">Model Performance</div>',
                        unsafe_allow_html=True)
            st.markdown("""
| Metric | Value | Interpretation |
|--------|-------|---------------|
| R² Score | **0.78** | 78% variance explained |
| MAE | **0.77 Lakh** | Avg prediction error |
| RMSE | **1.60 Lakh** | Root mean sq. error |
| CV R² (5-fold) | **0.72** | Cross-validated score |
            """)

        with col_b:
            st.markdown('<div class="section-header">API Endpoints</div>',
                        unsafe_allow_html=True)
            st.markdown("""
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/health` | Liveness check |
| `GET`  | `/api/model-info` | Metrics & options |
| `GET`  | `/api/dataset-info` | Dataset stats |
| `POST` | `/api/predict` | Single prediction |
| `POST` | `/api/batch-predict` | Batch predictions |
            """)

            st.markdown('<div class="section-header">Input Features (11 total)</div>',
                        unsafe_allow_html=True)
            st.markdown(
                " ".join(
                    f"<span class='feature-pill'>&#9881; {f}</span>"
                    for f in ["Brand", "Location", "Car Age", "KM Driven",
                               "Fuel Type", "Transmission", "Owner Type",
                               "Mileage (kmpl)", "Engine (CC)", "Power (bhp)", "Seats"]
                ),
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-header">How to Run</div>',
                        unsafe_allow_html=True)
            st.markdown("""
| Step | Command |
|------|---------|
| 1. Install | `pip install -r requirements.txt` |
| 2. Train | `python backend/train_model.py` |
| 3. Flask API | `python backend/app.py` |
| 4. Streamlit | `streamlit run backend/app.py` |
            """)

        st.markdown(
            "<p style='color:#94a3b8;font-size:0.82rem;text-align:center;"
            "margin-top:2rem;padding-top:0.9rem;border-top:1px solid #e2e8f0;'>"
            "&#x1F916; Made with IBM BOB &nbsp;&bull;&nbsp; Arpan Santra &nbsp;&bull;&nbsp; 2026</p>",
            unsafe_allow_html=True,
        )
