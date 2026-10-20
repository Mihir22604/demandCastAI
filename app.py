"""
DemandCast AI - Daily Logistics Demand Planner (Streamlit Web Dashboard)
================================================================================
Single unified frontend application for daily demand forecasting.
Provides real-time capacity tier forecasting, confidence metrics, multi-class
probability distributions, and actionable warehouse staffing directives.

Connects to the FastAPI REST API (port 8000) with automatic zero-downtime
local fallback to the serialized champion model.

Author: DemandCast AI Core Architecture Team
Assessment Phase: Stage 14 - Web UI
Run:
    streamlit run app.py
"""

import os
import sys
import time
import json
import math
import warnings
import requests
import numpy as np
import pandas as pd
import streamlit as st

# Suppress sklearn unpickling warnings and multiprocessing probe warnings
warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "2"

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="DemandCast AI - Logistics Demand Planner",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# PATH CONSTANTS & FALLBACK LOCAL MODEL
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
API_DEFAULT_URL = os.environ.get("DEMANDCAST_API_URL", "http://127.0.0.1:8000/predict")
HEALTH_DEFAULT_URL = os.environ.get("DEMANDCAST_HEALTH_URL", "http://127.0.0.1:8000/health")

FEATURE_COLUMNS = [
    "banking_orders_2",
    "banking_orders_1",
    "day_cos",
    "day_of_week",
    "banking_ratio_1_2",
    "traffic_controller_orders",
    "week_of_month",
    "target_lag5"
]

TIER_CONFIG = {
    0: {
        "label": "Low Demand (<240)",
        "badge_color": "#2563EB",  # Blue
        "bg_color": "#EFF6FF",
        "demand_range": "Below 240.0 thousand units",
        "action": "Ground surplus haulage tractors, reassign staff to cross-dock maintenance, and avoid excess third-party contractor hire.",
        "icon": "📉"
    },
    1: {
        "label": "Normal Demand (240-340)",
        "badge_color": "#059669",  # Emerald Green
        "bg_color": "#ECFDF5",
        "demand_range": "240.0 to 340.0 thousand units",
        "action": "Maintain baseline warehouse staffing, operate standard shift schedules, and sustain routine line-haul dispatch.",
        "icon": "✅"
    },
    2: {
        "label": "Surge Demand (>340)",
        "badge_color": "#DC2626",  # Crimson Red
        "bg_color": "#FEF2F2",
        "demand_range": "Exceeds 340.0 thousand units",
        "action": "Trigger warehouse overtime shifts, pre-stage dock bay doors, and dispatch reserve 3PL line-haul trailers immediately.",
        "icon": "🚨"
    }
}


# ------------------------------------------------------------------------------
# CACHED LOCAL MODEL LOADER (RELIABLE ZERO-CRASH FALLBACK)
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_local_model():
    """Loads champion model directly into memory as a zero-downtime fallback."""
    if os.path.exists(MODEL_PATH):
        try:
            import joblib
            return joblib.load(MODEL_PATH)
        except Exception:
            return None
    return None


local_champion_model = load_local_model()


# ------------------------------------------------------------------------------
# INFERENCE DISPATCH FUNCTION (API WITH LOCAL FALLBACK)
# ------------------------------------------------------------------------------
def execute_forecast(payload: dict, api_url: str):
    """
    Submits payload to the FastAPI REST API. If the API is unreachable,
    gracefully executes the cached local champion model without disrupting the user.
    """
    t_start = time.perf_counter()
    used_mode = "REST API (FastAPI)"

    # Attempt REST API call
    try:
        response = requests.post(api_url, json=payload, timeout=3.0)
        if response.status_code == 200:
            result = response.json()
            latency = round((time.perf_counter() - t_start) * 1000.0, 2)
            return {
                "success": True,
                "tier_id": result["predicted_capacity_tier"],
                "tier_label": result["tier_label"],
                "confidence": result["confidence_score"],
                "probabilities": result["class_probabilities"],
                "action": result["operational_dispatch_action"],
                "range": result["estimated_demand_range"],
                "latency_ms": latency,
                "engine": used_mode
            }
    except Exception:
        pass

    # Seamless Fallback to local model
    if local_champion_model is not None:
        used_mode = "Direct Embedded Model (Fallback)"
        # Derive features
        day_cos = float(math.cos(2.0 * math.pi * (payload["day_of_week"] - 2) / 5.0))
        banking_ratio = float(payload["banking_orders_1"] / (payload["banking_orders_2"] + 1.0))

        feat_df = pd.DataFrame([{
            "banking_orders_2": float(payload["banking_orders_2"]),
            "banking_orders_1": float(payload["banking_orders_1"]),
            "day_cos": day_cos,
            "day_of_week": int(payload["day_of_week"]),
            "banking_ratio_1_2": banking_ratio,
            "traffic_controller_orders": float(payload["traffic_controller_orders"]),
            "week_of_month": int(payload["week_of_month"]),
            "target_lag5": float(payload["target_lag5"]),
        }])[FEATURE_COLUMNS]

        pred_tier = int(local_champion_model.predict(feat_df)[0])
        try:
            probs = local_champion_model.predict_proba(feat_df)[0]
            prob_dict = {
                TIER_CONFIG[0]["label"]: round(float(probs[0]), 4),
                TIER_CONFIG[1]["label"]: round(float(probs[1]), 4),
                TIER_CONFIG[2]["label"]: round(float(probs[2]), 4),
            }
            conf = round(float(np.max(probs)), 4)
        except Exception:
            prob_dict = {TIER_CONFIG[pred_tier]["label"]: 1.0}
            conf = 1.0

        latency = round((time.perf_counter() - t_start) * 1000.0, 2)
        tier_meta = TIER_CONFIG[pred_tier]

        return {
            "success": True,
            "tier_id": pred_tier,
            "tier_label": tier_meta["label"],
            "confidence": conf,
            "probabilities": prob_dict,
            "action": tier_meta["action"],
            "range": tier_meta["demand_range"],
            "latency_ms": latency,
            "engine": used_mode
        }

    return {
        "success": False,
        "error": "Neither the REST API nor the local serialized model could be accessed."
    }


# ------------------------------------------------------------------------------
# SIDEBAR CONTROLS & PRESETS
# ------------------------------------------------------------------------------
with st.sidebar:
    st.title("🚚 DemandCast AI")
    st.markdown("**Daily Logistics Capacity Planner**")
    st.caption("Version 1.0.0 | LightGBM Champion")
    st.divider()

    st.subheader("🌐 Service Connection")
    api_endpoint = st.text_input("FastAPI Endpoint URL", value=API_DEFAULT_URL)

    # Check API health
    api_online = False
    try:
        r = requests.get(HEALTH_DEFAULT_URL, timeout=1.0)
        if r.status_code == 200:
            api_online = True
    except Exception:
        api_online = False

    if api_online:
        st.success("● REST API Connected (Port 8000)")
    else:
        st.warning("○ REST API Offline (Using Local Fallback)")

    st.divider()
    st.markdown(
        """
        <small style='color: #6B7280;'>
        <b>Capacity Thresholds:</b><br>
        • Low Demand: &lt; 240k orders<br>
        • Normal Demand: 240k – 340k orders<br>
        • Surge Demand: &gt; 340k orders
        </small>
        """,
        unsafe_allow_html=True
    )

# Sensible default input values
def_b2 = 146135.0
def_b1 = 71115.0
def_traffic = 31273.0
def_day = 2  # Monday
def_week = 3
def_lag5 = 295.0


# ------------------------------------------------------------------------------
# MAIN APPLICATION INTERFACE
# ------------------------------------------------------------------------------
st.markdown("## 📊 Logistics Demand Operational Capacity Forecaster")
st.markdown(
    "Submit daily exogenous indicators below to forecast tomorrow's operational capacity tier, "
    "view confidence probabilities, and receive actionable trailer dispatch instructions."
)

st.write("")

# Simple, clean 2-column input card
with st.container():
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 🏢 Commercial Banking & Freight Traffic")
        input_b2 = st.number_input(
            "Primary Banking Orders (Channel 2)",
            min_value=0.0,
            max_value=500000.0,
            value=float(def_b2),
            step=5000.0,
            help="Primary financial clearance volume from Commercial Banking Channel 2."
        )
        input_b1 = st.number_input(
            "Secondary Banking Orders (Channel 1)",
            min_value=0.0,
            max_value=300000.0,
            value=float(def_b1),
            step=5000.0,
            help="Corporate purchasing activity from Commercial Banking Channel 1."
        )
        input_traffic = st.number_input(
            "Traffic Controller Freight Volume",
            min_value=0.0,
            max_value=200000.0,
            value=float(def_traffic),
            step=2000.0,
            help="Municipal freight transport traffic controller volume."
        )

    with col2:
        st.markdown("#### 📅 Calendar & Historical Demand Momentum")
        day_options = {
            2: "Monday (Day 2)",
            3: "Tuesday (Day 3)",
            4: "Wednesday (Day 4)",
            5: "Thursday (Day 5)",
            6: "Friday (Day 6)"
        }
        input_day = st.selectbox(
            "Day of the Business Week",
            options=list(day_options.keys()),
            format_func=lambda x: day_options[x],
            index=list(day_options.keys()).index(def_day),
            help="Operational business days (weekends are non-operational)."
        )
        input_week = st.slider(
            "Week of the Month",
            min_value=1,
            max_value=5,
            value=int(def_week),
            help="Sequential week within the calendar month (1 to 5)."
        )
        input_lag5 = st.number_input(
            "Same Day Prior Week Total Orders (Lag-5)",
            min_value=50.0,
            max_value=1000.0,
            value=float(def_lag5),
            step=10.0,
            help="Historical total orders on the same day last week (in thousands)."
        )

st.write("")
predict_clicked = st.button("🚀 Forecast Operational Capacity", type="primary", use_container_width=True)

# ------------------------------------------------------------------------------
# RESULTS PRESENTATION
# ------------------------------------------------------------------------------
if predict_clicked:
    payload = {
        "banking_orders_2": float(input_b2),
        "banking_orders_1": float(input_b1),
        "traffic_controller_orders": float(input_traffic),
        "day_of_week": int(input_day),
        "week_of_month": int(input_week),
        "target_lag5": float(input_lag5)
    }

    with st.spinner("Generating logistics demand forecast..."):
        result = execute_forecast(payload, api_endpoint)

    if result.get("success"):
        tier_id = result["tier_id"]
        tier_meta = TIER_CONFIG[tier_id]

        st.divider()
        st.markdown("### 🎯 Forecast Results & Operational Guidance")

        # Top 3 KPI Cards
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric(
                label="Predicted Operational Tier",
                value=f"{tier_meta['icon']} {tier_meta['label']}"
            )
        with kpi2:
            st.metric(
                label="Confidence Score",
                value=f"{result['confidence'] * 100:.1f}%"
            )
        with kpi3:
            st.metric(
                label="Inference Latency",
                value=f"{result['latency_ms']} ms",
                delta=f"Engine: {result['engine']}",
                delta_color="off"
            )

        st.write("")

        # Actionable Logistics Guidance Box
        st.markdown(
            f"""
            <div style="background-color: {tier_meta['bg_color']}; border-left: 6px solid {tier_meta['badge_color']}; padding: 18px 24px; border-radius: 8px; margin-bottom: 20px;">
                <h4 style="margin: 0 0 8px 0; color: {tier_meta['badge_color']};">
                    {tier_meta['icon']} Dispatch Directive: {tier_meta['label']}
                </h4>
                <p style="margin: 0 0 8px 0; font-size: 15px; color: #1F2937;">
                    <b>Estimated Demand Volume:</b> {result['range']}
                </p>
                <p style="margin: 0; font-size: 15px; color: #374151;">
                    <b>Warehouse & Freight Action:</b> {result['action']}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Multi-class probability distribution
        st.markdown("#### 📈 Multi-Class Probability Distribution")
        probs = result["probabilities"]

        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            st.markdown(f"**Low Demand (&lt;240)**: `{probs.get(TIER_CONFIG[0]['label'], 0.0) * 100:.1f}%`")
            st.progress(float(probs.get(TIER_CONFIG[0]["label"], 0.0)))
        with p_col2:
            st.markdown(f"**Normal Demand (240-340)**: `{probs.get(TIER_CONFIG[1]['label'], 0.0) * 100:.1f}%`")
            st.progress(float(probs.get(TIER_CONFIG[1]["label"], 0.0)))
        with p_col3:
            st.markdown(f"**Surge Demand (&gt;340)**: `{probs.get(TIER_CONFIG[2]['label'], 0.0) * 100:.1f}%`")
            st.progress(float(probs.get(TIER_CONFIG[2]["label"], 0.0)))

    else:
        st.error(f"Prediction failed: {result.get('error', 'Unknown error')}")

# ------------------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------------------
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #9CA3AF; font-size: 13px;">
        DemandCast AI Enterprise System • End-to-End Audited Machine Learning Pipeline • Powered by FastAPI & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
