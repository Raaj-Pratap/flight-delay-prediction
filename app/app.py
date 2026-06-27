# app_v2.py  —  Flight Delay Prediction System (Upgraded)
# Run with:  streamlit run app_v2.py
#
# What's new vs v1:
#   ✅ Real-time weather from Open-Meteo API (free, no key)
#   ✅ ATC congestion simulation based on time of day
#   ✅ NOTAM alerts simulation
#   ✅ Slot availability / Ground Delay Program
#   ✅ Live factor breakdown — see WHY a flight will delay
#   ✅ Smart auto-fill — selecting airport auto-fetches weather
#   ✅ Clean loading states

import streamlit as st
import pandas as pd
import os
import joblib
import datetime

from airport_data import AIRPORTS, AIRLINES
from realtime_data import fetch_weather, get_atc_congestion, get_slot_availability, get_notam_alerts

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #0e1117; }
h1, h2, h3 { color: #00BFFF; }
.stMetric { background-color: #1c1f26; padding: 10px; border-radius: 10px; }
.factor-card {
    background: #1c1f26;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    border-left: 4px solid #00BFFF;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

@st.cache_resource
def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, '..', 'Models', 'flight_delay_model.pkl')
    features_path = os.path.join(BASE_DIR, '..', 'Models', 'model_features.pkl')

    model = joblib.load(model_path)
    features = joblib.load(features_path)
    if hasattr(features, 'tolist'):
        features = features.tolist()
    return model, features

@st.cache_data
def load_dataset():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, '..', 'data', 'indian_flight_delay_realistic.xlsx')
    return pd.read_excel(data_path)

try:
    model, features = load_model()
    df = load_dataset()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Could not load model files. Make sure `flight_delay_model.pkl` and `model_features.pkl` are in the same folder.\n\nError: {e}")
    st.stop()


# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────

st.title("✈️ Flight Delay Prediction System")
st.markdown(
    "Predicts **flight delay probability** using real-time weather data + "
    "ATC congestion, NOTAM alerts, and slot availability factors."
)
st.caption(f"🕐 Current IST time: {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}")

st.divider()


# ─────────────────────────────────────────────
# KPI OVERVIEW
# ─────────────────────────────────────────────

st.subheader("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Flights", f"{len(df):,}")
col2.metric("Avg Delay (min)", round(df["Total_Delay"].mean(), 1))
col3.metric("Max Delay (min)", df["Total_Delay"].max())
col4.metric("Delayed Flights", f"{(df['Total_Delay'] > 30).sum():,}")

st.divider()


# ─────────────────────────────────────────────
# SIDEBAR — FLIGHT PARAMETERS
# ─────────────────────────────────────────────

st.sidebar.header("🛫 Flight Parameters")

airport_codes = list(AIRPORTS.keys())

airline = st.sidebar.selectbox("Airline", AIRLINES)

origin = st.sidebar.selectbox(
    "Origin Airport",
    airport_codes,
    format_func=lambda x: f"{x} — {AIRPORTS[x]['city']}"
)

destination = st.sidebar.selectbox(
    "Destination Airport",
    [c for c in airport_codes if c != origin],
    format_func=lambda x: f"{x} — {AIRPORTS[x]['city']}"
)

turnaround = st.sidebar.slider(
    "Turnaround Time (minutes)",
    min_value=20,
    max_value=120,
    value=40,
    help="Time between previous arrival and this departure"
)

cancelled = st.sidebar.selectbox("Flight Cancelled?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
diverted = st.sidebar.selectbox("Flight Diverted?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

fetch_live = st.sidebar.button("🔄 Fetch Live Data & Predict", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Weather: Open-Meteo API (free)\nATC/NOTAM: Simulated (realistic)")


# ─────────────────────────────────────────────
# MAIN PREDICTION SECTION
# ─────────────────────────────────────────────

st.subheader("🔍 Delay Analysis")

if fetch_live:

    origin_info = AIRPORTS[origin]
    dest_info = AIRPORTS[destination]

    # ── Fetch all real-time data ──
    with st.spinner("Fetching live weather and operational data..."):

        origin_wx   = fetch_weather(origin_info["lat"], origin_info["lon"], origin)
        dest_wx     = fetch_weather(dest_info["lat"],   dest_info["lon"],   destination)
        origin_atc  = get_atc_congestion(origin, origin_info["congestion"])
        dest_atc    = get_atc_congestion(destination, dest_info["congestion"])
        origin_slot = get_slot_availability(origin, origin_info["congestion"])
        origin_notam= get_notam_alerts(origin)
        dest_notam  = get_notam_alerts(destination)

    # ── Factor Breakdown ──
    st.markdown("### 📋 Real-Time Operational Factors")

    col_orig, col_dest = st.columns(2)

    with col_orig:
        st.markdown(f"**🛫 Origin: {origin} — {origin_info['name']}**")

        wx_src = "🌐 Live" if origin_wx["source"] == "live" else "📊 Simulated"
        st.markdown(f"""
        <div class="factor-card">
        🌤️ <b>Weather</b> {wx_src}<br>
        Condition: <b>{origin_wx['weather_label']}</b><br>
        Wind: {origin_wx['wind_kmh']} km/h &nbsp;|&nbsp; Temp: {origin_wx['temperature_c']}°C<br>
        Delay impact: <b>+{origin_wx['weather_delay_min']} min</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="factor-card">
        🗼 <b>ATC Congestion</b><br>
        Level: <b>{origin_atc['congestion_label']}</b>
        {"(Peak Hour ⚡)" if origin_atc['is_peak_hour'] else ""}<br>
        Delay impact: <b>+{origin_atc['atc_delay_min']} min</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="factor-card">
        🅿️ <b>Slot / Ground Program</b><br>
        {origin_slot['gdp_label']}<br>
        Delay impact: <b>+{origin_slot['slot_delay_min']} min</b>
        </div>
        """, unsafe_allow_html=True)

        if origin_notam["notam_count"] > 0:
            notam_text = "<br>".join(f"• {n}" for n in origin_notam["active_notams"])
            st.markdown(f"""
            <div class="factor-card" style="border-left-color: #FF6B6B;">
            📢 <b>Active NOTAMs ({origin_notam['notam_count']})</b><br>
            {notam_text}<br>
            Delay impact: <b>+{origin_notam['notam_delay_min']} min</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="factor-card" style="border-left-color: #4CAF50;">
            📢 <b>NOTAMs</b><br>
            ✅ No active NOTAMs at origin
            </div>
            """, unsafe_allow_html=True)

    with col_dest:
        st.markdown(f"**🛬 Destination: {destination} — {dest_info['name']}**")

        wx_src_d = "🌐 Live" if dest_wx["source"] == "live" else "📊 Simulated"
        st.markdown(f"""
        <div class="factor-card">
        🌤️ <b>Weather</b> {wx_src_d}<br>
        Condition: <b>{dest_wx['weather_label']}</b><br>
        Wind: {dest_wx['wind_kmh']} km/h &nbsp;|&nbsp; Temp: {dest_wx['temperature_c']}°C<br>
        Delay impact: <b>+{dest_wx['weather_delay_min']} min</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="factor-card">
        🗼 <b>ATC Congestion</b><br>
        Level: <b>{dest_atc['congestion_label']}</b>
        {"(Peak Hour ⚡)" if dest_atc['is_peak_hour'] else ""}<br>
        Delay impact: <b>+{dest_atc['atc_delay_min']} min</b>
        </div>
        """, unsafe_allow_html=True)

        if dest_notam["notam_count"] > 0:
            notam_text_d = "<br>".join(f"• {n}" for n in dest_notam["active_notams"])
            st.markdown(f"""
            <div class="factor-card" style="border-left-color: #FF6B6B;">
            📢 <b>Active NOTAMs ({dest_notam['notam_count']})</b><br>
            {notam_text_d}<br>
            Delay impact: <b>+{dest_notam['notam_delay_min']} min</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="factor-card" style="border-left-color: #4CAF50;">
            📢 <b>NOTAMs</b><br>
            ✅ No active NOTAMs at destination
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Build input for ML model ──
    # Map weather label to model's expected categories
    def map_weather_for_model(label):
        label = label.lower()
        if "fog" in label:        return "Fog"
        if "rain" in label or "drizzle" in label or "shower" in label: return "Rain"
        if "storm" in label or "thunder" in label: return "Storm"
        return "Clear"

    origin_wx_model = map_weather_for_model(origin_wx["weather_label"])
    dest_wx_model   = map_weather_for_model(dest_wx["weather_label"])

    input_data = pd.DataFrame({
        "Turnaround_Time": [turnaround],
        "Cancelled":       [cancelled],
        "Diverted":        [diverted],
    })

    input_data[f"Airline_{airline}"]              = 1
    input_data[f"Origin_{origin}"]                = 1
    input_data[f"Destination_{destination}"]      = 1
    input_data[f"Origin_Weather_{origin_wx_model}"] = 1
    input_data[f"Dest_Weather_{dest_wx_model}"]   = 1

    input_data = input_data.reindex(columns=features, fill_value=0)

    # ── ML Prediction ──
    probability = model.predict_proba(input_data)[0][1]

    # ── Total estimated delay ──
    total_estimated = (
        origin_wx["weather_delay_min"] +
        dest_wx["weather_delay_min"] +
        origin_atc["atc_delay_min"] +
        origin_slot["slot_delay_min"] +
        origin_notam["notam_delay_min"] +
        dest_notam["notam_delay_min"]
    )

    # ── Display prediction ──
    st.markdown("### 🎯 Prediction Result")

    colA, colB, colC = st.columns(3)

    with colA:
        st.metric("ML Delay Probability", f"{round(probability * 100, 1)}%")
        st.progress(float(probability))
        if probability > 0.65:
            st.error("⚠️ Flight likely **DELAYED**")
        elif probability > 0.40:
            st.warning("⚡ Flight at **RISK of delay**")
        else:
            st.success("✅ Flight likely **ON TIME**")

    with colB:
        st.metric("Estimated Total Delay", f"{total_estimated} min")
        st.caption("Sum of all factor delays")

    with colC:
        st.markdown(f"""
        **Route:** {origin} ➜ {destination}  
        **Airline:** {airline}  
        **Weather (Origin):** {origin_wx['weather_label']}  
        **Weather (Dest):** {dest_wx['weather_label']}  
        **ATC:** {origin_atc['congestion_label']}  
        **GDP:** {'Active ⚠️' if origin_slot['gdp_active'] else 'None ✅'}
        """)

    # ── Factor bar chart ──
    st.markdown("### 📊 Delay Factor Breakdown")

    factors = {
        "Origin Weather":   origin_wx["weather_delay_min"],
        "Dest Weather":     dest_wx["weather_delay_min"],
        "Origin ATC":       origin_atc["atc_delay_min"],
        "Dest ATC":         dest_atc["atc_delay_min"],
        "Slot / GDP":       origin_slot["slot_delay_min"],
        "Origin NOTAMs":    origin_notam["notam_delay_min"],
        "Dest NOTAMs":      dest_notam["notam_delay_min"],
    }
    factors_df = pd.DataFrame({
        "Factor": list(factors.keys()),
        "Delay (min)": list(factors.values())
    }).set_index("Factor")

    st.bar_chart(factors_df)

else:
    st.info("👈 Select your flight parameters in the sidebar, then click **Fetch Live Data & Predict**")


# ─────────────────────────────────────────────
# ANALYTICS SECTION (from original app)
# ─────────────────────────────────────────────

st.divider()
st.header("📈 Airline Operations Analytics")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Average Delay by Airport")
    airport_delay = df.groupby("Origin")["Total_Delay"].mean().sort_values(ascending=False)
    st.bar_chart(airport_delay)

with col2:
    st.subheader("Delay Distribution (All Flights)")
    hist_data = df["Total_Delay"].clip(upper=300)
    st.line_chart(hist_data)

st.subheader("Airline Delay Comparison")
airline_delay = df.groupby("Airline")["Total_Delay"].mean().sort_values(ascending=False)
st.bar_chart(airline_delay)

st.subheader("Top Delay Causes")
delay_cols = ["Weather_Delay", "Reactionary_Delay", "ATC_Delay", "Slot_Delay", "Technical_Delay", "Crew_Delay"]
cause_means = df[delay_cols].mean().sort_values(ascending=False)
st.bar_chart(cause_means)
