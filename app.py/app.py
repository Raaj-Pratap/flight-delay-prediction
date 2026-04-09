import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Flight Delay Intelligence System", layout="wide")

# Load data
df = pd.read_excel("data/indian_flight_delay_realistic.xlsx")

# Generate hour column if missing
if "Departure_Hour" not in df.columns:
    df["Departure_Hour"] = np.random.randint(0,24,len(df))

# Load ML model
model = joblib.load("Models/flight_delay_model.pkl")
features = joblib.load("Models/model_features.pkl")

st.title("✈️ Indian Aviation Flight Delay Intelligence System")

st.markdown(
"""
Machine learning powered delay prediction combined with airline operational analytics.
"""
)

# ---------------- SIDEBAR ---------------- #

st.sidebar.header("Flight Parameters")

airports = ["DEL","BOM","BLR","HYD","MAA","CCU","PNQ","AMD"]

airline = st.sidebar.selectbox(
    "Airline",
    ["IndiGo","Air India","Vistara","SpiceJet","Akasa Air"]
)

origin = st.sidebar.selectbox(
    "Origin Airport",
    airports
)

destination = st.sidebar.selectbox(
    "Destination Airport",
    [a for a in airports if a != origin]
)

origin_weather = st.sidebar.selectbox(
    "Origin Weather",
    ["Clear","Fog","Rain","Storm"]
)

dest_weather = st.sidebar.selectbox(
    "Destination Weather",
    ["Clear","Fog","Rain","Storm"]
)

turnaround = st.sidebar.slider(
    "Turnaround Time (minutes)",
    20,120,40
)

cancelled = st.sidebar.selectbox(
    "Cancelled",
    [0,1]
)

diverted = st.sidebar.selectbox(
    "Diverted",
    [0,1]
)

# ---------------- PREDICTION ---------------- #

st.header("🔮 Flight Delay Prediction")

input_data = pd.DataFrame({
    "Turnaround_Time":[turnaround],
    "Cancelled":[cancelled],
    "Diverted":[diverted]
})

input_data[f"Airline_{airline}"] = 1
input_data[f"Origin_{origin}"] = 1
input_data[f"Destination_{destination}"] = 1
input_data[f"Origin_Weather_{origin_weather}"] = 1
input_data[f"Dest_Weather_{dest_weather}"] = 1

input_data = input_data.reindex(columns=features, fill_value=0)

if st.button("Predict Delay"):

    probability = model.predict_proba(input_data)[0][1]

    st.progress(float(probability))

    st.metric(
        "Delay Probability",
        str(round(probability*100,2)) + "%"
    )

    if probability > 0.5:
        st.error("⚠️ Flight likely delayed")
    else:
        st.success("✅ Flight likely on time")

# ---------------- OPERATIONAL ANALYTICS ---------------- #

st.header("📊 Airport Operational Analytics")

selected_airport = st.selectbox(
    "Select Airport",
    sorted(df["Origin"].unique())
)

airport_df = df[df["Origin"] == selected_airport]

# ---------------- KPIs ---------------- #

col1,col2,col3 = st.columns(3)

col1.metric(
    "Total Flights",
    len(airport_df)
)

col2.metric(
    "Average Delay (minutes)",
    round(airport_df["Total_Delay"].mean(),2)
)

col3.metric(
    "Maximum Delay (minutes)",
    airport_df["Total_Delay"].max()
)

# ---------------- DELAY BY AIRLINE ---------------- #

st.subheader("Average Delay by Airline")

fig1 = px.bar(
    airport_df.groupby("Airline")["Total_Delay"].mean().reset_index(),
    x="Airline",
    y="Total_Delay",
    labels={"Total_Delay":"Average Delay (minutes)"},
    title="Average Delay by Airline"
)

st.plotly_chart(fig1,use_container_width=True)

# ---------------- FLIGHT VOLUME ---------------- #

st.subheader("Flight Volume by Destination")

fig2 = px.bar(
    airport_df["Destination"].value_counts().reset_index(),
    x="Destination",
    y="count",
    labels={"count":"Number of Flights"},
    title="Flights by Destination"
)

st.plotly_chart(fig2,use_container_width=True)

# ---------------- DELAY DISTRIBUTION ---------------- #

st.subheader("Flight Delay Distribution")

delay_bins = pd.cut(
    df["Total_Delay"],
    bins=[0,15,30,60,120,300],
    labels=["0-15","15-30","30-60","60-120","120+"]
)

dist = delay_bins.value_counts().sort_index().reset_index()

fig3 = px.bar(
    dist,
    x="Total_Delay",
    y="count",
    labels={
        "Total_Delay":"Delay Range (minutes)",
        "count":"Number of Flights"
    },
    title="Distribution of Flight Delays"
)

st.plotly_chart(fig3,use_container_width=True)

# ---------------- HEATMAP ---------------- #

st.subheader("Airport vs Airline Delay Heatmap")

heatmap = df.pivot_table(
    values="Total_Delay",
    index="Origin",
    columns="Airline",
    aggfunc="mean"
)

fig4 = px.imshow(
    heatmap,
    labels=dict(color="Avg Delay (minutes)"),
    title="Average Delay by Airport and Airline"
)

st.plotly_chart(fig4,use_container_width=True)

# ---------------- DELAY TREND ---------------- #

st.subheader("Flight Delay Trend by Hour")

hour_delay = df.groupby("Departure_Hour")["Total_Delay"].mean().reset_index()

fig5 = px.line(
    hour_delay,
    x="Departure_Hour",
    y="Total_Delay",
    markers=True,
    labels={
        "Departure_Hour":"Hour of Day",
        "Total_Delay":"Average Delay (minutes)"
    },
    title="Average Delay by Departure Hour"
)

st.plotly_chart(fig5,use_container_width=True)