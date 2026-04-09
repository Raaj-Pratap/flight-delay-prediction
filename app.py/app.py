import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Airline Flight Delay Analytics",
    page_icon="✈️",
    layout="wide"
)

# Load data
df = pd.read_excel("data/indian_flight_delay_realistic.xlsx")

model = joblib.load("Models/flight_delay_model.pkl")
features = joblib.load("Models/model_features.pkl")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["✈️ Delay Prediction","📊 Operations Analytics"]
)

# ================================
# PAGE 1 : PREDICTION SYSTEM
# ================================

if page == "✈️ Delay Prediction":

    st.title("✈️ Flight Delay Prediction System")

    st.write("Predict whether a flight will be delayed (>30 minutes)")

    st.sidebar.header("Flight Parameters")

    airline = st.sidebar.selectbox(
        "Airline",
        ["IndiGo","Air India","Vistara","SpiceJet","Akasa Air"]
    )

    origin = st.sidebar.selectbox(
        "Origin Airport",
        ["DEL","BOM","BLR","HYD","MAA","CCU","PNQ","AMD"]
    )

    destination = st.sidebar.selectbox(
        "Destination Airport",
        ["DEL","BOM","BLR","HYD","MAA","CCU","PNQ","AMD"]
    )

    if origin == destination:
        st.sidebar.error("Origin and Destination airports cannot be the same.")
        st.stop()   

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
        "Flight Cancelled",
        [0,1]
    )

    diverted = st.sidebar.selectbox(
        "Flight Diverted",
        [0,1]
    )

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

        col1,col2 = st.columns(2)

        with col1:

            st.metric(
                "Delay Probability",
                str(round(probability*100,2)) + "%"
            )

            st.progress(float(probability))

            if probability > 0.5:
                st.error("⚠️ Flight likely DELAYED")
            else:
                st.success("✅ Flight likely ON TIME")

        with col2:

            st.info(f"""
            Airline: {airline}

            Route: {origin} ➜ {destination}

            Turnaround Time: {turnaround} minutes
            """)

# ================================
# PAGE 2 : ANALYTICS DASHBOARD
# ================================

if page == "📊 Operations Analytics":

    st.title("📊 Airline Operations Analytics Dashboard")

    st.write("Exploratory Data Analysis of Indian Airline Operations")

    # KPIs
    col1,col2,col3 = st.columns(3)

    col1.metric(
        "Total Flights",
        len(df)
    )

    col2.metric(
        "Average Delay",
        round(df["Total_Delay"].mean(),2)
    )

    col3.metric(
        "Maximum Delay",
        df["Total_Delay"].max()
    )

    st.divider()

    # Airport Delay
    st.subheader("Average Delay by Airport")

    airport_delay = df.groupby("Origin")["Total_Delay"].mean()

    st.bar_chart(airport_delay)

    # Airline Delay
    st.subheader("Average Delay by Airline")

    airline_delay = df.groupby("Airline")["Total_Delay"].mean()

    st.bar_chart(airline_delay)

    # Delay Distribution
    st.subheader("Delay Distribution")

    st.line_chart(df["Total_Delay"])

    # Delay Causes
    st.subheader("Operational Delay Causes")

    delay_causes = df[[
        "Weather_Delay",
        "ATC_Delay",
        "Technical_Delay",
        "Crew_Delay",
        "Reactionary_Delay"
    ]].mean()

    st.bar_chart(delay_causes)