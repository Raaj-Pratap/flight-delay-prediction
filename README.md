# ✈️ Flight Delay Prediction System — v2 (Upgraded)

## What's New in v2

| Feature | v1 (Old) | v2 (New) |
|---|---|---|
| Weather | Manual dropdown | **Real-time from Open-Meteo API** |
| ATC Congestion | Not included | **Simulated by airport + time of day** |
| NOTAM Alerts | Not included | **Simulated operational notices** |
| Slot / GDP | Not included | **Ground Delay Program simulation** |
| Delay breakdown | Single number | **Factor-by-factor chart** |
| Prediction | Basic probability | **Probability + estimated delay minutes** |

---

## Files You Need

```
your_project_folder/
│
├── app_v2.py                          ← Main Streamlit app (NEW)
├── realtime_data.py                   ← Weather API + ATC/NOTAM logic (NEW)
├── airport_data.py                    ← Airport coordinates & metadata (NEW)
│
├── flight_delay_model.pkl             ← Your trained ML model (same as before)
├── model_features.pkl                 ← Your model features (same as before)
├── indian_flight_delay_realistic.xlsx ← Your dataset (same as before)
│
└── (optional) Delay_Predict_Model.py  ← Re-run this to retrain model
```

---

## Setup & Run

### Step 1 — Install dependencies
```bash
pip install streamlit pandas joblib scikit-learn requests openpyxl
```

### Step 2 — Put all files in one folder
Copy `app_v2.py`, `realtime_data.py`, `airport_data.py` into the same folder
as your existing `flight_delay_model.pkl`, `model_features.pkl`, and `.xlsx` file.

### Step 3 — Run the app
```bash
streamlit run app_v2.py
```

The app opens at: http://localhost:8501

---

## How It Works (Simple Explanation)

```
User selects: Origin, Destination, Airline, Turnaround Time
        │
        ▼
┌─────────────────────────────────────────┐
│  REAL-TIME DATA FETCH                   │
│  • Open-Meteo API → live weather        │
│    (free, no API key needed)            │
│  • ATC congestion → simulated by        │
│    airport size + current hour          │
│  • NOTAM alerts → randomly simulated   │
│  • Slot / GDP → simulated by airport   │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  ML MODEL (Random Forest Classifier)    │
│  Input: weather category, airline,      │
│         route, turnaround, cancelled    │
│  Output: delay probability (0–100%)     │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  RESULT DISPLAY                         │
│  • ML probability score                 │
│  • Estimated delay (sum of factors)     │
│  • Factor breakdown bar chart           │
│  • On Time / At Risk / Delayed label    │
└─────────────────────────────────────────┘
```

---

## Weather API Details

- **Provider:** Open-Meteo (https://open-meteo.com)
- **Cost:** Free, no API key required
- **Data:** Live weather every 15 minutes
- **Fallback:** If internet is unavailable, uses seasonal simulation

---

## About ATC / NOTAM / Slot Data

Real ATC and NOTAM data requires:
- ICAO credentials (not free)
- AviationStack paid API
- DGCA India internal systems

**For this project**, these are realistically simulated:
- ATC congestion based on airport size + peak hours (07–10, 17–21 IST)
- NOTAMs randomly activated with realistic messages
- GDP (Ground Delay Program) probability based on airport traffic level

This is good enough for learning, projects, and interviews. ✅

---

## Interview Talking Points

When asked about this project, you can say:

> "I upgraded the system to pull live weather data from a public API and built a simulation layer for ATC congestion and NOTAM alerts based on realistic airport traffic patterns. The ML model takes these real-time inputs instead of manual dropdowns, so predictions reflect current conditions."

**Key concepts you can explain:**
- Open-Meteo API → REST API, JSON response parsing
- WMO weather codes → standardized meteorological coding
- Random Forest Classifier → ensemble method, predict_proba()
- SMOTE → handling class imbalance
- Streamlit caching → @st.cache_resource, @st.cache_data
- ATC Peak hours → operational domain knowledge
