# ✈️ Flight Delay Prediction & Aviation Analytics

Live App:
https://your-app-name.streamlit.app

Note: If the app is sleeping it may take ~20 seconds to wake up.

# ✈️ Indian Aviation Flight Delay Intelligence System

## Overview
This project builds a machine learning system to predict flight delays in the Indian aviation market and provides an operational analytics dashboard for airport performance monitoring.

The application combines predictive modeling with interactive visual analytics to help understand delay patterns across airlines, airports, and time periods.

---

## Features

### Machine Learning Prediction
- Predicts probability of flight delay (>30 minutes)
- Uses operational parameters like:
  - Airline
  - Origin and destination airport
  - Weather conditions
  - Turnaround time
  - Flight diversion/cancellation

### Operational Analytics Dashboard
- Airport performance KPIs
- Average delay by airline
- Flight volume by destination
- Delay distribution analysis
- Airport vs Airline delay heatmap
- Flight delay trend by hour of day

---

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- NumPy

---

## Project Structure

```
app.py
flight_delay_model.pkl
model_features.pkl
indian_flight_delay_realistic.xlsx
requirements.txt
README.md
```

---

## How to Run Locally

Install dependencies

```
pip install -r requirements.txt
```

Run the Streamlit app

```
streamlit run app.py
```

---

## Future Improvements

- Real aviation data integration
- Weather API integration
- Route delay prediction
- Airline performance benchmarking

---

## Author

Raj Singh
