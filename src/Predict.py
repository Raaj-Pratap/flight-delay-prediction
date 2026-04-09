import pandas as pd
import joblib

# load model
model = joblib.load("flight_delay_model.pkl")

# load feature columns
features = joblib.load("model_features.pkl")

# example flight input
input_data = {
    "Turnaround_Time": 40,
    "Cancelled": 0,
    "Diverted": 0,
    "Airline_IndiGo": 1,
    "Airline_Air India": 0,
    "Airline_Vistara": 0,
    "Airline_SpiceJet": 0,
    "Airline_Akasa Air": 0,
    "Origin_DEL": 1,
    "Origin_BOM": 0,
    "Origin_BLR": 0,
    "Destination_BOM": 1,
    "Destination_DEL": 0,
    "Destination_BLR": 0,
    "Origin_Weather_Fog": 1,
    "Origin_Weather_Clear": 0,
    "Dest_Weather_Clear": 1
}

# convert to dataframe
input_df = pd.DataFrame([input_data])

# align with training features
input_df = input_df.reindex(columns=features, fill_value=0)

# predict probability
prob = model.predict_proba(input_df)[0][1]

print("Delay Probability:", round(prob*100,2), "%")

if prob > 0.5:
    print("Prediction: Flight likely DELAYED")
else:
    print("Prediction: Flight likely ON TIME")