# airport_data.py
# All airport info in one place — coordinates, names, congestion levels

AIRPORTS = {
    "DEL": {
        "name": "Indira Gandhi Intl, Delhi",
        "lat": 28.5665,
        "lon": 77.1031,
        "congestion": "high",      # used for ATC/slot delay simulation
        "city": "Delhi"
    },
    "BOM": {
        "name": "Chhatrapati Shivaji Intl, Mumbai",
        "lat": 19.0896,
        "lon": 72.8656,
        "congestion": "high",
        "city": "Mumbai"
    },
    "BLR": {
        "name": "Kempegowda Intl, Bangalore",
        "lat": 13.1986,
        "lon": 77.7066,
        "congestion": "medium",
        "city": "Bangalore"
    },
    "HYD": {
        "name": "Rajiv Gandhi Intl, Hyderabad",
        "lat": 17.2403,
        "lon": 78.4294,
        "congestion": "medium",
        "city": "Hyderabad"
    },
    "MAA": {
        "name": "Chennai Intl Airport",
        "lat": 12.9941,
        "lon": 80.1709,
        "congestion": "medium",
        "city": "Chennai"
    },
    "CCU": {
        "name": "Netaji Subhas Chandra Bose Intl, Kolkata",
        "lat": 22.6547,
        "lon": 88.4467,
        "congestion": "low",
        "city": "Kolkata"
    },
    "PNQ": {
        "name": "Pune Airport",
        "lat": 18.5822,
        "lon": 73.9197,
        "congestion": "low",
        "city": "Pune"
    },
    "AMD": {
        "name": "Sardar Vallabhbhai Patel Intl, Ahmedabad",
        "lat": 23.0772,
        "lon": 72.6347,
        "congestion": "low",
        "city": "Ahmedabad"
    },
}

AIRLINES = ["IndiGo", "Air India", "Vistara", "SpiceJet", "Akasa Air"]
