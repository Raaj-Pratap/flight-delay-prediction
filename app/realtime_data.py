# realtime_data.py
# Fetches real weather from Open-Meteo API (free, no key needed)
# Simulates ATC congestion, NOTAM alerts, and slot availability
# Falls back gracefully if internet is unavailable

import requests
import random
import datetime


# ─────────────────────────────────────────────
# WMO Weather Code → Human Label + Delay Factor
# Ref: https://open-meteo.com/en/docs (WMO codes)
# ─────────────────────────────────────────────
WMO_CODE_MAP = {
    0:  ("Clear",  0),
    1:  ("Mostly Clear", 0),
    2:  ("Partly Cloudy", 0),
    3:  ("Overcast", 5),
    45: ("Fog", 40),
    48: ("Dense Fog", 70),
    51: ("Light Drizzle", 10),
    53: ("Moderate Drizzle", 15),
    55: ("Heavy Drizzle", 20),
    61: ("Light Rain", 15),
    63: ("Moderate Rain", 30),
    65: ("Heavy Rain", 50),
    71: ("Light Snow", 20),
    73: ("Moderate Snow", 40),
    75: ("Heavy Snow", 80),
    80: ("Rain Showers", 25),
    81: ("Heavy Showers", 45),
    82: ("Violent Showers", 90),
    95: ("Thunderstorm", 75),
    96: ("Thunderstorm with Hail", 100),
    99: ("Severe Thunderstorm", 120),
}

def get_weather_label(code):
    """Return (label, delay_minutes) for a WMO weather code."""
    return WMO_CODE_MAP.get(code, ("Unknown", 10))


def fetch_weather(lat, lon, airport_code):
    """
    Fetch current weather from Open-Meteo API.
    Returns a dict with weather label, wind speed, visibility, delay estimate.
    Falls back to simulated data if API is unavailable.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,weathercode,windspeed_10m,visibility,precipitation"
        f"&timezone=Asia/Kolkata"
    )

    try:
        response = requests.get(url, timeout=6)
        response.raise_for_status()
        data = response.json()["current"]

        wmo_code = data.get("weathercode", 0)
        weather_label, weather_delay = get_weather_label(wmo_code)

        wind_kmh = data.get("windspeed_10m", 0)
        visibility_m = data.get("visibility", 10000)
        precip_mm = data.get("precipitation", 0)
        temp_c = data.get("temperature_2m", 25)

        # Extra wind delay: crosswind > 40 km/h adds delay
        wind_delay = max(0, (wind_kmh - 40) * 0.5) if wind_kmh > 40 else 0

        total_weather_delay = round(weather_delay + wind_delay)

        return {
            "source": "live",
            "weather_label": weather_label,
            "temperature_c": temp_c,
            "wind_kmh": wind_kmh,
            "visibility_m": visibility_m,
            "precipitation_mm": precip_mm,
            "weather_delay_min": total_weather_delay,
            "wmo_code": wmo_code,
        }

    except Exception as e:
        # Fallback: simulate weather based on current month/season
        return _simulate_weather(airport_code)


def _simulate_weather(airport_code):
    """
    Fallback weather simulation when API is unavailable.
    Uses seasonal patterns for Indian airports.
    """
    month = datetime.datetime.now().month

    # Indian weather seasons
    if month in [12, 1, 2]:   # Winter — Delhi/North foggy
        if airport_code in ["DEL", "AMD"]:
            label, delay = random.choice([("Fog", 45), ("Dense Fog", 70), ("Clear", 0), ("Clear", 0)])
        else:
            label, delay = "Clear", 0
    elif month in [6, 7, 8, 9]:  # Monsoon
        label, delay = random.choice([
            ("Heavy Rain", 50), ("Moderate Rain", 30),
            ("Thunderstorm", 75), ("Clear", 0), ("Clear", 0)
        ])
    else:  # Spring/Autumn — mostly clear
        label, delay = random.choice([("Clear", 0), ("Clear", 0), ("Partly Cloudy", 0), ("Overcast", 5)])

    return {
        "source": "simulated",
        "weather_label": label,
        "temperature_c": random.randint(20, 38),
        "wind_kmh": random.randint(5, 35),
        "visibility_m": 8000 if "Fog" not in label else 200,
        "precipitation_mm": 0,
        "weather_delay_min": delay,
        "wmo_code": None,
    }


# ─────────────────────────────────────────────
# ATC CONGESTION SIMULATION
# Real NOTAM/ATC APIs require ICAO credentials.
# We simulate realistically based on airport traffic
# patterns and time of day.
# ─────────────────────────────────────────────

def get_atc_congestion(airport_code, congestion_level):
    """
    Simulate ATC delay based on airport congestion level + time of day.
    Peak hours (07:00–10:00 and 17:00–21:00 IST) have higher delays.
    Returns dict with delay estimate and congestion label.
    """
    hour = datetime.datetime.now().hour  # local IST

    # Peak hour multiplier
    is_peak = (7 <= hour <= 10) or (17 <= hour <= 21)
    peak_factor = 1.8 if is_peak else 1.0

    # Base ATC delay by congestion level
    base_ranges = {
        "high":   (10, 35),
        "medium": (5, 20),
        "low":    (0, 10),
    }

    lo, hi = base_ranges.get(congestion_level, (0, 10))
    base_delay = random.uniform(lo, hi)
    atc_delay = round(base_delay * peak_factor)

    # Congestion label
    if atc_delay > 25:
        label = "Heavy"
    elif atc_delay > 12:
        label = "Moderate"
    else:
        label = "Low"

    return {
        "atc_delay_min": atc_delay,
        "congestion_label": label,
        "is_peak_hour": is_peak,
        "peak_hours_note": "Peak: 07:00–10:00 & 17:00–21:00 IST"
    }


def get_slot_availability(airport_code, congestion_level):
    """
    Simulate slot availability / ground delay program (GDP).
    High traffic airports sometimes have slot constraints.
    Returns dict with slot delay and GDP status.
    """
    # Probability of a GDP (Ground Delay Program) being active
    gdp_prob = {"high": 0.30, "medium": 0.10, "low": 0.03}
    prob = gdp_prob.get(congestion_level, 0.05)

    gdp_active = random.random() < prob
    slot_delay = random.randint(15, 45) if gdp_active else random.randint(0, 8)

    return {
        "slot_delay_min": slot_delay,
        "gdp_active": gdp_active,
        "gdp_label": "⚠️ Ground Delay Program ACTIVE" if gdp_active else "✅ Normal Slot Operations",
    }


def get_notam_alerts(airport_code):
    """
    Simulate NOTAM (Notice to Airmen) alerts.
    Real NOTAMs require ICAO/FAA API credentials.
    We generate plausible operational NOTAMs.
    """
    possible_notams = [
        ("Runway 10/28 reduced capacity — maintenance", 15),
        ("ILS approach unavailable — visual approaches only", 20),
        ("Taxiway Alpha closed — expect longer taxi times", 10),
        ("Construction near Apron B — reduced parking stands", 8),
        ("ATIS frequency temporarily changed", 0),
        ("VOR navigation aid under maintenance", 5),
    ]

    notams = []
    delay_from_notams = 0

    for notam_text, delay in possible_notams:
        if random.random() < 0.07:   # 7% chance each NOTAM is active
            notams.append(notam_text)
            delay_from_notams += delay

    return {
        "active_notams": notams,
        "notam_delay_min": delay_from_notams,
        "notam_count": len(notams),
    }
