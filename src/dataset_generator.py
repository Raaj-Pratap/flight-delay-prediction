import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# -----------------------------
# CONFIG
# -----------------------------
NUM_AIRCRAFT = 40
FLIGHTS_PER_AIRCRAFT = 10

airlines = ['IndiGo', 'Air India', 'Vistara', 'SpiceJet', 'Akasa Air']
airports = ['DEL', 'BOM', 'BLR', 'HYD', 'MAA', 'CCU', 'PNQ', 'AMD']

high_traffic_airports = ['DEL', 'BOM']
alternate_airports = ['JAI', 'LKO', 'GOI']

# -----------------------------
# HELPERS
# -----------------------------

def flight_number(airline):
    codes = {'IndiGo':'6E','Air India':'AI','Vistara':'UK','SpiceJet':'SG','Akasa Air':'QP'}
    return codes[airline] + str(random.randint(100,9999))

def route():
    o = random.choice(airports)
    d = random.choice([x for x in airports if x != o])
    return o, d

def weather(month):
    if month in [12,1]:
        return np.random.choice(['Fog','Clear'], p=[0.6,0.4])
    elif month in [6,7,8]:
        return np.random.choice(['Rain','Clear'], p=[0.6,0.4])
    return np.random.choice(['Clear','Storm'], p=[0.9,0.1])

def delay_from_weather(w):
    if w == 'Fog': return random.randint(40,180)
    if w == 'Rain': return random.randint(10,60)
    if w == 'Storm': return random.randint(60,240)
    return 0

# -----------------------------
# DATA GENERATION
# -----------------------------

data = []

for ac in range(NUM_AIRCRAFT):
    aircraft = f"AC_{ac}"
    
    prev_arrival = datetime(2025,1,1, random.randint(4,8),0)
    prev_delay = 0
    crew_duty = 0  # track duty time

    for leg in range(FLIGHTS_PER_AIRCRAFT):
        airline = random.choice(airlines)
        fn = flight_number(airline)
        origin, dest = route()
        
        tat = random.randint(30,60)
        sched_dep = prev_arrival + timedelta(minutes=tat)

        w_origin = weather(sched_dep.month)
        w_dest = weather(sched_dep.month)

        delay = 0
        delay_weather = delay_from_weather(w_origin)

        # Reactionary delay
        reactionary = max(0, prev_delay - tat)
        
        # ATC delay
        atc = random.randint(5,30) if origin in high_traffic_airports else random.randint(0,10)
        
        # Slot delay
        slot_delay = random.randint(10,40) if origin in high_traffic_airports and random.random() < 0.5 else 0
        
        # Technical delay
        technical = random.randint(15,120) if random.random() < 0.05 else 0
        
        # Crew logic
        crew_delay = 0
        crew_duty += tat + 90  # assume avg flight time
        
        if crew_duty > 480:  # 8 hours
            crew_delay = random.randint(20,60)

        # Total delay
        delay = delay_weather + reactionary + atc + slot_delay + technical + crew_delay
        
        # Cancellation logic
        cancelled = 1 if (delay > 300 or w_origin == 'Storm' and random.random() < 0.3) else 0
        
        # Diversion logic
        diverted = 1 if (w_dest == 'Storm' and random.random() < 0.3) else 0
        diversion_airport = random.choice(alternate_airports) if diverted else None

        # Actual departure
        actual_dep = sched_dep + timedelta(minutes=delay)

        # Flight duration
        duration = random.randint(60,180)
        
        # Holding delay
        holding = random.randint(10,40) if dest in high_traffic_airports else 0
        
        actual_arr = actual_dep + timedelta(minutes=duration + holding)

        data.append([
            aircraft, airline, fn, origin, dest,
            sched_dep, actual_dep, actual_arr,
            delay, delay_weather, reactionary, atc,
            slot_delay, technical, crew_delay,
            holding, cancelled, diverted, diversion_airport,
            w_origin, w_dest, tat
        ])

        prev_arrival = actual_arr
        prev_delay = delay

# -----------------------------
# DATAFRAME
# -----------------------------

cols = [
    'Aircraft_ID','Airline','Flight_Number','Origin','Destination',
    'Scheduled_Departure','Actual_Departure','Actual_Arrival',
    'Total_Delay','Weather_Delay','Reactionary_Delay','ATC_Delay',
    'Slot_Delay','Technical_Delay','Crew_Delay',
    'Holding_Delay','Cancelled','Diverted','Diversion_Airport',
    'Origin_Weather','Dest_Weather','Turnaround_Time'
]

df = pd.DataFrame(data, columns=cols)

# -----------------------------
# SAVE TO EXCEL
# -----------------------------

file_name = "indian_flight_delay_realistic.xlsx"

df.to_excel(file_name, index=False)

print(f"Dataset saved as {file_name}")
print(df.head())