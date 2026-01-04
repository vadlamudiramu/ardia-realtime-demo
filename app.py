import os
import random
import datetime as dt
import requests
import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(page_title="Ardia Health Labs – Real-time Clinical Intelligence", layout="wide")

LAT = float(os.getenv("ARDIA_LAT", "33.1212"))
LON = float(os.getenv("ARDIA_LON", "-97.1839"))
ZIP = os.getenv("AIRNOW_ZIP", "76226")
AIRNOW_KEY = os.getenv("AIRNOW_API_KEY", "")

@st.cache_data(ttl=900)
def get_weather():
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}"
        "&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
        "&timezone=auto"
    )
    return requests.get(url, timeout=15).json()

@st.cache_data(ttl=900)
def get_aqi():
    if not AIRNOW_KEY:
        return None
    url = (
        "https://www.airnowapi.org/aq/observation/zipCode/current/"
        f"?format=application/json&zipCode={ZIP}&distance=25&API_KEY={AIRNOW_KEY}"
    )
    r = requests.get(url, timeout=15).json()
    return r[0] if isinstance(r, list) and r else None

weather = get_weather()
aqi = get_aqi()

def risk(score):
    return "Critical" if score > 75 else "High" if score > 50 else "Moderate" if score > 25 else "Low"

patients = []
for i in range(200):
    base = random.uniform(10, 40)
    temp = weather["current"]["temperature_2m"]
    humid = weather["current"]["relative_humidity_2m"]
    env = abs(temp - 22) * 1.2 + max(0, humid - 60) * 0.6
    score = min(100, base + env + random.uniform(0, 20))

    patients.append({
        "Patient": f"AR-{i+1:04d}",
        "Condition": random.choice(["Asthma", "COPD", "Heart Failure", "CKD"]),
        "Risk Score": round(score, 1),
        "Risk Level": risk(score),
        "Last Check-in": (dt.datetime.now() - dt.timedelta(hours=random.randint(1, 72))).strftime("%Y-%m-%d %H:%M")
    })

df = pd.DataFrame(patients).sort_values("Risk Score", ascending=False)

st.title("🧠 Ardia Health Labs – Real-time Clinical Intelligence")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Temperature °C", weather["current"]["temperature_2m"])
c2.metric("Humidity %", weather["current"]["relative_humidity_2m"])
c3.metric("Precipitation mm", weather["current"]["precipitation"])
c4.metric("AQI", aqi["AQI"] if aqi else "—")

tabs = st.tabs(["Patient View", "Provider View", "Payer View"])

with tabs[0]:
    st.subheader("Patient Experience")
    p = df.iloc[0]
    st.json(p.to_dict())
    st.markdown("- Environmental triggers detected")
    st.markdown("- Proactive care recommended")

with tabs[1]:
    st.subheader("Provider Triage Queue")
    st.dataframe(df.head(50), use_container_width=True)

with tabs[2]:
    st.subheader("Payer Risk & Cost Lens")
    high = df[df["Risk Level"].isin(["High", "Critical"])]
    st.metric("Members Monitored", len(df))
    st.metric("High / Critical Risk", len(high))
    st.metric("Estimated ER Avoidance (30d)", round(len(high) * 0.08, 1))
    st.metric("Estimated Cost Avoidance ($)", int(len(high) * 0.08 * 1500))

st.caption("Live environment data + demo-safe synthetic patients.")