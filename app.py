import os
import random
import datetime as dt
from typing import Optional, Dict, Any

import requests
import pandas as pd
import numpy as np
import streamlit as st

# -------------------------------
# Page + Brand
# -------------------------------
st.set_page_config(
    page_title="Ardia Health Labs – Clinical Intelligence Demo",
    page_icon="🧠",
    layout="wide",
)

# ---- lightweight CSS polish (investor-friendly) ----
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      .ardia-hero {
        padding: 18px 18px 14px 18px;
        border-radius: 16px;
        background: linear-gradient(90deg, rgba(24,39,255,0.10), rgba(0,187,255,0.08));
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 14px;
      }
      .ardia-title { font-size: 34px; font-weight: 750; margin: 0; }
      .ardia-sub { margin: 6px 0 0 0; opacity: 0.85; font-size: 15px; }
      .kpi-card {
        padding: 14px 14px 10px 14px;
        border-radius: 14px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
      }
      .kpi-label { font-size: 12px; opacity: 0.8; margin-bottom: 6px;}
      .kpi-value { font-size: 24px; font-weight: 750; line-height: 1.0; }
      .kpi-sub { font-size: 12px; opacity: 0.75; margin-top: 6px;}
      .badge {
        display:inline-block; padding: 4px 10px; border-radius: 999px;
        font-size: 12px; font-weight: 650; margin-left: 10px;
        border: 1px solid rgba(255,255,255,0.12);
      }
      .badge-low { background: rgba(46, 204, 113, 0.12); }
      .badge-mod { background: rgba(241, 196, 15, 0.14); }
      .badge-high { background: rgba(230, 126, 34, 0.14); }
      .badge-crit { background: rgba(231, 76, 60, 0.14); }
      .panel {
        padding: 14px;
        border-radius: 14px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
      }
      .small { font-size: 12px; opacity: .75; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Inputs / Config
# -------------------------------
DEFAULT_LAT = float(os.getenv("ARDIA_LAT", "33.1212"))   # Argyle TX approx
DEFAULT_LON = float(os.getenv("ARDIA_LON", "-97.1839"))
DEFAULT_ZIP = os.getenv("AIRNOW_ZIP", "76226")
AIRNOW_KEY = os.getenv("AIRNOW_API_KEY", "")

CONDITIONS = ["Asthma", "COPD", "Heart Failure", "CKD"]

def risk_bucket(score: float) -> str:
    if score < 25: return "Low"
    if score < 50: return "Moderate"
    if score < 75: return "High"
    return "Critical"

def badge_class(level: str) -> str:
    return {
        "Low": "badge badge-low",
        "Moderate": "badge badge-mod",
        "High": "badge badge-high",
        "Critical": "badge badge-crit"
    }.get(level, "badge")

@st.cache_data(ttl=600)
def fetch_weather(lat: float, lon: float) -> Dict[str, Any]:
    # Open-Meteo: free + no key
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
        "&timezone=auto"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=600)
def fetch_aqi(zip_code: str, api_key: str) -> Optional[Dict[str, Any]]:
    if not api_key:
        return None
    url = (
        "https://www.airnowapi.org/aq/observation/zipCode/current/"
        f"?format=application/json&zipCode={zip_code}&distance=25&API_KEY={api_key}"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    return data[0] if isinstance(data, list) and data else None

def simulate_patients(n: int, weather: Dict[str, Any], aqi: Optional[int], zip_code: str) -> pd.DataFrame:
    cur = weather.get("current", {})
    temp = float(cur.get("temperature_2m", 22.0))
    rh = float(cur.get("relative_humidity_2m", 50.0))
    precip = float(cur.get("precipitation", 0.0))
    now = dt.datetime.now()

    rows = []
    for i in range(n):
        condition = random.choice(CONDITIONS)
        age = random.randint(18, 85)
        adherence = random.uniform(0.6, 1.0)
        symptom = random.uniform(0, 10)
        baseline = random.uniform(10, 45)

        env = 0.0
        if condition in ("Asthma", "COPD"):
            env += max(0, (rh - 55) * 0.4)
            env += max(0, abs(temp - 22) * 0.9)
            if aqi is not None:
                env += max(0, (aqi - 50) * 0.25)
        elif condition == "Heart Failure":
            env += max(0, precip * 6)
        else:  # CKD
            env += max(0, abs(temp - 20) * 0.3)

        score = baseline + env + (1.0 - adherence) * 30 + symptom * 3
        score = max(0, min(100, score))

        rows.append({
            "patient_id": f"AR-{i+1:04d}",
            "age": age,
            "condition": condition,
            "risk_score": round(score, 1),
            "risk_level": risk_bucket(score),
            "adherence": round(adherence, 2),
            "symptom_index": round(symptom, 1),
            "last_checkin": (now - dt.timedelta(hours=random.randint(1, 72))).strftime("%Y-%m-%d %H:%M"),
            "zip": zip_code,
        })

    df = pd.DataFrame(rows).sort_values("risk_score", ascending=False).reset_index(drop=True)
    return df

def reasoned_summary(p: Dict[str, Any], temp: float, rh: float, precip: float, wind: float, aqi: Optional[int]) -> Dict[str, Any]:
    """
    Investor-friendly explanation (not raw JSON).
    """
    reasons = []
    actions = []
    condition = p["condition"]
    score = p["risk_score"]

    if score >= 75:
        actions += ["Escalate: care-team review within 2 hours", "Verify meds + refill status", "Assess symptom escalation + triggers"]
    elif score >= 50:
        actions += ["Outreach: within 24 hours", "Review symptoms + adherence trend", "Adjust care plan per protocol if needed"]
    else:
        actions += ["Continue monitoring", "Send weekly summary to provider"]

    # Context-based reasoning
    if condition in ("Asthma", "COPD"):
        if aqi is not None and aqi >= 101:
            reasons.append("Air quality is elevated (AQI) → higher respiratory trigger risk")
            actions.append("Recommend limiting outdoor exposure + reinforce rescue plan")
        if rh >= 70:
            reasons.append("High humidity → can worsen respiratory symptoms")
        if temp <= 5 or temp >= 32:
            reasons.append("Temperature extremes → known trigger for respiratory exacerbations")
    if condition == "Heart Failure" and precip > 0:
        reasons.append("Weather instability (precipitation) → proxy for stressors that may affect symptoms")
    if p["adherence"] < 0.8:
        reasons.append("Medication adherence appears suboptimal → increased exacerbation risk")
    if p["symptom_index"] >= 7:
        reasons.append("Symptom index trending high → needs proactive follow-up")

    if not reasons:
        reasons.append("No major risk drivers detected; continue routine monitoring")

    return {
        "clinical_story": f"{condition} member with {p['risk_level']} risk (score {score}).",
        "top_reasons": reasons[:4],
        "recommended_actions": actions[:5],
        "context_used": {
            "temperature_c": temp,
            "humidity_pct": rh,
            "precip_mm": precip,
            "wind_kmh": wind,
            "aqi": aqi if aqi is not None else "—"
        }
    }

# -------------------------------
# Sidebar controls
# -------------------------------
with st.sidebar:
    st.header("Demo Controls")
    lat = st.number_input("Latitude", value=DEFAULT_LAT, format="%.6f")
    lon = st.number_input("Longitude", value=DEFAULT_LON, format="%.6f")
    zip_code = st.text_input("ZIP (AQI optional)", value=DEFAULT_ZIP)
    n_patients = st.slider("Cohort size (synthetic)", 50, 600, 250, step=25)
    st.caption("Tip: add AIRNOW_API_KEY in Streamlit Secrets to show live AQI.")
    if st.button("Refresh now"):
        st.cache_data.clear()

# -------------------------------
# Fetch live signals
# -------------------------------
weather = fetch_weather(lat, lon)
cur = weather.get("current", {})
temp = float(cur.get("temperature_2m", 0.0))
rh = float(cur.get("relative_humidity_2m", 0.0))
precip = float(cur.get("precipitation", 0.0))
wind = float(cur.get("wind_speed_10m", 0.0))

aqi_obs = fetch_aqi(zip_code, AIRNOW_KEY)
aqi_val = int(aqi_obs.get("AQI")) if aqi_obs and aqi_obs.get("AQI") is not None else None

df = simulate_patients(n_patients, weather, aqi_val, zip_code)

# -------------------------------
# Hero
# -------------------------------
st.markdown(
    f"""
    <div class="ardia-hero">
      <div class="ardia-title">🧠 Ardia Health Labs – Real-time Clinical Intelligence</div>
      <div class="ardia-sub">
        Not alerts. <b>Reasoned</b> next-best actions using live context (weather/AQI) + patient history signals (demo-safe cohort).
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# KPI Row (nice cards)
# -------------------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Temperature</div><div class="kpi-value">{temp:.1f}°C</div><div class="kpi-sub">Live signal</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Humidity</div><div class="kpi-value">{rh:.0f}%</div><div class="kpi-sub">Live signal</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Precipitation</div><div class="kpi-value">{precip:.1f} mm</div><div class="kpi-sub">Live signal</div></div>""", unsafe_allow_html=True)
with k4:
    aqi_display = str(aqi_val) if aqi_val is not None else "—"
    st.markdown(f"""<div class="kpi-card"><div class="kpi-label">AQI</div><div class="kpi-value">{aqi_display}</div><div class="kpi-sub">Live if key set</div></div>""", unsafe_allow_html=True)

# -------------------------------
# Tabs
# -------------------------------
tab1, tab2, tab3 = st.tabs(["👤 Patient", "🩺 Provider", "🏦 Payer"])

# ===== Patient =====
with tab1:
    st.subheader("Patient Experience (Explainable)")
    left, right = st.columns([1.1, 1.4])

    with left:
        pid = st.selectbox("Select a member", df["patient_id"].tolist(), index=0)
        p = df[df["patient_id"] == pid].iloc[0].to_dict()
        badge = badge_class(p["risk_level"])
        st.markdown(
            f"""
            <div class="panel">
              <div style="font-size:16px; font-weight:750;">
                {p["patient_id"]}
                <span class="{badge}">{p["risk_level"]}</span>
              </div>
              <div class="small" style="margin-top:6px;">Condition: <b>{p["condition"]}</b> • Risk score: <b>{p["risk_score"]}</b></div>
              <div class="small">Adherence: <b>{p["adherence"]}</b> • Symptom index: <b>{p["symptom_index"]}</b></div>
              <div class="small">Last check-in: {p["last_checkin"]} • ZIP: {p["zip"]}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        story = reasoned_summary(p, temp, rh, precip, wind, aqi_val)
        st.markdown("""<div class="panel">""", unsafe_allow_html=True)
        st.markdown(f"### Clinical Story\n{story['clinical_story']}")
        st.markdown("**Top reasons (explainable):**")
        for r in story["top_reasons"]:
            st.write(f"• {r}")
        st.markdown("**Recommended next actions:**")
        for a in story["recommended_actions"]:
            st.write(f"• {a}")
        st.markdown("**Context used (live):**")
        st.json(story["context_used"])
        st.markdown("""</div>""", unsafe_allow_html=True)

    # Add a small trend chart for investor feel
    st.markdown("#### 14-day risk trend (demo-safe)")
    days = pd.date_range(end=pd.Timestamp.now(), periods=14, freq="D")
    base = float(p["risk_score"])
    trend = np.clip(np.random.normal(loc=base, scale=8, size=len(days)), 0, 100)
    st.line_chart(pd.DataFrame({"risk_score": trend}, index=days))

# ===== Provider =====
with tab2:
    st.subheader("Provider Triage Queue (reduces alert fatigue)")
    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        condition_filter = st.multiselect("Condition", CONDITIONS, default=CONDITIONS)
    with c2:
        risk_filter = st.multiselect("Risk level", ["Critical", "High", "Moderate", "Low"], default=["Critical","High","Moderate","Low"])
    with c3:
        show_n = st.slider("Rows", 20, 200, 60, step=10)

    view = df[df["condition"].isin(condition_filter) & df["risk_level"].isin(risk_filter)].head(show_n).copy()

    # Add a provider-friendly "why" column
    def why(row):
        reasons = []
        if row["risk_level"] in ("High","Critical"): reasons.append("elevated risk")
        if row["adherence"] < 0.8: reasons.append("low adherence")
        if row["symptom_index"] >= 7: reasons.append("symptoms rising")
        return ", ".join(reasons) if reasons else "stable"
    view["why_now"] = view.apply(why, axis=1)

    st.dataframe(
        view[["patient_id","condition","risk_level","risk_score","why_now","last_checkin"]],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("#### Risk distribution (provider panel)")
    dist = df["risk_level"].value_counts().reindex(["Critical","High","Moderate","Low"]).fillna(0)
    st.bar_chart(dist)

# ===== Payer =====
with tab3:
    st.subheader("Payer View (Outcomes + Cost Lens)")

    total = len(df)
    highcrit = int(df["risk_level"].isin(["High","Critical"]).sum())
    moderate = int((df["risk_level"]=="Moderate").sum())
    low = total - highcrit - moderate

    # Simple demo economics (replace with claims in pilots)
    avoided_er_30d = round(highcrit * 0.08, 1)
    cost_per_er = 1500
    cost_avoid = int(avoided_er_30d * cost_per_er)

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Members</div><div class="kpi-value">{total}</div><div class="kpi-sub">Monitored cohort</div></div>""", unsafe_allow_html=True)
    with p2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">High / Critical</div><div class="kpi-value">{highcrit}</div><div class="kpi-sub">Intervention candidates</div></div>""", unsafe_allow_html=True)
    with p3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Avoidable ER (30d)</div><div class="kpi-value">{avoided_er_30d}</div><div class="kpi-sub">Demo estimate</div></div>""", unsafe_allow_html=True)
    with p4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Cost avoidance (30d)</div><div class="kpi-value">${cost_avoid:,}</div><div class="kpi-sub">Demo estimate</div></div>""", unsafe_allow_html=True)

    st.markdown("#### Risk distribution (payer lens)")
    dist = pd.DataFrame({
        "risk_level": ["Critical","High","Moderate","Low"],
        "count": [int((df["risk_level"]=="Critical").sum()),
                  int((df["risk_level"]=="High").sum()),
                  int((df["risk_level"]=="Moderate").sum()),
                  int((df["risk_level"]=="Low").sum())]
    }).set_index("risk_level")
    st.bar_chart(dist)

    st.info(
        "Investor note: This demo uses live context (weather/AQI) + synthetic members (no PHI). "
        "In pilots, we plug in real FHIR + claims to prove reduced ER visits/readmissions."
    )

st.caption("Demo-safe cohort. Live weather via Open-Meteo. AQI becomes live when AIRNOW_API_KEY is set in Streamlit Secrets.")
