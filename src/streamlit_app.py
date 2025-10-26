# src/streamlit_app.py
import json, joblib, numpy as np, pandas as pd
import streamlit as st
from pathlib import Path
from datetime import timedelta

from load_ausgrid_wide import load_ausgrid_wide
from get_weather import fetch_openmeteo_history
from features import make_features
from dispatch_battery import optimize_battery  # requires: pip install cvxpy

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "ausgrid" / "2012-2013 Solar home electricity data v2.csv"
MODEL_DIR = ROOT / "models"

st.set_page_config(page_title="Predictive Energy Management", layout="wide")

# ---------- Caches ----------
@st.cache_data
def load_hist():
    return load_ausgrid_wide(str(CSV_PATH), aggregate="sum_all")

@st.cache_resource
def load_models():
    with open(MODEL_DIR / "feature_cols.json") as f:
        feat_cols = json.load(f)["feature_cols"]
    q10 = joblib.load(MODEL_DIR / "q10.pkl")
    q50 = joblib.load(MODEL_DIR / "q50.pkl")
    q90 = joblib.load(MODEL_DIR / "q90.pkl")
    return {"q10": q10, "q50": q50, "q90": q90}, feat_cols

hist = load_hist()
models, feat_cols = load_models()

# ---------- UI ----------
st.title("Solar + EV Microgrid: Forecast → Battery Dispatch")
st.caption("Net-load forecasting with uncertainty and cost-optimized battery scheduling")

min_day = pd.to_datetime(hist["DATETIME"].min()).date()
max_day = pd.to_datetime(hist["DATETIME"].max()).date()
day = st.sidebar.date_input("Pick a day", value=max_day, min_value=min_day, max_value=max_day)
start = pd.to_datetime(day); end = start + timedelta(days=1)

# ---------- Weather for the chosen day ----------
LAT, LON = -33.86, 151.20  # Sydney (Ausgrid region)
w = fetch_openmeteo_history(
        LAT, LON,
        start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
        timezone="Australia/Sydney",
        use_cache=True, fallback=True, timeout=20
    )
st.caption(f"Weather source: {w.attrs.get('source','unknown')} (Open‑Meteo with cache/fallback)")

# ---------- Features and predictions ----------
X_all, _, df_all = make_features(hist[hist["DATETIME"] < end], w)
mask = (df_all["DATETIME"] >= start) & (df_all["DATETIME"] < end)
X_day = X_all.loc[mask, :]
ts_day = df_all.loc[mask, "DATETIME"].reset_index(drop=True)

p10 = models["q10"].predict(X_day[feat_cols])
p50 = models["q50"].predict(X_day[feat_cols])
p90 = models["q90"].predict(X_day[feat_cols])

df_pred = pd.DataFrame({"DATETIME": ts_day, "P10": p10, "P50": p50, "P90": p90})

# Actuals for comparison (since we are picking a historical day)
actual = hist[(hist["DATETIME"] >= start) & (hist["DATETIME"] < end)][["DATETIME","NET_LOAD_KW"]]
df_plot = pd.merge(df_pred, actual, on="DATETIME", how="left").rename(columns={"NET_LOAD_KW":"Actual"})

# ---------- Charts ----------
st.subheader(f"Net-Load forecast on {day}")
st.line_chart(df_plot.set_index("DATETIME")[["Actual","P50","P10","P90"]])

# Quick metrics
def rmse(a, b): 
    mask = ~np.isnan(a) & ~np.isnan(b)
    return float(np.sqrt(np.mean((a[mask] - b[mask])**2)))
def mae(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    return float(np.mean(np.abs(a[mask] - b[mask])))

col1, col2, col3 = st.columns(3)
col1.metric("RMSE (kW)", f"{rmse(df_plot['Actual'].values, df_plot['P50'].values):.2f}")
col2.metric("MAE (kW)", f"{mae(df_plot['Actual'].values, df_plot['P50'].values):.2f}")
coverage = np.mean((df_plot["Actual"].values >= df_plot["P10"].values) &
                   (df_plot["Actual"].values <= df_plot["P90"].values))
col3.metric("Coverage (P10–P90)", f"{coverage*100:.1f}%")

# ---------- Battery + Tariff ----------
st.sidebar.subheader("Battery & Tariff")
cap = st.sidebar.number_input("Battery capacity (kWh)", 10, 500, 50, step=10)
power = st.sidebar.number_input("Charge/Discharge limit (kW)", 5, 200, 20, step=5)
dem_rs = st.sidebar.number_input("Demand charge (Rs/kW)", 0, 1000, 300, step=50)

def tou_price(ts):
    hr = ts.hour
    if 18 <= hr < 23: return 12.0  # evening peak
    if 8 <= hr < 16:  return 8.0   # daytime
    return 5.0                     # off-peak

prices = np.array([tou_price(ts) for ts in df_pred["DATETIME"]])
dt_hours = 0.5

# No-battery cost and peak (using Actual if available, otherwise P50)
base_series = df_plot["Actual"].fillna(df_pred["P50"]).values
base_cost = float(np.sum(prices * dt_hours * base_series))
base_peak = float(np.max(base_series))

# Optimize battery using P50 forecast
opt = optimize_battery(df_pred["P50"].values, prices,
                       dt_hours=dt_hours, e_cap_kwh=cap, p_cap_kw=power,
                       demand_charge_rs_per_kw=dem_rs)

dispatch = pd.DataFrame({
    "DATETIME": df_pred["DATETIME"],
    "NetLoad_P50_kW": df_pred["P50"],
    "GridImport_kW": opt["grid"],
    "Charge_kW": opt["charge"],
    "Discharge_kW": opt["discharge"],
    "SOC_kWh": opt["soc"]
})

opt_cost = float(np.sum(prices * dt_hours * dispatch["GridImport_kW"].values) + dem_rs * np.max(dispatch["GridImport_kW"].values))
opt_peak = float(np.max(dispatch["GridImport_kW"].values))

c1,c2 = st.columns([2,1])
with c1:
    st.subheader("Battery dispatch and grid import")
    st.line_chart(dispatch.set_index("DATETIME")[["NetLoad_P50_kW","GridImport_kW","Charge_kW","Discharge_kW"]])
with c2:
    st.metric("Peak without battery (kW)", f"{base_peak:.1f}")
    st.metric("Peak with battery (kW)", f"{opt_peak:.1f}")
    st.metric("Peak reduction", f"{(1 - opt_peak/max(base_peak,1e-6))*100:.1f}%")
    st.metric("Cost without battery (Rs)", f"{base_cost:.0f}")
    st.metric("Cost with battery (Rs)", f"{opt_cost:.0f}")
    st.metric("Savings (Rs)", f"{base_cost - opt_cost:.0f}")
st.subheader("State of Charge")
st.line_chart(dispatch.set_index("DATETIME")[["SOC_kWh"]])

st.caption("Model: Quantile Gradient Boosting. Features: time (cyclic), lags/rolling, and ERA5 weather (radiation, temperature, etc.).")