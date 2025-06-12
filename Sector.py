import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sector Momentum Strategy (Debug)", layout="wide")
st.title("🛠️ Sector Momentum Strategy — Debug Mode")

# --- Sector fund scheme codes ---
sector_funds = {
    "Nifty auto": "150645", 
    "Pharma": "150929", 
    "IT": "153320", 
    "FMCG": "149072", 
    # Add more if needed
}

# --- Utility Functions ---
@st.cache_data(ttl=86400)
def get_nav_history(scheme_code):
    try:
        url = f"https://api.mfapi.in/mf/{scheme_code}"
        r = requests.get(url)
        data = r.json()
        navs = data.get("data", [])
        df = pd.DataFrame(navs)
        df["date"] = pd.to_datetime(df["date"], dayfirst=True)
        df["nav"] = pd.to_numeric(df["nav"], errors='coerce')
        df = df.dropna().sort_values("date").reset_index(drop=True)
        return df[["date", "nav"]]
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch NAV for scheme code {scheme_code}: {e}")
        return pd.DataFrame(columns=["date", "nav"])

# --- Fetch NAV data ---
st.subheader("📥 Fetching NAV Data")
nav_data = {}
for name, code in sector_funds.items():
    df = get_nav_history(code)
    if df.empty:
        st.error(f"❌ No NAV data for {name} (Code: {code})")
    else:
        st.success(f"✅ Loaded NAV data for {name}")
    nav_data[name] = df.set_index("date")["nav"]

prices = pd.DataFrame(nav_data).dropna(how='all').fillna(method='ffill')

st.subheader("📊 Preview of Merged NAV Data")
st.write(prices.tail())

# --- Check for valid data ---
if prices.empty or prices.shape[1] < 2:
    st.error("🚫 No valid NAV data found. Cannot proceed with strategy.")
    st.stop()

# --- Optional: Print coverage of each fund
st.subheader("🕒 Date Coverage Per Sector")
coverage = {col: f"{series.first_valid_index().date()} → {series.last_valid_index().date()}" 
            for col, series in prices.items()}
st.write(coverage)
