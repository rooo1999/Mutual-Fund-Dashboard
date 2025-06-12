import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sector Momentum Strategy", layout="wide")
st.title("📈 Sector Momentum Strategy Dashboard")

# --- Sector fund scheme codes ---
sector_funds = {
    "Nifty auto": "150645", 
    "Pharma": "150929", 
    "IT": "153320", 
    "FMCG": "149072"
    # Add more sector funds here if needed
}

# --- Utility Function ---
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
    except:
        return pd.DataFrame(columns=["date", "nav"])

# --- Fetch NAV data ---
nav_data = {}
for name, code in sector_funds.items():
    df = get_nav_history(code)
    nav_data[name] = df.set_index("date")["nav"]

prices = pd.DataFrame(nav_data).dropna(how='all').fillna(method='ffill')

if prices.empty or prices.shape[1] < 2:
    st.error("🚫 Not enough NAV data to proceed.")
    st.stop()

# --- Momentum Strategy ---
months = pd.date_range(prices.index[0], prices.index[-1], freq='M')
port_nav = []
port_dates = []
benchmark_code = "118550"  # Nifty 50 index fund
benchmark_df = get_nav_history(benchmark_code).set_index("date")
benchmark = benchmark_df["nav"].reindex(prices.index).fillna(method='ffill')

port_prices = []
top_picks = []

for month in months:
    month_str = month.strftime("%b-%Y")
    if month not in prices.index:
        continue

    past_21 = month - pd.DateOffset(days=21)
    past_42 = month - pd.DateOffset(days=42)
    if past_21 not in prices.index or past_42 not in prices.index:
        continue

    r1 = prices.loc[month] / prices.loc[past_21] - 1
    r2 = prices.loc[month] / prices.loc[past_42] - 1
    score = 0.6 * r1 + 0.4 * r2

    sma20 = prices.rolling(window=20).mean()
    sma50 = prices.rolling(window=50).mean()
    filter_mask = sma20.loc[month] > sma50.loc[month]
    score[~filter_mask] = -np.inf

    top_sectors = score.nlargest(2).index.tolist()
    top_scores = score.loc[top_sectors].values.tolist()

    top_picks.append({
        "Month": month_str,
        "Sector 1": top_sectors[0] if len(top_sectors) > 0 else None,
        "Score 1": top_scores[0] if len(top_scores) > 0 else None,
        "Sector 2": top_sectors[1] if len(top_sectors) > 1 else None,
        "Score 2": top_scores[1] if len(top_scores) > 1 else None
    })

    port_prices.append(prices.loc[month, top_sectors].mean())
    port_dates.append(month)

# --- Portfolio NAV ---
portfolio_nav = pd.Series(port_prices, index=port_dates).dropna()
portfolio_nav = portfolio_nav / portfolio_nav.iloc[0] * 100
benchmark = benchmark.reindex(portfolio_nav.index).dropna()
benchmark = benchmark / benchmark.iloc[0] * 100

# --- Plot NAV chart ---
st.subheader("📈 Strategy vs Nifty 50 NAV")
fig, ax = plt.subplots(figsize=(10, 4))
portfolio_nav.plot(ax=ax, label="Momentum Strategy")
benchmark.plot(ax=ax, label="Nifty 50 Index")
ax.set_ylabel("NAV (Indexed)")
ax.legend()
st.pyplot(fig)

# --- Performance Metrics ---
def calculate_performance(nav_series):
    nav_series = nav_series.dropna()
    total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    num_years = (nav_series.index[-1] - nav_series.index[0]).days / 365.25
    cagr = (nav_series.iloc[-1] / nav_series.iloc[0])**(1/num_years) - 1
    roll_max = nav_series.cummax()
    drawdown = nav_series / roll_max - 1
    max_dd = drawdown.min()
    daily_returns = nav_series.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    return {
        "CAGR": cagr,
        "Max Drawdown": max_dd,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe
    }

strategy_perf = calculate_performance(portfolio_nav)
benchmark_perf = calculate_performance(benchmark)

perf_df = pd.DataFrame([strategy_perf, benchmark_perf], index=["Momentum Strategy", "Nifty 50 Index"])
st.subheader("📊 Performance Summary")
st.dataframe(perf_df.style.format("{:.2%}").highlight_max(color='lightgreen', axis=0))

# --- Monthly Top Sector Picks Table ---
top_picks_df = pd.DataFrame(top_picks)
top_picks_df["Month_dt"] = pd.to_datetime(top_picks_df["Month"], format="%b-%Y")
top_picks_df = top_picks_df.sort_values("Month_dt").drop("Month_dt", axis=1)

st.subheader("🏆 Top 2 Sector Picks Each Month")
st.dataframe(top_picks_df.style.format({"Score 1": "{:.2%}", "Score 2": "{:.2%}"}))
