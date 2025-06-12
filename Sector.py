import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sector Momentum Strategy", layout="wide")
st.title("📊 Sector Momentum Strategy Dashboard")

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
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    r = requests.get(url)
    data = r.json()
    navs = data.get("data", [])
    df = pd.DataFrame(navs)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df["nav"] = pd.to_numeric(df["nav"], errors='coerce')
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return df[["date", "nav"]]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- Fetch all NAVs ---
nav_data = {}
for name, code in sector_funds.items():
    df = get_nav_history(code)
    nav_data[name] = df.set_index("date")["nav"]

prices = pd.DataFrame(nav_data).dropna(how='all').fillna(method='ffill')

# --- Momentum Calculation ---
monthly_scores = []
monthly_weights = {}
start_date = prices.index.min() + pd.Timedelta(days=60)
dates = prices.loc[start_date:].resample("M").last().index

for date in dates:
    data = prices.loc[:date].tail(60)  # last 60 days of data
    if len(data) < 50: continue

    scores = {}
    for sector in data.columns:
        series = data[sector]
        if len(series.dropna()) < 50:
            continue

        r1 = series[-1] / series[-21] - 1     # 1M return
        r2 = series[-1] / series[-42] - 1     # 2M return
        sma50 = series.rolling(50).mean().iloc[-1]
        sma20 = series.rolling(20).mean().iloc[-1]
        sma_ratio = series[-1] / sma50 if sma50 else 0
        rsi = calculate_rsi(series).iloc[-1]
        vol = series.pct_change().rolling(21).std().iloc[-1]
        vol_adj_return = r2 / vol if vol else 0
        trend_ok = sma20 > sma50

        scores[sector] = {
            "1M": r1, "2M": r2, "SMA_ratio": sma_ratio,
            "RSI": rsi, "Sharpe": vol_adj_return, "Trend": trend_ok
        }

    df_scores = pd.DataFrame(scores).T.dropna()
    if df_scores.empty: continue

    # Normalize and rank
    for col in ["1M", "2M", "SMA_ratio", "RSI", "Sharpe"]:
        df_scores[col + "_rank"] = df_scores[col].rank(pct=True)

    df_scores["momentum_score"] = (
        0.25 * df_scores["1M_rank"] +
        0.25 * df_scores["2M_rank"] +
        0.10 * df_scores["SMA_ratio_rank"] +
        0.10 * df_scores["RSI_rank"] +
        0.10 * df_scores["Sharpe_rank"] +
        0.10 * df_scores["1M_rank"] +  # relative rank reused
        0.10 * df_scores["Trend"].astype(int)
    )

    top2 = df_scores[df_scores["Trend"]].sort_values("momentum_score", ascending=False).head(2)
    weights = {s: 1/len(top2) for s in top2.index}
    monthly_scores.append((date, df_scores))
    monthly_weights[date] = weights

# --- Portfolio Simulation ---
portfolio_nav = pd.Series(index=prices.index, dtype=float)
portfolio_nav.iloc[0] = 100
last_date = prices.index[0]

for date in monthly_weights.keys():
    if date not in prices.index: continue
    weights = monthly_weights[date]
    next_date = date + pd.DateOffset(months=1)
    future_prices = prices.loc[date:next_date]
    for i, (curr, next_) in enumerate(zip(future_prices.index[:-1], future_prices.index[1:])):
        ret = sum((future_prices.loc[next_, s] / future_prices.loc[curr, s] - 1) * w 
                  for s, w in weights.items() if future_prices.loc[curr, s] and future_prices.loc[next_, s])
        portfolio_nav[next_] = portfolio_nav[curr] * (1 + ret)

portfolio_nav = portfolio_nav.dropna()

# --- Benchmark (e.g. Nifty 50 Index Fund) ---
benchmark_code = "147587"  # ICICI Nifty 50 Index Fund (example)
benchmark_df = get_nav_history(benchmark_code).set_index("date")["nav"]
benchmark = benchmark_df.reindex_like(portfolio_nav).fillna(method="ffill")
benchmark = benchmark / benchmark.iloc[0] * 100

# --- Plot NAVs ---
st.subheader("Portfolio vs Benchmark")
fig, ax = plt.subplots(figsize=(10, 4))
portfolio_nav.plot(ax=ax, label="Strategy Portfolio", color="blue")
benchmark.plot(ax=ax, label="Nifty 50 Index", color="orange")
ax.set_ylabel("NAV")
ax.legend()
st.pyplot(fig)

# --- Monthly Returns Table ---
monthly_returns = portfolio_nav.resample("M").last().pct_change().to_frame("Portfolio")
monthly_returns["Benchmark"] = benchmark.resample("M").last().pct_change()
monthly_returns.index = monthly_returns.index.strftime("%b-%Y")

st.subheader("📅 Monthly Returns")
st.dataframe(monthly_returns.style
    .background_gradient(cmap="RdYlGn", axis=0)
    .format("{:.2%}")
)

# --- Optional: Show latest momentum scores ---
if st.checkbox("Show Latest Momentum Scores"):
    st.subheader("🔍 Latest Momentum Scores")
    last_date, last_scores = monthly_scores[-1]
    st.write(f"**As of: {last_date.strftime('%d-%b-%Y')}**")
    st.dataframe(last_scores.sort_values("momentum_score", ascending=False).style.format("{:.2f}"))
