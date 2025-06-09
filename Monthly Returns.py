import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(page_title="Mutual Fund Returns Dashboard", layout="wide")
st.title("📈 Mutual Fund Returns Dashboard")

today = datetime.today() - timedelta(days=1)
start_date = datetime(2024, 1, 1)

# Custom Excel-style gradient colormap
excel_cmap = LinearSegmentedColormap.from_list("excel_like", ["#f8696b", "#ffeb84", "#63be7b"])

@st.cache_data(ttl=86400)
def get_nav_history(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    response = requests.get(url)
    data = response.json()
    if 'data' not in data or not isinstance(data['data'], list) or len(data['data']) == 0:
        return pd.DataFrame(), None
    df = pd.DataFrame(data['data'])
    df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y", errors='coerce')
    df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
    df.dropna(subset=['date', 'nav'], inplace=True)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    scheme_name = data.get("meta", {}).get("scheme_name", f"Scheme {scheme_code}")
    return df, scheme_name

def calculate_monthly_returns(nav_df):
    # Sort by date
    nav_df = nav_df.sort_index()

    # Month ends from Jan 2024 to today
    month_ends = pd.date_range(start=start_date, end=today, freq='M')

    month_end_navs = []
    last_valid_nav = None

    for me in month_ends:
        eligible_navs = nav_df.loc[nav_df.index <= me]
        if not eligible_navs.empty:
            last_valid_nav = eligible_navs['nav'].iloc[-1]
        month_end_navs.append(last_valid_nav)

    monthly_nav = pd.Series(month_end_navs, index=month_ends)

    # Calculate MoM returns in %
    mom_returns = monthly_nav.pct_change() * 100
    mom_returns = mom_returns.round(2)

    # Format index as 'Mon-YYYY'
    mom_returns.index = mom_returns.index.strftime('%b-%Y')

    return mom_returns

# Benchmark funds — you can add up to 5 benchmarks here
benchmark_scheme_codes = {
    "Nifty 50": "147794",
    "Nifty 500": "147625",
    "Smallcap 250": "147623",
    "Nifty Midcap 150": "147626",
    "Nifty Bank": "147642"
}

default_portfolio_names = [
    "High Growth Active", "High Growth Passive", "Sector Rotation",
    "Season's Flavor", "Smart Debt", "Global Equity"
]

default_portfolios = [
    {"102000": 0.24, "106235": 0.24, "105758": 0.11, "140225": 0.11, "122640": 0.15, "109522": 0.15},
    {"147795": 0.3, "106235": 0.2, "113296": 0.15, "152352": 0.1, "150490": 0.15, "152232": 0.1},
    {"145077": 0.3, "102431": 0.25, "135794": 0.25, "152724": 0.2},
    {"145077": 0.6, "101262": 0.4},
    {"111524": 0.4, "151314": 0.4, "101042": 0.2},
    {"148486": 0.3, "148064": 0.2, "140242": 0.2, "132005": 0.3}
]

st.header("📊 Monthly Returns from Jan 2024")

for i in range(len(default_portfolio_names)):
    name = default_portfolio_names[i]
    portfolio = default_portfolios[i]

    st.subheader(f"{name} Portfolio Monthly Returns")

    if not portfolio or sum(portfolio.values()) == 0:
        st.write("No valid allocation.")
        continue

    fund_returns_dict = {}

    with st.spinner(f"Fetching NAVs for {name}..."):
        for scheme_code, weight in portfolio.items():
            nav_df, scheme_name = get_nav_history(scheme_code)
            if nav_df.empty:
                st.warning(f"Data not available for scheme {scheme_code} in {name}")
                continue

            monthly_returns = calculate_monthly_returns(nav_df)
            # Multiply by weight
            weighted_returns = monthly_returns * weight
            fund_returns_dict[scheme_name] = weighted_returns

    if not fund_returns_dict:
        st.write("No NAV data found for any funds in this portfolio.")
        continue

    # Create DataFrame: rows = funds, columns = months
    fund_df = pd.DataFrame(fund_returns_dict).T

    # Portfolio returns = sum of weighted returns per month
    portfolio_returns = fund_df.sum(axis=0).to_frame(name=f"{name} Portfolio")

    # Get benchmark returns
    benchmark_returns_dict = {}
    for b_name, b_code in benchmark_scheme_codes.items():
        nav_df, _ = get_nav_history(b_code)
        if nav_df.empty:
            st.warning(f"Benchmark data not available for {b_name}")
            continue
        bench_returns = calculate_monthly_returns(nav_df)
        benchmark_returns_dict[b_name] = bench_returns

    benchmark_df = pd.DataFrame(benchmark_returns_dict)

    # Combine portfolio and benchmark returns vertically
    combined_df = pd.concat([portfolio_returns, benchmark_df.T])

    # Show table with color gradient
    styled_df = combined_df.style.format("{:.2f}").background_gradient(cmap=excel_cmap, axis=1)
    st.dataframe(styled_df)

    st.markdown("---")
