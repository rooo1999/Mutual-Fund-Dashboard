import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(page_title="Mutual Fund Monthly Returns Dashboard", layout="wide")
st.title("📈 Mutual Fund Monthly Returns Dashboard")

start_date = datetime(2024, 1, 1)
today = datetime.today() - timedelta(days=1)
display_start_month = datetime(2024, 6, 1)

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
    df = df[df.index >= start_date]
    scheme_name = data.get("meta", {}).get("scheme_name", f"Scheme {scheme_code}")
    return df, scheme_name

def calculate_monthly_returns(nav_df):
    monthly_nav = nav_df['nav'].resample('M').last()
    mom_returns = monthly_nav.pct_change() * 100
    mom_returns = mom_returns.round(2)
    return mom_returns

def calculate_mtd(nav_df):
    this_month_start = datetime(today.year, today.month, 1)

    # End NAV = latest NAV on or before today
    navs_till_today = nav_df.loc[:today]
    if navs_till_today.empty:
        return None
    end_nav = navs_till_today['nav'].iloc[-1]

    # Start NAV = latest NAV on or before 1st of the month
    navs_till_month_start = nav_df.loc[:this_month_start]
    if navs_till_month_start.empty:
        return None
    start_nav = navs_till_month_start['nav'].iloc[-1]

    mtd = ((end_nav - start_nav) / start_nav) * 100
    return round(mtd, 2)

# Month range
all_months_dt = pd.date_range(start=start_date, end=today, freq='M')
months_from_june_dt = [m for m in all_months_dt if m >= display_start_month]
month_display_labels = [m.strftime('%b-%Y') for m in months_from_june_dt]

# Excel-style colormap
excel_cmap = LinearSegmentedColormap.from_list("excel_like", ["#f8696b", "#ffeb84", "#63be7b"])

# Benchmarks
benchmark_scheme_codes = {
    "Nifty 50": "147794",
    "Nifty 500": "147625",
    "Smallcap 250": "147623",
    "Midcap 150": "147622",
    "Sensex": "119597"
}

# Portfolios
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

# Benchmark preloading
benchmark_returns = {}
benchmark_mtds = {}

for b_name, b_code in benchmark_scheme_codes.items():
    nav_df, _ = get_nav_history(b_code)
    if nav_df.empty:
        continue
    monthly_returns = calculate_monthly_returns(nav_df).reindex(months_from_june_dt)
    monthly_returns.index = month_display_labels
    benchmark_returns[b_name] = monthly_returns
    benchmark_mtds[b_name] = calculate_mtd(nav_df)

st.header("📆 Monthly MoM Returns (from Jun 2024)")

# Portfolio display
for i in range(len(default_portfolio_names)):
    name = default_portfolio_names[i]
    portfolio = default_portfolios[i]

    st.subheader(f"🧾 {name} Portfolio")

    if not portfolio or sum(portfolio.values()) == 0:
        st.write("No valid allocation.")
        continue

    fund_monthly_returns = {}
    weighted_returns = pd.Series(0, index=month_display_labels)
    fund_nav_data = {}
    portfolio_mtds = []
    fund_weights = {}

    with st.spinner(f"Fetching NAVs for {name}..."):
        for scheme_code, weight in portfolio.items():
            nav_df, scheme_name = get_nav_history(scheme_code)
            if nav_df.empty:
                st.warning(f"Data not available for scheme {scheme_code}")
                continue
            monthly_returns = calculate_monthly_returns(nav_df).reindex(months_from_june_dt)
            monthly_returns.index = month_display_labels
            fund_monthly_returns[scheme_name] = monthly_returns
            weighted_returns += monthly_returns.fillna(0) * weight
            fund_nav_data[scheme_name] = nav_df
            fund_weights[scheme_name] = weight
            mtd = calculate_mtd(nav_df)
            if mtd is not None:
                portfolio_mtds.append(mtd * weight)

    if not fund_monthly_returns:
        st.write("No data available for this portfolio.")
        continue

    # Table 1: Fund-wise table
    fund_df = pd.DataFrame(fund_monthly_returns).T[month_display_labels]
    fund_df["MTD"] = [calculate_mtd(fund_nav_data.get(fund)) for fund in fund_df.index]
    fund_df["Weight"] = fund_df.index.map(fund_weights).round(2)
    fund_df = fund_df[["Weight"] + month_display_labels + ["MTD"]]

    styled_fund_df = fund_df.style.format(lambda x: f"{x:.2f}" if pd.notnull(x) else "").background_gradient(
        cmap=excel_cmap, axis=0)
    st.markdown("**Fund-wise Performance:**")
    st.dataframe(styled_fund_df, use_container_width=True)

    # Table 2: Portfolio + Benchmarks table
    summary_df = pd.DataFrame(columns=month_display_labels + ["MTD"])

    # Portfolio row
    summary_df.loc[f"{name} Portfolio"] = list(weighted_returns.values) + [round(sum(portfolio_mtds), 2) if portfolio_mtds else None]

    # Benchmarks
    for b_name, b_returns in benchmark_returns.items():
        row = list(b_returns.values) + [benchmark_mtds.get(b_name)]
        summary_df.loc[f"📊 {b_name}"] = row

    styled_summary_df = summary_df.style.format(lambda x: f"{x:.2f}" if pd.notnull(x) else "").background_gradient(
        cmap=excel_cmap, axis=0)

    st.markdown("**Portfolio vs Benchmarks:**")
    st.dataframe(styled_summary_df, use_container_width=True)

    st.markdown("---")
