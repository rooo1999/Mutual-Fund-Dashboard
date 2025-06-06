import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="Mutual Fund Returns Dashboard", layout="wide")
st.title("📈 Mutual Fund Returns Dashboard")

# Use yesterday as "today" because today's data may not be available yet
today = datetime.today() - timedelta(days=1)

# Return periods
date_dict = {
    "1D": today - timedelta(days=1),
    "1W": today - timedelta(weeks=1),
    "1M": today - relativedelta(months=1),
    "3M": today - relativedelta(months=3),
    "6M": today - relativedelta(months=6),
    "1Y": today - relativedelta(years=1),
    "3Y": today - relativedelta(years=3),
    "5Y": today - relativedelta(years=5),
    "YTD": datetime(today.year, 1, 1),
    "MTD": datetime(today.year, today.month, 1)
}

@st.cache_data(ttl=86400)
def get_nav_history(scheme_code):
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    response = requests.get(url)
    data = response.json()

    if 'data' not in data or not isinstance(data['data'], list) or len(data['data']) == 0:
        return pd.DataFrame(), None

    first_entry = data['data'][0]
    if 'date' not in first_entry or 'nav' not in first_entry:
        return pd.DataFrame(), None

    df = pd.DataFrame(data['data'])
    df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y", errors='coerce')
    df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
    df.dropna(subset=['date', 'nav'], inplace=True)
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    scheme_name = data.get("meta", {}).get("scheme_name", f"Scheme {scheme_code}")
    return df, scheme_name

def calculate_returns(nav_df):
    returns = {}
    current_navs = nav_df[nav_df.index <= today]
    if current_navs.empty:
        return {k: None for k in date_dict}
    current_nav = current_navs['nav'].iloc[-1]
    for label, date in date_dict.items():
        past_navs = nav_df[nav_df.index <= date]
        if not past_navs.empty:
            past_nav = past_navs['nav'].iloc[-1]
            if label in ["1Y", "3Y", "5Y"]:
                years = int(label[0])
                cagr = ((current_nav / past_nav) ** (1 / years) - 1) * 100
                returns[label] = cagr
            else:
                abs_return = (current_nav - past_nav) / past_nav * 100
                returns[label] = abs_return
        else:
            returns[label] = None
    return returns

# Benchmark fund scheme codes
benchmark_scheme_codes = {
    "Nifty 50 Index Fund": "147795",
    "Nifty Next 50 Index Fund": "140825",
    "Nifty Midcap 150 Index Fund": "140242"
}

# Predefined portfolio names and allocations
default_portfolio_names = [
    "High Growth Active",
    "High Growth Passive",
    "Sector Rotation",
    "Season's Flavor",
    "Smart Debt",
    "Global Equity"
]

default_portfolios = [
    {"102000": 0.24, "106235": 0.24, "105758": 0.11, "140225": 0.11, "122640": 0.15, "109522": 0.15},
    {"147795": 0.3, "106235": 0.2, "113296": 0.15, "152352": 0.1, "150490": 0.15, "152232": 0.1},
    {"145077": 0.3, "102431": 0.25, "135794": 0.25, "152724": 0.2},
    {"145077": 0.6, "101262": 0.4},
    {"111524": 0.4, "151314": 0.4, "101042": 0.2},
    {"148486": 0.3, "148064": 0.2, "140242": 0.2, "132005": 0.3}
]

def input_portfolio(portfolio_num, default_portfolio, default_name):
    st.subheader(f"Portfolio {portfolio_num} Allocation")
    portfolio_name = st.text_input(f"Portfolio {portfolio_num} Name", value=default_name, key=f"portfolio_name_{portfolio_num}")

    num_funds = st.number_input(f"Number of funds in {portfolio_name}", min_value=1, max_value=10, value=len(default_portfolio), key=f"numfunds{portfolio_num}")
    portfolio = {}
    for i in range(num_funds):
        default_code = list(default_portfolio.keys())[i] if i < len(default_portfolio) else ""
        default_weight = list(default_portfolio.values())[i] if i < len(default_portfolio) else 0.0
        scheme_code = st.text_input(f"Scheme Code {i+1} ({portfolio_name})", value=default_code, key=f"code{portfolio_num}_{i}")
        weight = st.number_input(f"Weight {i+1} ({portfolio_name})", min_value=0.0, max_value=1.0, value=default_weight, key=f"weight{portfolio_num}_{i}")
        if scheme_code:
            portfolio[scheme_code] = weight
    total_weight = sum(portfolio.values())
    if total_weight > 0:
        for k in portfolio:
            portfolio[k] /= total_weight
    return portfolio_name, portfolio

portfolio_data = []
for i in range(1, 7):
    default_name = default_portfolio_names[i-1] if i-1 < len(default_portfolio_names) else f"Portfolio {i}"
    name, portfolio = input_portfolio(i, default_portfolios[i-1] if i-1 < len(default_portfolios) else {}, default_name)
    portfolio_data.append((name, portfolio))

st.markdown("---")
st.header("📊 Calculated Returns")

for name, portfolio in portfolio_data:
    st.subheader(f"{name} Returns")
    if not portfolio or sum(portfolio.values()) == 0:
        st.write("No valid allocation.")
        continue

    fund_returns_dict = {}
    portfolio_returns = {label: 0 for label in date_dict}

    with st.spinner(f"Fetching NAVs for {name}..."):
        for scheme_code, weight in portfolio.items():
            nav_df, scheme_name = get_nav_history(scheme_code)
            if nav_df.empty:
                st.warning(f"Data not available for scheme {scheme_code} in {name}")
                continue
            returns = calculate_returns(nav_df)
            fund_returns_dict[scheme_name] = {"Weight": weight, **returns}
            for label in returns:
                if returns[label] is not None:
                    portfolio_returns[label] += returns[label] * weight

        # Append benchmark returns to fund_returns_dict
        for benchmark_name, scheme_code in benchmark_scheme_codes.items():
            nav_df, _ = get_nav_history(scheme_code)
            if nav_df.empty:
                continue
            returns = calculate_returns(nav_df)
            fund_returns_dict[f"🟨 Benchmark: {benchmark_name}"] = {"Weight": None, **returns}

    fund_df = pd.DataFrame(fund_returns_dict).T
    fund_df.index.name = "Fund Name"
    for period in ["1Y", "3Y", "5Y"]:
        if period in fund_df.columns:
            fund_df.rename(columns={period: period + " (CAGR)"}, inplace=True)
    st.dataframe(fund_df.style.format("{:.2f}"))

    portfolio_df = pd.DataFrame([portfolio_returns], index=[f"{name} Weighted Returns (%)"])
    for period in ["1Y", "3Y", "5Y"]:
        if period in portfolio_df.columns:
            portfolio_df.rename(columns={period: period + " (CAGR)"}, inplace=True)
    st.dataframe(portfolio_df.style.format("{:.2f}"))
    st.markdown("---")
