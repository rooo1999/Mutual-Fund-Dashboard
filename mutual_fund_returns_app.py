import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="Mutual Fund Returns Dashboard", layout="wide")
st.title("📈 Mutual Fund Returns Dashboard")

today = datetime.today() - timedelta(days=1)

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

benchmark_scheme_codes = {
    "Motilal Nifty 50": "147794",
    "Motilal Nifty 500": "147625",
    "Motilal Smallcap 250": "147623"
}

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

st.header("📊 Calculated Returns")

for i in range(len(default_portfolio_names)):
    name = default_portfolio_names[i]
    portfolio = default_portfolios[i]

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

    fund_df = pd.DataFrame(fund_returns_dict).T
    fund_df.index.name = "Fund Name"
    for period in ["1Y", "3Y", "5Y"]:
        if period in fund_df.columns:
            fund_df.rename(columns={period: period + " (CAGR)"}, inplace=True)

    fund_styled = fund_df.style.format("{:.2f}")
    numeric_cols = [col for col in fund_df.columns if fund_df[col].dtype in ['float64', 'int64']]
    fund_styled = fund_styled.background_gradient(cmap='BrBG', axis=0, subset=numeric_cols)
    st.dataframe(fund_styled)

    combined_df = pd.DataFrame([portfolio_returns], index=[f"{name} Weighted Returns (%)"])
    for benchmark_name, scheme_code in benchmark_scheme_codes.items():
        nav_df, _ = get_nav_history(scheme_code)
        if nav_df.empty:
            continue
        returns = calculate_returns(nav_df)
        combined_df.loc[f"Benchmark: {benchmark_name}"] = returns

    for period in ["1Y", "3Y", "5Y"]:
        if period in combined_df.columns:
            combined_df.rename(columns={period: period + " (CAGR)"}, inplace=True)

    combined_styled = combined_df.style.format("{:.2f}")
    numeric_cols_combined = [col for col in combined_df.columns if combined_df[col].dtype in ['float64', 'int64']]
    combined_styled = combined_styled.background_gradient(cmap='BrBG', axis=0, subset=numeric_cols_combined)
    st.dataframe(combined_styled)

    st.markdown("---")
