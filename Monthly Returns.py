import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(page_title="Mutual Fund Monthly Returns Dashboard", layout="wide")
st.title("📈 Mutual Fund Monthly Returns Dashboard")

# Start date for monthly returns
start_date = datetime(2024, 1, 1)
today = datetime.today() - timedelta(days=1)

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
    mom_returns.index = mom_returns.index.strftime('%b-%Y')
    return mom_returns

# Excel-style color gradient
excel_cmap = LinearSegmentedColormap.from_list("excel_like", ["#f8696b", "#ffeb84", "#63be7b"])

# Portfolio definitions
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

st.header("📆 Monthly MoM Returns (from Jan 2024)")

for i in range(len(default_portfolio_names)):
    name = default_portfolio_names[i]
    portfolio = default_portfolios[i]

    st.subheader(f"{name} Monthly Returns")
    if not portfolio or sum(portfolio.values()) == 0:
        st.write("No valid allocation.")
        continue

    fund_monthly_returns = {}
    with st.spinner(f"Fetching NAVs for {name}..."):
        for scheme_code, weight in portfolio.items():
            nav_df, scheme_name = get_nav_history(scheme_code)
            if nav_df.empty:
                st.warning(f"Data not available for scheme {scheme_code} in {name}")
                continue
            monthly_returns = calculate_monthly_returns(nav_df)
            fund_monthly_returns[scheme_name] = monthly_returns

    if not fund_monthly_returns:
        st.write("No data available for this portfolio.")
        continue

    monthly_df = pd.DataFrame(fund_monthly_returns).T
    styled_df = monthly_df.style.format("{:.2f}").background_gradient(cmap=excel_cmap, axis=0)
    st.dataframe(styled_df, use_container_width=True)

    st.markdown("---")
