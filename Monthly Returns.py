import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(page_title="Mutual Fund Monthly Returns Dashboard", layout="wide")
st.title("📈 Mutual Fund Monthly Returns Dashboard")

start_date = datetime(2024, 1, 1)
today = datetime.today() - timedelta(days=1)
display_start_month = "Jun-2024"

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

def calculate_mtd_return(nav_df):
    start_of_month = datetime(today.year, today.month, 1)
    try:
        nav_start = nav_df.loc[nav_df.index >= start_of_month, 'nav'].iloc[0]
        nav_now = nav_df['nav'].iloc[-1]
        mtd = ((nav_now - nav_start) / nav_start) * 100
        return round(mtd, 2)
    except:
        return None

def get_all_months(start_date):
    today = datetime.today()
    months = pd.date_range(start=start_date, end=today, freq='M').strftime('%b-%Y')
    return months.tolist()

# Color gradient
excel_cmap = LinearSegmentedColormap.from_list("excel_like", ["#f8696b", "#ffeb84", "#63be7b"])
all_months = get_all_months(start_date)
months_from_june = [m for m in all_months if m >= display_start_month]

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

# Preload benchmark data
benchmark_returns = {}
benchmark_mtd = {}

for b_name, b_code in benchmark_scheme_codes.items():
    nav_df, _ = get_nav_history(b_code)
    if nav_df.empty:
        continue
    benchmark_returns[b_name] = calculate_monthly_returns(nav_df).reindex(all_months)
    benchmark_mtd[b_name] = calculate_mtd_return(nav_df)

# Display
st.header("📆 Monthly MoM Returns (from Jun 2024) + 📊 MTD")

for i in range(len(default_portfolio_names)):
    name = default_portfolio_names[i]
    portfolio = default_portfolios[i]

    st.subheader(f"🧾 {name} Portfolio Returns + Benchmarks")

    if not portfolio or sum(portfolio.values()) == 0:
        st.write("No valid allocation.")
        continue

    fund_monthly_returns = {}
    weighted_returns = pd.Series(0, index=all_months)
    mtd_row = {}

    with st.spinner(f"Fetching NAVs for {name}..."):
        for scheme_code, weight in portfolio.items():
            nav_df, scheme_name = get_nav_history(scheme_code)
            if nav_df.empty:
                st.warning(f"Data not available for scheme {scheme_code} in {name}")
                continue
            monthly_returns = calculate_monthly_returns(nav_df).reindex(all_months)
            mtd = calculate_mtd_return(nav_df)
            mtd_row[scheme_name] = mtd
            fund_monthly_returns[scheme_name] = monthly_returns
            weighted_returns = weighted_returns.add(monthly_returns.fillna(0) * weight, fill_value=0)

    if not fund_monthly_returns:
        st.write("No data available for this portfolio.")
        continue

    # Build main DataFrame
    combined_df = pd.DataFrame(fund_monthly_returns).T[months_from_june]
    combined_df["MTD"] = pd.Series(mtd_row)

    # Weighted row
    weighted_df = weighted_returns[months_from_june].to_frame().T
    weighted_df["MTD"] = calculate_mtd_return(weighted_returns.to_frame(name="nav"))
    weighted_df.index = [f"{name} Portfolio"]
    combined_df = pd.concat([combined_df, weighted_df])

    # Add benchmarks
    for b_name, b_returns in benchmark_returns.items():
        b_row = b_returns[months_from_june].to_frame().T
        b_row["MTD"] = benchmark_mtd.get(b_name, None)
        b_row.index = [f"📊 {b_name}"]
        combined_df = pd.concat([combined_df, b_row])

    # Bold Portfolio and Benchmarks
    bold_rows = [f"{name} Portfolio"] + [f"📊 {bn}" for bn in benchmark_scheme_codes.keys()]
    
    def bold_style(val, row_idx):
        return "font-weight: bold" if row_idx in bold_rows else ""

    styled_df = combined_df.style.format("{:.2f}")\
        .apply(lambda df: [[bold_style(val, df.index[i]) for val in row] for i, row in enumerate(df.values)], axis=None)\
        .background_gradient(cmap=excel_cmap, axis=0)

    st.dataframe(styled_df, use_container_width=True)
    st.markdown("---")
