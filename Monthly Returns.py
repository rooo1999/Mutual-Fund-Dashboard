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
    return mom_returns

def calculate_mtd(nav_df):
    this_month_start = datetime(today.year, today.month, 1)
    recent_df = nav_df[nav_df.index >= this_month_start]
    if recent_df.empty or this_month_start not in nav_df.index:
        return None
    start_nav = nav_df.loc[nav_df.index >= this_month_start].iloc[0]['nav']
    end_nav = recent_df.iloc[-1]['nav']
    mtd = ((end_nav - start_nav) / start_nav) * 100
    return round(mtd, 2)

# Setup month range
all_months_dt = pd.date_range(start=start_date, end=today, freq='M')
months_from_june_dt = [m for m in all_months_dt if m.strftime('%b-%Y') >= display_start_month]
month_display_labels = [m.strftime('%b-%Y') for m in months_from_june_dt]

# Excel-like color map
excel_cmap = LinearSegmentedColormap.from_list("excel_like", ["#f8696b", "#ffeb84", "#63be7b"])

# Benchmark definitions
benchmark_scheme_codes = {
    "Nifty 50": "147794",
    "Nifty 500": "147625",
    "Smallcap 250": "147623",
    "Midcap 150": "147622",
    "Sensex": "119597"
}

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

# Preload benchmark returns
benchmark_returns = {}
benchmark_mtds = {}

for b_name, b_code in benchmark_scheme_codes.items():
    nav_df, _ = get_nav_history(b_code)
    if nav_df.empty:
        continue
    monthly_returns = calculate_monthly_returns(nav_df).reindex(months_from_june_dt)
    monthly_returns.index = month_display_labels
    benchmark_returns[b_name] = monthly_returns
    mtd = calculate_mtd(nav_df)
    benchmark_mtds[b_name] = mtd

st.header("📆 Monthly MoM Returns (from Jun 2024)")

# Show portfolios
for i in range(len(default_portfolio_names)):
    name = default_portfolio_names[i]
    portfolio = default_portfolios[i]

    st.subheader(f"🧾 {name} Portfolio Returns + Benchmarks")

    if not portfolio or sum(portfolio.values()) == 0:
        st.write("No valid allocation.")
        continue

    fund_monthly_returns = {}
    weighted_returns = pd.Series(0, index=month_display_labels)
    portfolio_mtds = []
    
    with st.spinner(f"Fetching NAVs for {name}..."):
        for scheme_code, weight in portfolio.items():
            nav_df, scheme_name = get_nav_history(scheme_code)
            if nav_df.empty:
                st.warning(f"Data not available for scheme {scheme_code} in {name}")
                continue
            monthly_returns = calculate_monthly_returns(nav_df).reindex(months_from_june_dt)
            monthly_returns.index = month_display_labels
            fund_monthly_returns[scheme_name] = monthly_returns
            weighted_returns = weighted_returns.add(monthly_returns.fillna(0) * weight, fill_value=0)
            mtd = calculate_mtd(nav_df)
            if mtd is not None:
                portfolio_mtds.append(mtd * weight)

    if not fund_monthly_returns:
        st.write("No data available for this portfolio.")
        continue

    combined_df = pd.DataFrame(fund_monthly_returns).T[month_display_labels]

    # Add weighted portfolio row
    weighted_df = weighted_returns.to_frame().T
    weighted_df.index = [f"{name} Portfolio"]
    combined_df = pd.concat([combined_df, weighted_df])

    # Add benchmarks
    for b_name, b_returns in benchmark_returns.items():
        b_row = b_returns.to_frame().T
        b_row.index = [f"📊 {b_name}"]
        combined_df = pd.concat([combined_df, b_row])

    # Add MTD column
    mtd_column = []
    for label in combined_df.index:
        if label.endswith("Portfolio"):
            mtd_val = round(sum(portfolio_mtds), 2) if portfolio_mtds else None
        elif label.startswith("📊"):
            benchmark_name = label[2:].strip()
            mtd_val = benchmark_mtds.get(benchmark_name)
        else:
            # Extract scheme code by name lookup (optional optimization)
            found = False
            for port in default_portfolios:
                for code, weight in port.items():
                    _, scheme = get_nav_history(code)
                    if scheme == label:
                        df, _ = get_nav_history(code)
                        mtd_val = calculate_mtd(df)
                        found = True
                        break
                if found:
                    break
            else:
                mtd_val = None
        mtd_column.append(mtd_val)
    combined_df["MTD"] = mtd_column

    # Display styled table
    styled_df = combined_df.style.format("{:.2f}").background_gradient(cmap=excel_cmap, axis=0)
    st.dataframe(styled_df, use_container_width=True)
    st.markdown("---")
