import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Configuration ---
st.set_page_config(
    page_title="Sector Momentum Strategy",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants & Configuration ---
API_URL_TEMPLATE = "https://api.mfapi.in/mf/{}"
DEFAULT_SECTOR_FUNDS = {
    "Auto": "150645",
    "IT": "148466",
    "Pharma": "150929",
    "Bank": "149859",
    "Momentum": "150454",
}
DEFAULT_BENCHMARK_CODE = "147626" # ICICI Prudential Nifty 50 Index Fund

# --- Caching & Data Fetching ---
@st.cache_data(ttl=3600)
def get_nav_history(scheme_code: str):
    """Fetches and processes NAV history for a given mutual fund scheme code."""
    try:
        url = API_URL_TEMPLATE.format(scheme_code)
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        if "data" not in data or not data["data"]:
            st.error(f"No data found for scheme code {scheme_code}. It might be an invalid code.")
            return None

        df = pd.DataFrame(data["data"])
        df["date"] = pd.to_datetime(df["date"], dayfirst=True)
        df["nav"] = pd.to_numeric(df["nav"], errors='coerce')
        df = df.dropna().sort_values("date").reset_index(drop=True)
        return df.set_index("date")["nav"]
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch data for scheme code {scheme_code}: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while processing data for {scheme_code}: {e}")
        return None

# --- Calculation Functions ---
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_momentum_scores(prices: pd.DataFrame, lookback_periods: dict, sma_period: int, start_date: datetime):
    """Calculates momentum scores for each sector at each rebalancing date."""
    monthly_scores = []
    monthly_weights = {}

    # --- FIX 1: Use Business Month End to avoid weekend/holiday issues ---
    all_rebalance_dates = prices.resample("BM").last().index
    rebalance_dates = all_rebalance_dates[all_rebalance_dates >= start_date]

    # --- FIX 2: The main bug fix. Correctly calculate required data points. ---
    # To calculate a `d`-day return, we need `d+1` data points (e.g., today and d days ago).
    max_lookback_for_return = max(lookback_periods.values()) + 1
    required_data_points = max(max_lookback_for_return, sma_period)

    for date in rebalance_dates:
        # Check if this rebalance date actually exists in our price index
        try:
            date_loc = prices.index.get_loc(date)
        except KeyError:
            # If the business month-end is a holiday, it might not exist. Skip it.
            continue

        # Ensure we have enough historical data *in the full prices dataframe*
        if date_loc < required_data_points:
            continue

        data_window = prices.loc[:date]

        scores = {}
        for sector in data_window.columns:
            series = data_window[sector].dropna()
            # This check is now robust because `required_data_points` is correct
            if len(series) < required_data_points:
                continue

            try:
                # To get a d-day return, we divide today's price (iloc[-1])
                # by the price d-days ago (iloc[-(d+1)])
                returns = {f"{p}M": series.iloc[-1] / series.iloc[-(d + 1)] - 1 for p, d in lookback_periods.items()}
                sma_val = series.rolling(sma_period).mean().iloc[-1]
                rsi_val = calculate_rsi(series).iloc[-1]

                scores[sector] = {
                    **returns,
                    "SMA_Ratio": series.iloc[-1] / sma_val if sma_val else 0,
                    "RSI": rsi_val,
                }
            except (IndexError, KeyError, ZeroDivisionError):
                # Catching errors robustly, but the main IndexError should now be gone
                continue

        if not scores:
            continue

        df_scores = pd.DataFrame(scores).T.dropna()
        if df_scores.empty:
            continue

        rank_cols = []
        for col in df_scores.columns:
            rank_col = col + "_rank"
            df_scores[rank_col] = df_scores[col].rank(pct=True)
            rank_cols.append(rank_col)

        df_scores["momentum_score"] = df_scores[rank_cols].mean(axis=1)
        monthly_scores.append((date, df_scores))
        
        top_n = df_scores.sort_values("momentum_score", ascending=False).head(st.session_state.top_n_sectors)
        
        num_assets = len(top_n)
        weights = {s: 1 / num_assets for s in top_n.index} if num_assets > 0 else {}
        monthly_weights[date] = weights
        
    return monthly_scores, monthly_weights

def run_backtest(prices: pd.DataFrame, monthly_weights: dict):
    """Simulates the portfolio performance based on monthly weights."""
    if not monthly_weights:
        return pd.Series(dtype=float)

    portfolio_nav = pd.Series(index=prices.index, dtype=float)
    daily_returns = prices.pct_change()
    sorted_dates = sorted(monthly_weights.keys())
    
    start_date = sorted_dates[0]
    portfolio_nav.loc[start_date] = 100
    
    for i, current_rebalance_date in enumerate(sorted_dates):
        start_period = current_rebalance_date
        end_period = sorted_dates[i + 1] if i + 1 < len(sorted_dates) else prices.index[-1]
        
        start_nav_series = portfolio_nav.loc[:start_period].ffill()
        if start_nav_series.empty: continue
        start_nav = start_nav_series.iloc[-1]

        weights = monthly_weights[start_period]
        period_daily_returns = daily_returns.loc[start_period:end_period]
        
        if not weights or period_daily_returns.empty:
            portfolio_nav.loc[start_period:end_period] = start_nav
            continue
            
        aligned_weights = pd.Series(weights).reindex(period_daily_returns.columns, fill_value=0)
        portfolio_period_returns = period_daily_returns.dot(aligned_weights)
        
        # Start calculating cumulative product from the day *after* rebalancing
        nav_path = start_nav * (1 + portfolio_period_returns.iloc[1:]).cumprod()
        portfolio_nav.update(nav_path)

    return portfolio_nav.ffill().dropna()

def calculate_kpis(nav_series: pd.Series):
    """Calculates key performance indicators for a given NAV series."""
    if nav_series.empty or len(nav_series) < 2:
        return {"CAGR": 0, "Max Drawdown": 0, "Sharpe Ratio": 0, "Drawdown Series": pd.Series()}

    returns = nav_series.pct_change().dropna()
    
    total_return = (nav_series.iloc[-1] / nav_series.iloc[0])
    days = (nav_series.index[-1] - nav_series.index[0]).days
    cagr = (total_return ** (365.25 / days)) - 1 if days > 0 else 0
    
    cumulative_max = nav_series.cummax()
    drawdown = (nav_series - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    return {
        "CAGR": cagr,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe_ratio,
        "Drawdown Series": drawdown
    }

def plot_performance_chart(portfolio_nav, benchmark_nav, portfolio_drawdown, benchmark_drawdown):
    """Creates an interactive Plotly chart for performance and drawdown."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
        row_heights=[0.7, 0.3], subplot_titles=("Portfolio vs. Benchmark NAV", "Drawdown")
    )

    fig.add_trace(go.Scatter(x=portfolio_nav.index, y=portfolio_nav, name="Strategy Portfolio", line=dict(color='royalblue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=benchmark_nav.index, y=benchmark_nav, name="Benchmark (Nifty 50)", line=dict(color='darkorange', width=2)), row=1, col=1)

    fig.add_trace(go.Scatter(x=portfolio_drawdown.index, y=portfolio_drawdown, name="Strategy Drawdown", line=dict(color='royalblue', width=1), fill='tozeroy'), row=2, col=1)
    fig.add_trace(go.Scatter(x=benchmark_drawdown.index, y=benchmark_drawdown, name="Benchmark Drawdown", line=dict(color='darkorange', width=1), fill='tozeroy'), row=2, col=1)

    fig.update_layout(
        height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis1_title="Normalized NAV", yaxis2_title="Drawdown", yaxis2_tickformat=".0%", hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Streamlit UI ---
st.title("📊 Sector Momentum Strategy Dashboard")
st.markdown("""
This dashboard backtests a **rotational momentum strategy** on sectoral mutual funds. 
The strategy invests in the top-performing sectors each month based on a composite momentum score.
Configure the parameters in the sidebar and analyze the results below.
""")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Timeframe")
    start_date_input = st.date_input(
        "Strategy Start Date",
        value=datetime(2020, 1, 1),
        min_value=datetime(2013, 1, 1), 
        max_value=datetime.today(),
        help="Select the date from which the backtest should begin. The actual start may be slightly later to gather enough data for initial calculations."
    )

    st.subheader("Strategy Parameters")
    st.session_state.top_n_sectors = st.slider(
        "Number of Top Sectors to Invest In", 
        min_value=1, max_value=5, value=2, step=1,
        help="The strategy will allocate capital equally among this many top-ranked sectors."
    )

    sma_period = st.number_input("SMA Lookback (days)", min_value=20, max_value=200, value=50, step=10)
    # Using business days for lookbacks
    lookback_periods = {"1M": 21, "3M": 63, "6M": 126}

    st.subheader("Benchmark")
    benchmark_code = st.text_input("Benchmark Scheme Code", value=DEFAULT_BENCHMARK_CODE)

# --- Main Application Logic ---
all_nav_data = {}
for name, code in DEFAULT_SECTOR_FUNDS.items():
    nav_series = get_nav_history(code)
    if nav_series is not None:
        all_nav_data[name] = nav_series

if not all_nav_data:
    st.error("Could not fetch data for any sector funds. Please check the scheme codes or try again later.")
    st.stop()

# Use the full, unfiltered price history for calculations
prices = pd.DataFrame(all_nav_data).dropna(how='all').ffill()

if prices.empty:
    st.error(f"No fund data available after the selected start date. Please choose an earlier date.")
    st.stop()

# Pass the user's selected start date to the calculation function
start_date = pd.to_datetime(start_date_input)
monthly_scores, monthly_weights = calculate_momentum_scores(prices, lookback_periods, sma_period, start_date)

if not monthly_weights:
    st.warning("Could not generate any investment signals. This could be due to insufficient data for the selected start date and lookback periods. Try an earlier start date or adjust lookback parameters.")
    st.stop()

portfolio_nav = run_backtest(prices, monthly_weights)

if portfolio_nav.empty:
    st.error("Backtest resulted in an empty portfolio. This might be due to insufficient data for the selected timeframe.")
    st.stop()
    
benchmark_nav_raw = get_nav_history(benchmark_code)
if benchmark_nav_raw is None:
    st.error("Failed to load benchmark data. Cannot proceed with comparison.")
    st.stop()

# Align benchmark to the portfolio's timeline and normalize
benchmark_nav = benchmark_nav_raw.reindex(portfolio_nav.index).ffill()
if benchmark_nav.dropna().empty:
    st.error("Benchmark has no data for the backtest period. Cannot perform comparison.")
    st.stop()

benchmark_nav = (benchmark_nav / benchmark_nav.dropna().iloc[0]) * 100

# --- Display Results ---
st.header("🚀 Performance Summary")
portfolio_kpis = calculate_kpis(portfolio_nav)
benchmark_kpis = calculate_kpis(benchmark_nav)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Strategy CAGR", f"{portfolio_kpis['CAGR']:.2%}", f"{(portfolio_kpis['CAGR'] - benchmark_kpis['CAGR']):.2%}")
    st.metric("Benchmark CAGR", f"{benchmark_kpis['CAGR']:.2%}")
with col2:
    st.metric("Strategy Max Drawdown", f"{portfolio_kpis['Max Drawdown']:.2%}", f"{(portfolio_kpis['Max Drawdown'] - benchmark_kpis['Max Drawdown']):.2%}", delta_color="inverse")
    st.metric("Benchmark Max Drawdown", f"{benchmark_kpis['Max Drawdown']:.2%}")
with col3:
    st.metric("Strategy Sharpe Ratio", f"{portfolio_kpis['Sharpe Ratio']:.2f}", f"{portfolio_kpis['Sharpe Ratio'] - benchmark_kpis['Sharpe Ratio']:.2f}")
    st.metric("Benchmark Sharpe Ratio", f"{benchmark_kpis['Sharpe Ratio']:.2f}")


st.header("📈 Performance & Drawdown Chart")
plot_performance_chart(portfolio_nav, benchmark_nav, portfolio_kpis['Drawdown Series'], benchmark_kpis['Drawdown Series'])


st.header("📊 Current Allocation & Momentum Scores")
col1, col2 = st.columns([0.4, 0.6])

with col1:
    st.subheader("Current Recommended Allocation")
    if monthly_weights:
        last_rebalance_date = max(monthly_weights.keys())
        current_weights = monthly_weights[last_rebalance_date]
        
        if current_weights:
            current_alloc_df = pd.DataFrame.from_dict(current_weights, orient='index', columns=['Weight'])
            current_alloc_df.index.name = 'Sector'
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=current_alloc_df.index, values=current_alloc_df['Weight'], 
                hole=.3, pull=[0.05] * len(current_alloc_df)
            )])
            fig_pie.update_layout(
                title_text=f"As of {last_rebalance_date.strftime('%b %Y')}",
                title_x=0.5, showlegend=False, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("The strategy currently recommends holding cash (no investment).")
    else:
        st.info("No allocation data to display.")

with col2:
    st.subheader("Latest Momentum Scores")
    if monthly_scores:
        last_date, last_scores = monthly_scores[-1]
        st.dataframe(
            last_scores.sort_values("momentum_score", ascending=False)
            .style.background_gradient(cmap="viridis", subset=["momentum_score"])
            .format("{:.2f}")
        )

with st.expander("📅 View Monthly Returns"):
    monthly_returns = portfolio_nav.resample("M").ffill().pct_change().to_frame("Portfolio")
    monthly_returns["Benchmark"] = benchmark_nav.resample("M").ffill().pct_change()
    monthly_returns.index = monthly_returns.index.strftime("%b-%Y")
    
    st.dataframe(
        monthly_returns.style.format("{:.2%}").background_gradient(cmap="RdYlGn", axis=0)
    )
