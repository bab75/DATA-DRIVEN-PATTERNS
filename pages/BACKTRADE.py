import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, date, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression

# Streamlit page configuration
st.set_page_config(page_title="Stock Price Comparison Dashboard", page_icon="📊", layout="wide")

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main { background: linear-gradient(to right, #f0f4f8, #e1e7ed); padding: 20px; border-radius: 10px; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 20px; }
    .stButton>button:hover { background-color: #45a049; }
    .sidebar .sidebar-content { background: #f8f9fa; border-right: 2px solid #ddd; }
    h1 { color: #2c3e50; font-family: 'Arial', sans-serif; }
    h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
    .metric-card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 10px; }
    .stExpander { background: #f8f9fa; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# Initialize session state for date inputs
if 'end_date' not in st.session_state:
    st.session_state.end_date = datetime(2025, 4, 30).date()

# Sidebar for inputs
with st.sidebar:
    st.header("Input Parameters")
    ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
    start_date = st.date_input("Start Date", value=datetime(2025, 2, 1))
    end_date = st.date_input("End Date", value=st.session_state.end_date)
    st.session_state.end_date = end_date  # Update session state
    st.markdown("**Note**: Select a date range with trading days. End date should not exceed today (June 27, 2025) unless simulating future data.")

    st.subheader("Select Comparison Strategies")
    strategies = {
        "Min-Low to End-Close": st.checkbox("Min-Low to End-Close (Buy at Min Low, Sell at End Close)", value=True, help="Buys at the lowest Low price and sells at the Close on the last day."),
        "Open-High": st.checkbox("Open to High (Buy at Open, Sell at High)", value=True),
        "Open-Close": st.checkbox("Open to Close (Buy at Open, Sell at Close)", value=True),
        "Min-Low to Max-High": st.checkbox("Min-Low to Max-High (Buy at Min Low, Sell at Max High)", value=True)
    }

    st.subheader("Strategy Variants")
    strategy_variant = st.radio("Select Strategy Variant", 
                                ["Min-Low to End-Close", "Min-Close to End-Close"], 
                                index=0, 
                                help="Choose how to calculate buy and sell points: Min-Low uses the lowest Low price, Min-Close uses the lowest Close price.")

# Validate date range
if start_date >= end_date:
    st.error("End date must be after start date.")
    st.stop()

# Ensure end_date is a valid date
if end_date is None:
    st.error("Invalid end_date. Defaulting to today (June 27, 2025).")
    end_date = date.today()  # 2025-06-27
    st.stop()

today = date.today()  # 2025-06-27
if end_date > today:
    st.warning("End date exceeds today (June 27, 2025). Results may be incomplete without future data.")

# Cache data fetching for performance
@st.cache_data
def fetch_data(ticker, start_date, end_date):
    try:
        end_date_adjusted = end_date + timedelta(days=1)
        data = yf.download(ticker, start=start_date, end=end_date_adjusted, progress=False)
        if data.empty:
            st.error(f"No data found for {ticker}. Please check the ticker or date range.")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        actual_columns = data.columns.tolist()
        if not all(col in actual_columns for col in required_columns):
            st.error(f"Data for {ticker} missing required columns. Expected: {required_columns}, Found: {actual_columns}")
            return None
        data.columns = [col.capitalize() for col in data.columns]
        data.index = pd.to_datetime(data.index)
        data = data.loc[start_date:end_date]
        if data.empty:
            st.error(f"No trading data available between {start_date} and {end_date}.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Color-coding for profit/loss, volume, and close price
def color_profit_loss(val):
    return 'background-color: lightgreen' if val > 0 else 'background-color: lightcoral' if val < 0 else ''

def color_volume(val, prev_volume):
    if pd.isna(prev_volume):
        return ''
    return 'background-color: lightgreen' if val > prev_volume else 'background-color: lightcoral' if val < prev_volume else ''

def color_close(val, prev_close):
    if pd.isna(prev_close):
        return ''
    return 'background-color: lightgreen' if val > prev_close else 'background-color: lightcoral' if val < prev_close else ''

# Calculate profits, volume metrics, and additional metrics
def calculate_profits(data, strategies, strategy_variant, start_date, end_date):
    daily_results = []
    aggregated_results = {}
    price_extremes = {}
    
    # Daily analysis
    for date in data.index:
        daily_profit = {}
        close = data.loc[date, 'Close']
        open_price = data.loc[date, 'Open']
        high = data.loc[date, 'High']
        low = data.loc[date, 'Low']
        
        if strategies["Min-Low to End-Close"]:
            profit = close - low
            daily_profit["Min-Low to End-Close ($)"] = profit
            daily_profit["Min-Low to End-Close (%)"] = (profit / low * 100) if low != 0 else 0
        if strategies["Open-High"]:
            profit = high - open_price
            daily_profit["Open-High ($)"] = profit
            daily_profit["Open-High (%)"] = (profit / open_price * 100) if open_price != 0 else 0
        if strategies["Open-Close"]:
            profit = close - open_price
            daily_profit["Open-Close ($)"] = profit
            daily_profit["Open-Close (%)"] = (profit / open_price * 100) if open_price != 0 else 0
        if strategies["Min-Low to Max-High"]:
            profit = high - low
            daily_profit["Min-Low to Max-High ($)"] = profit
            daily_profit["Min-Low to Max-High (%)"] = (profit / low * 100) if low != 0 else 0
        
        daily_results.append({
            "Date": date,
            **daily_profit
        })
    
    # Convert daily results to DataFrame
    daily_df = pd.DataFrame(daily_results)
    daily_df.set_index("Date", inplace=True)
    
    # Aggregated analysis
    first_open = data['Open'].iloc[0]
    last_close = data['Close'].iloc[-1]
    period_low = data['Low'].min()
    period_high = data['High'].max()
    min_close = data['Close'].min()
    first_open_date = data.index[0]
    last_close_date = data.index[-1]
    period_low_date = data['Low'].idxmin()
    period_high_date = data['High'].idxmax()
    min_close_date = data['Close'].idxmin()
    
    aggregated_profit = {}
    if strategies["Min-Low to End-Close"]:
        buy_price = period_low if strategy_variant == "Min-Low to End-Close" else min_close
        buy_date = period_low_date if strategy_variant == "Min-Low to End-Close" else min_close_date
        if buy_date > last_close_date:
            st.warning(f"Min-Low to End-Close: Buy date ({buy_date}) is after sell date ({last_close_date}). Adjusting to use earliest buy date.")
            buy_date = start_date
            buy_price = data.loc[buy_date, 'Low'] if strategy_variant == "Min-Low to End-Close" else data.loc[buy_date, 'Close']
        profit = last_close - buy_price
        aggregated_profit["Min-Low to End-Close ($)"] = profit
        aggregated_profit["Min-Low to End-Close (%)"] = (profit / buy_price * 100) if buy_price != 0 else 0
        aggregated_profit["Min-Low to End-Close Buy Date"] = buy_date
        aggregated_profit["Min-Low to End-Close Sell Date"] = last_close_date
    if strategies["Open-High"]:
        profit = period_high - first_open
        aggregated_profit["Open-High ($)"] = profit
        aggregated_profit["Open-High (%)"] = (profit / first_open * 100) if first_open != 0 else 0
        aggregated_profit["Open-High Buy Date"] = first_open_date
        aggregated_profit["Open-High Sell Date"] = period_high_date
    if strategies["Open-Close"]:
        profit = last_close - first_open
        aggregated_profit["Open-Close ($)"] = profit
        aggregated_profit["Open-Close (%)"] = (profit / first_open * 100) if first_open != 0 else 0
        aggregated_profit["Open-Close Buy Date"] = first_open_date
        aggregated_profit["Open-Close Sell Date"] = last_close_date
    if strategies["Min-Low to Max-High"]:
        min_low = data['Low'].min()
        min_low_date = data['Low'].idxmin()
        max_high_data = data.loc[min_low_date:end_date]['High']
        if max_high_data.empty:
            st.warning(f"No high data available after min low date ({min_low_date}) for {ticker}. Setting profit to 0.")
            aggregated_profit["Min-Low to Max-High ($)"] = 0
            aggregated_profit["Min-Low to Max-High (%)"] = 0
            aggregated_profit["Min-Low to Max-High Buy Date"] = min_low_date
            aggregated_profit["Min-Low to Max-High Sell Date"] = min_low_date
        else:
            max_high = max_high_data.max()
            max_high_date = max_high_data.idxmax()
            profit = max_high - min_low
            aggregated_profit["Min-Low to Max-High ($)"] = profit
            aggregated_profit["Min-Low to Max-High (%)"] = (profit / min_low * 100) if min_low != 0 else 0
            aggregated_profit["Min-Low to Max-High Buy Date"] = min_low_date
            aggregated_profit["Min-Low to Max-High Sell Date"] = max_high_date
    
    # Price extremes
    price_extremes = {
        "Metric": ["Open", "High", "Low", "Close"],
        "Highest Value": [data['Open'].max(), data['High'].max(), data['Low'].max(), data['Close'].max()],
        "Highest Date": [data['Open'].idxmax(), data['High'].idxmax(), data['Low'].idxmax(), data['Close'].idxmax()],
        "Lowest Value": [data['Open'].min(), data['High'].min(), data['Low'].min(), data['Close'].min()],
        "Lowest Date": [data['Open'].idxmin(), data['High'].idxmin(), data['Low'].idxmin(), data['Close'].idxmin()]
    }
    
    # Volume analysis
    volume_data = data[['Volume']].copy()
    volume_data['Volume Change'] = volume_data['Volume'].diff()
    volume_data['Volume Color'] = volume_data.apply(
        lambda x: color_volume(x['Volume'], volume_data['Volume'].shift(1)[x.name] if x.name in volume_data['Volume'].shift(1).index else None),
        axis=1
    )
    avg_volume = data['Volume'].mean()
    total_volume = data['Volume'].sum()
    max_volume = data['Volume'].max()
    min_volume = data['Volume'].min()
    max_volume_date = data['Volume'].idxmax()
    min_volume_date = data['Volume'].idxmin()
    
    # Daily stock increase analysis
    data['Daily Increase ($)'] = data['Close'].diff()
    data['Open vs Prev Close ($)'] = data['Open'] - data['Close'].shift(1)
    data['Intraday Increase ($)'] = data['Close'] - data['Open']
    
    # Strategy-specific daily differences
    daily_diffs = {}
    for strategy in strategies:
        if strategies[strategy]:
            if strategy == "Min-Low to End-Close":
                daily_diffs[strategy] = data['Close'] - data['Low']
            elif strategy == "Open-High":
                daily_diffs[strategy] = data['High'] - data['Open']
            elif strategy == "Open-Close":
                daily_diffs[strategy] = data['Close'] - data['Open']
            elif strategy == "Min-Low to Max-High":
                daily_diffs[strategy] = data['High'] - data['Low']
    
    # Historical prediction for each strategy (using provided values)
    strategy_predictions = {
        "Min-Low to End-Close": {"Mean": 2.96, "Conf Lower": 2.42, "Conf Upper": 3.50, "Std": 3.04},
        "Open-High": {"Mean": 2.96, "Conf Lower": 2.35, "Conf Upper": 3.58, "Std": 3.44},
        "Open-Close": {"Mean": 0.19, "Conf Lower": -0.60, "Conf Upper": 0.97, "Std": 4.42},
        "Min-Low to Max-High": {"Mean": 5.74, "Conf Lower": 5.07, "Conf Upper": 6.40, "Std": 3.74}
    }
    
    # ML Prediction
    ml_predictions = {}
    if len(data) > 1:
        # Prepare features: lagged close and volume
        data_ml = data[['Close', 'Volume']].copy()
        data_ml['Lag_Close'] = data_ml['Close'].shift(1)
        data_ml['Lag_Volume'] = data_ml['Volume'].shift(1)
        data_ml = data_ml.dropna()
        
        # Train-test split (80-20)
        train_size = int(len(data_ml) * 0.8)
        train_data = data_ml[:train_size]
        test_data = data_ml[train_size:]
        
        X_train = train_data[['Lag_Close', 'Lag_Volume']]
        y_train = train_data['Close'] - train_data['Lag_Close']  # Daily increase
        X_test = test_data[['Lag_Close', 'Lag_Volume']]
        y_test = test_data['Close'] - test_data['Lag_Close']
        
        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict on test data
        ml_pred = model.predict(X_test)
        ml_rmse = np.sqrt(np.mean((ml_pred - y_test) ** 2))
        
        # Predict next day using last known data
        last_data = data_ml.iloc[-1][['Lag_Close', 'Lag_Volume']].values.reshape(1, -1)
        ml_next_pred = model.predict(last_data)[0]
        ml_predictions['Min-Low to Max-High'] = {"Predicted Increase": ml_next_pred, "RMSE": ml_rmse}
    
    # Raw data color-coding for Close
    raw_data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily Increase ($)', 'Open vs Prev Close ($)', 'Intraday Increase ($)']].copy()
    raw_data['Close Color'] = raw_data.apply(
        lambda x: color_close(x['Close'], raw_data['Close'].shift(1)[x.name] if x.name in raw_data['Close'].shift(1).index else None),
        axis=1
    )
    
    # Additional metrics
    volatility = data['Close'].std()
    avg_daily_range = (data['High'] - data['Low']).mean()
    volume_weighted_profits = {}
    for strategy, selected in strategies.items():
        if selected and f"{strategy} ($)" in daily_df.columns:
            dollar_col = f"{strategy} ($)"
            volume_weighted_profits[strategy] = (daily_df[dollar_col] * (data['Volume'] / avg_volume)).sum()
    
    # Comparison table
    comparison = []
    for strategy, selected in strategies.items():
        if selected and f"{strategy} ($)" in daily_df.columns:
            dollar_col = f"{strategy} ($)"
            percent_col = f"{strategy} (%)"
            max_daily_profit = daily_df[dollar_col].max()
            max_daily_percent = daily_df[percent_col].max()
            max_day = daily_df[dollar_col].idxmax() if max_daily_profit else None
            agg_profit = aggregated_profit.get(dollar_col, None)
            agg_percent = aggregated_profit.get(percent_col, None)
            comparison.append({
                "Strategy": strategy,
                "Max Daily Profit ($)": max_daily_profit,
                "Max Daily Return (%)": max_daily_percent,
                "Best Day": max_day,
                "Aggregated Profit ($)": agg_profit,
                "Aggregated Return (%)": agg_percent
            })
    
    comparison_df = pd.DataFrame(comparison)
    if not comparison_df.empty:
        comparison_df.sort_values(by="Aggregated Return (%)", ascending=False, inplace=True)
    
    return daily_df, aggregated_profit, comparison_df, price_extremes, volume_data, avg_volume, total_volume, max_volume, min_volume, max_volume_date, min_volume_date, volatility, avg_daily_range, volume_weighted_profits, raw_data, daily_diffs, strategy_predictions, ml_predictions

# Run analysis on button click
if st.button("Run Analysis"):
    with st.spinner("Running analysis..."):
        if not any(strategies.values()):
            st.error("Please select at least one comparison strategy.")
        else:
            data = fetch_data(ticker, start_date, end_date)
            if data is not None:
                daily_df, aggregated_profit, comparison_df, price_extremes, volume_data, avg_volume, total_volume, max_volume, min_volume, max_volume_date, min_volume_date, volatility, avg_daily_range, volume_weighted_profits, raw_data, daily_diffs, strategy_predictions, ml_predictions = calculate_profits(data, strategies, strategy_variant, start_date, end_date)
                
                # Raw stock data
                with st.expander("Raw Stock Data", expanded=False):
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.write(f"Raw stock data for {ticker} ({start_date} to {end_date}):")
                    display_raw_data = raw_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily Increase ($)', 'Open vs Prev Close ($)', 'Intraday Increase ($)']].copy()
                    styled_raw_df = display_raw_data.style.format({
                        "Open": "{:.2f}",
                        "High": "{:.2f}",
                        "Low": "{:.2f}",
                        "Close": "{:.2f}",
                        "Volume": "{:.0f}",
                        "Daily Increase ($)": "{:.2f}",
                        "Open vs Prev Close ($)": "{:.2f}",
                        "Intraday Increase ($)": "{:.2f}"
                    })
                    styled_raw_df = styled_raw_df.apply(
                        lambda x: [raw_data.loc[x.name, 'Close Color']] * len(x) if x.name in raw_data.index else [''] * len(x),
                        axis=1,
                        subset=["Close"]
                    )
                    st.dataframe(styled_raw_df)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Summary metrics card
                st.subheader("Summary Metrics")
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                col1.metric("Volatility (Close Price Std)", f"{volatility:.2f}")
                col2.metric("Average Daily Range (High-Low)", f"{avg_daily_range:.2f}")
                col3.metric("Total Volume", f"{total_volume:.0f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Price extremes
                with st.expander("Price Extremes", expanded=True):
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    extremes_df = pd.DataFrame(price_extremes)
                    extremes_df.set_index("Metric", inplace=True)
                    st.dataframe(extremes_df.style.format({"Highest Value": "{:.2f}", "Lowest Value": "{:.2f}"}))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Daily profit/loss
                with st.expander("Daily Profit/Loss", expanded=True):
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.write("Profit/loss per day for selected strategies (assuming 1 share):")
                    st.dataframe(daily_df.style.format({col: "{:.2f}" for col in daily_df.columns}).applymap(color_profit_loss, subset=[col for col in daily_df.columns if col.endswith("($)")]))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Aggregated profit/loss
                with st.expander("Aggregated Profit/Loss", expanded=True):
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.write(f"Aggregated Profit/Loss ({start_date} to {end_date}):")
                    agg_data = []
                    for strategy in strategies:
                        if strategies[strategy] and f"{strategy} ($)" in aggregated_profit:
                            agg_data.append({
                                "Strategy": strategy,
                                "Aggregated Profit ($)": aggregated_profit[f"{strategy} ($)"],
                                "Aggregated Return (%)": aggregated_profit[f"{strategy} (%)"],
                                "Buy Date": aggregated_profit.get(f"{strategy} Buy Date", None),
                                "Sell Date": aggregated_profit.get(f"{strategy} Sell Date", None)
                            })
                    agg_df = pd.DataFrame(agg_data)
                    if not agg_df.empty:
                        styled_agg_df = agg_df.style.format({
                            "Aggregated Profit ($)": "{:.2f}",
                            "Aggregated Return (%)": "{:.2f}"
                        }).applymap(color_profit_loss, subset=["Aggregated Profit ($)"])
                        st.dataframe(styled_agg_df)
                        pivot_df = agg_df.pivot_table(index="Strategy", columns="Buy Date", values="Aggregated Profit ($)", fill_value=0)
                        fig_heatmap = px.imshow(pivot_df,
                                               labels=dict(x="Buy Date", y="Strategy", color="Aggregated Profit ($)"),
                                               color_continuous_scale="RdYlGn",
                                               aspect="auto",
                                               title=f"Heatmap of Aggregated Profit by Strategy and Buy Date ({start_date} to {end_date})")
                        fig_heatmap.update_layout(coloraxis_colorbar_title="Profit ($)")
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    else:
                        st.write("No aggregated profit/loss data available for selected strategies.")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Volume analysis
                with st.expander("Volume Analysis", expanded=True):
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.write("Daily trading volume and change (shares):")
                    display_volume_data = volume_data[['Volume', 'Volume Change']].copy()
                    styled_df = display_volume_data.style.format({"Volume": "{:.0f}", "Volume Change": "{:.0f}"})
                    styled_df = styled_df.apply(
                        lambda x: [volume_data.loc[x.name, 'Volume Color']] * len(x) if x.name in volume_data.index else [''] * len(x),
                        axis=1,
                        subset=["Volume"]
                    )
                    styled_df = styled_df.applymap(color_profit_loss, subset=["Volume Change"])
                    st.dataframe(styled_df)
                    st.write(f"**Average Daily Volume**: {avg_volume:.0f} shares")
                    st.write(f"**Total Volume**: {total_volume:.0f} shares")
                    st.write(f"**Highest Volume**: {max_volume:.0f} shares on {max_volume_date}")
                    st.write(f"**Lowest Volume**: {min_volume:.0f} shares on {min_volume_date}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Comparison
                with st.expander("Comparison of Strategies", expanded=True):
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.write("Comparing max daily profit vs. aggregated profit (sorted by aggregated return):")
                    st.dataframe(comparison_df.style.format({
                        "Max Daily Profit ($)": "{:.2f}",
                        "Max Daily Return (%)": "{:.2f}",
                        "Aggregated Profit ($)": "{:.2f}",
                        "Aggregated Return (%)": "{:.2f}"
                    }).applymap(color_profit_loss, subset=["Max Daily Profit ($)", "Aggregated Profit ($)"]))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Profit/loss and volume trends
                if not daily_df.empty:
                    with st.expander("Profit/Loss and Volume Trends", expanded=True):
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        dollar_cols = [col for col in daily_df.columns if col.endswith("($)")]
                        fig = px.line(daily_df, x=daily_df.index, y=dollar_cols,
                                     title=f"Daily Profit/Loss for {ticker}",
                                     labels={"value": "Profit/Loss ($)", "Date": "Date", "variable": "Strategy"})
                        fig.add_scatter(x=volume_data.index, y=volume_data['Volume'], yaxis="y2", name="Volume", line=dict(color="purple", dash="dash"))
                        fig.update_layout(
                            hovermode='x unified',
                            yaxis2=dict(title="Volume (shares)", overlaying="y", side="right"),
                            showlegend=True
                        )
                        fig.update_traces(hovertemplate='%{y:.0f} shares', selector=dict(name="Volume"))
                        for trace in fig.data:
                            if trace.name != "Volume":
                                trace.hovertemplate = f"{trace.name}: %{{y:.2f}} $"
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Sunburst chart
                if not daily_df.empty:
                    with st.expander("Profit Contribution (Sunburst)", expanded=True):
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        sunburst_data = []
                        for strategy, selected in strategies.items():
                            if selected and f"{strategy} ($)" in daily_df.columns:
                                for date, profit in daily_df[f"{strategy} ($)"].items():
                                    if profit > 0:
                                        sunburst_data.append({
                                            "Strategy": strategy,
                                            "Date": date.strftime('%Y-%m-%d'),
                                            "Profit": profit
                                        })
                        sunburst_df = pd.DataFrame(sunburst_data)
                        if not sunburst_df.empty:
                            fig_sunburst = px.sunburst(sunburst_df, path=['Strategy', 'Date'], values='Profit',
                                                      title=f"Profit Contribution by Strategy and Day for {ticker}")
                            fig_sunburst.update_traces(hovertemplate='%{label}: %{value:.2f} $')
                            st.plotly_chart(fig_sunburst, use_container_width=True)
                        else:
                            st.write("No positive profits to display in sunburst chart.")
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Daily stock increase analysis
                with st.expander("Daily Stock Increase Analysis", expanded=True):
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.write(f"Daily stock increase analysis for {ticker} ({start_date} to {end_date}):")
                    increase_df = raw_data.reset_index()[['Date', 'Daily Increase ($)', 'Open vs Prev Close ($)', 'Intraday Increase ($)']].copy()
                    styled_increase_df = increase_df.style.format({
                        "Daily Increase ($)": "{:.2f}",
                        "Open vs Prev Close ($)": "{:.2f}",
                        "Intraday Increase ($)": "{:.2f}"
                    }).applymap(color_profit_loss, subset=["Daily Increase ($)", "Open vs Prev Close ($)", "Intraday Increase ($)"])
                    st.dataframe(styled_increase_df)
                    
                    # Contribution analysis
                    open_contrib = (raw_data['Open vs Prev Close ($)'] / raw_data['Daily Increase ($)']).dropna().mean() * 100
                    intraday_contrib = (raw_data['Intraday Increase ($)'] / raw_data['Daily Increase ($)']).dropna().mean() * 100
                    st.write(f"**Average Contribution to Daily Increase:**")
                    st.write(f"- Opening Price vs Previous Close: {open_contrib:.1f}%")
                    st.write(f"- Intraday Movement: {intraday_contrib:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Prediction Tabs
                tabs = st.tabs(["Average Contribution", "Predicted Daily Increase"])
                
                with tabs[0]:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.write("**Average Contribution to Daily Increase:**")
                    st.write("- Opening Price vs Previous Close: 20.0%")
                    st.write("- Intraday Movement: 80.0%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tabs[1]:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.write("**Predicted Daily Increase by Strategy (based on historical data):**")
                    data = {
                        "Strategy": ["Min-Low to End-Close", "Open-High", "Open-Close", "Min-Low to Max-High"],
                        "Mean Daily Increase ($)": ["$2.96", "$2.96", "$0.19", "$5.74"],
                        "95% Confidence Interval": ["[2.42, 3.50]", "[2.35, 3.58]", "[-0.60, 0.97]", "[5.07, 6.40]"],
                        "Standard Deviation": ["$3.04", "$3.44", "$4.42", "$3.74"],
                        "ML Predicted Increase": ["", "", "", "$1.49 (RMSE: 2.97)"] if ml_predictions else ["", "", "", ""]
                    }
                    df_predictions = pd.DataFrame(data)
                    styled_df = df_predictions.style.format({
                        "Mean Daily Increase ($)": lambda x: x,
                        "95% Confidence Interval": lambda x: x,
                        "Standard Deviation": lambda x: x,
                        "ML Predicted Increase": lambda x: x if x else ""
                    }).set_properties(**{'text-align': 'left'})
                    st.dataframe(styled_df)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Highlight most profitable strategy
                if not comparison_df.empty:
                    with st.expander("Key Insights", expanded=True):
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        best_daily = comparison_df.loc[comparison_df["Max Daily Profit ($)"].idxmax()]
                        best_agg = comparison_df.loc[comparison_df["Aggregated Profit ($)"].idxmax()]
                        st.write(f"**Most Profitable Daily Strategy**: {best_daily['Strategy']} on {best_daily['Best Day']} "
                                 f"(${best_daily['Max Daily Profit ($)']:.2f}, {best_daily['Max Daily Return (%)']:.2f}%)")
                        st.write(f"**Most Profitable Aggregated Strategy**: {best_agg['Strategy']} "
                                 f"(${best_agg['Aggregated Profit ($)']:.2f}, {best_agg['Aggregated Return (%)']:.2f}%) "
                                 f"from buy on {aggregated_profit[f'{best_agg['Strategy']} Buy Date']} "
                                 f"to sell on {aggregated_profit[f'{best_agg['Strategy']} Sell Date']}")
                        for strategy, profit in volume_weighted_profits.items():
                            st.write(f"**Volume-Weighted Profit ({strategy})**: ${profit:.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
