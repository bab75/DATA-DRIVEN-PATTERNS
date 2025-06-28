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
    st.session_state.end_date = datetime(2025, 6, 28).date()  # Current date

# Sidebar for inputs
with st.sidebar:
    st.header("Input Parameters")
    ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
    start_date = st.date_input("Start Date", value=datetime(2025, 2, 1))
    end_date = st.date_input("End Date", value=st.session_state.end_date)
    st.session_state.end_date = end_date
    st.markdown("**Note**: Select a date range with trading days. End date should not exceed today (June 28, 2025).")

    st.subheader("Select Comparison Strategies")
    strategies = {
        "Min-Low to End-Close": st.checkbox("Min-Low to End-Close (Buy at Min Low, Sell at End Close)", value=True, help="Measures gap from daily Low to Close to gauge market sentiment."),
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
    st.error("End date must be after start_date.")
    st.stop()

if end_date is None:
    st.error("Invalid end_date. Defaulting to today (June 28, 2025).")
    end_date = date.today()
    st.stop()

today = date.today()
if end_date > today:
    st.error("End date cannot exceed today (June 28, 2025). Please adjust the date range.")
    st.stop()

# Cache data fetching for performance
@st.cache_data
def fetch_data(ticker, start_date, end_date):
    try:
        end_date_adjusted = end_date + timedelta(days=1)
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(start=start_date, end=end_date_adjusted)
        if data.empty:
            st.error(f"No data found for {ticker} between {start_date} and {end_date}. Ensure the date range includes trading days.")
            return None, None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        actual_columns = data.columns.tolist()
        if not all(col in actual_columns for col in required_columns):
            st.error(f"Missing required columns. Expected: {required_columns}, Found: {actual_columns}")
            return None, None
        data.columns = [col.capitalize() for col in data.columns]
        print(f"Columns after fetch: {data.columns.tolist()}")  # Debug print
        # Ensure timezone-naive index
        data.index = pd.to_datetime(data.index).tz_localize(None)
        print(f"Index timezone after normalization: {data.index.tz}")  # Debug timezone
        data = data.loc[start_date:end_date]
        if data.empty:
            st.error(f"No trading data available between {start_date} and {end_date}.")
            return None, None
        
        # Fetch company info and fundamentals
        info = ticker_obj.info
        company_details = {
            'Name': info.get('longName', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
            'P/E Ratio': info.get('trailingPE', 'N/A'),
            'EPS': info.get('trailingEps', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A'),
            'Beta': info.get('beta', 'N/A')
        }
        print(f"Returning data: {type(data)}, shape: {data.shape}, Company info: {company_details}")  # Debug return value
        return data, company_details
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, None

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
    
    # Check if data is valid
    if data is None or data.empty or not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        st.error("Invalid or empty data. Cannot calculate profits.")
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
    
    for date in data.index:
        daily_profit = {}
        close = data.loc[date, 'Close']
        open_price = data.loc[date, 'Open']
        high = data.loc[date, 'High']
        low = data.loc[date, 'Low']
        
        if strategies["Min-Low to End-Close"]:
            gap = close - low
            daily_profit["Min-Low to End-Close ($)"] = gap
            daily_profit["Min-Low to End-Close (%)"] = (gap / low * 100) if low != 0 else 0
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
    
    daily_df = pd.DataFrame(daily_results)
    daily_df.set_index("Date", inplace=True)
    
    first_open = data['Open'].iloc[0] if not data['Open'].empty else None
    last_close = data['Close'].iloc[-1] if not data['Close'].empty else None
    period_low = data['Low'].min() if not data['Low'].empty else None
    period_high = data['High'].max() if not data['High'].empty else None
    min_close = data['Close'].min() if not data['Close'].empty else None
    first_open_date = data.index[0] if not data.index.empty else None
    last_close_date = data.index[-1] if not data.index.empty else None
    period_low_date = data['Low'].idxmin() if not data['Low'].empty else None
    period_high_date = data['High'].idxmax() if not data['High'].empty else None
    min_close_date = data['Close'].idxmin() if not data['Close'].empty else None
    
    aggregated_profit = {}
    if strategies["Min-Low to End-Close"] and period_low is not None and last_close is not None:
        buy_price = period_low if strategy_variant == "Min-Low to End-Close" else min_close
        buy_date = period_low_date if strategy_variant == "Min-Low to End-Close" else min_close_date
        if buy_date and last_close_date and buy_date > last_close_date:
            st.warning(f"Min-Low to End-Close: Buy date ({buy_date}) is after sell date ({last_close_date}). Adjusting to use earliest buy date.")
            buy_date = start_date
            buy_price = data.loc[buy_date, 'Low'] if strategy_variant == "Min-Low to End-Close" and buy_date in data.index else data.loc[buy_date, 'Close']
        if buy_price is not None and last_close is not None:
            profit = last_close - buy_price
            aggregated_profit["Min-Low to End-Close ($)"] = profit
            aggregated_profit["Min-Low to End-Close (%)"] = (profit / buy_price * 100) if buy_price != 0 else 0
            aggregated_profit["Min-Low to End-Close Buy Date"] = buy_date
            aggregated_profit["Min-Low to End-Close Sell Date"] = last_close_date
    if strategies["Open-High"] and period_high is not None and first_open is not None:
        profit = period_high - first_open
        aggregated_profit["Open-High ($)"] = profit
        aggregated_profit["Open-High (%)"] = (profit / first_open * 100) if first_open != 0 else 0
        aggregated_profit["Open-High Buy Date"] = first_open_date
        aggregated_profit["Open-High Sell Date"] = period_high_date
    if strategies["Open-Close"] and last_close is not None and first_open is not None:
        profit = last_close - first_open
        aggregated_profit["Open-Close ($)"] = profit
        aggregated_profit["Open-Close (%)"] = (profit / first_open * 100) if first_open != 0 else 0
        aggregated_profit["Open-Close Buy Date"] = first_open_date
        aggregated_profit["Open-Close Sell Date"] = last_close_date
    if strategies["Min-Low to Max-High"] and period_low is not None and period_high is not None:
        min_low = period_low
        min_low_date = period_low_date
        max_high_data = data.loc[min_low_date:end_date]['High'] if min_low_date in data.index else pd.Series()
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
    
    # Only populate price_extremes if data is valid
    if not data.empty and all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
        price_extremes = {
            "Metric": ["Open", "High", "Low", "Close"],
            "Highest Value": [data['Open'].max(), data['High'].max(), data['Low'].max(), data['Close'].max()],
            "Highest Date": [data['Open'].idxmax(), data['High'].idxmax(), data['Low'].idxmax(), data['Close'].idxmax()],
            "Lowest Value": [data['Open'].min(), data['High'].min(), data['Low'].min(), data['Close'].min()],
            "Lowest Date": [data['Open'].idxmin(), data['High'].idxmin(), data['Low'].idxmin(), data['Close'].idxmin()]
        }
    
    volume_data = data[['Volume']].copy() if 'Volume' in data.columns else pd.DataFrame()
    if not volume_data.empty:
        volume_data['Volume Change'] = volume_data['Volume'].diff()
        volume_data['Volume Color'] = volume_data.apply(
            lambda x: color_volume(x['Volume'], volume_data['Volume'].shift(1)[x.name] if x.name in volume_data['Volume'].shift(1).index else None),
            axis=1
        )
    avg_volume = data['Volume'].mean() if 'Volume' in data.columns and not data['Volume'].empty else 0
    total_volume = data['Volume'].sum() if 'Volume' in data.columns and not data['Volume'].empty else 0
    max_volume = data['Volume'].max() if 'Volume' in data.columns and not data['Volume'].empty else None
    min_volume = data['Volume'].min() if 'Volume' in data.columns and not data['Volume'].empty else None
    max_volume_date = data['Volume'].idxmax() if 'Volume' in data.columns and not data['Volume'].empty else None
    min_volume_date = data['Volume'].idxmin() if 'Volume' in data.columns and not data['Volume'].empty else None
    
    if not data.empty:
        data['Daily Increase ($)'] = data['Close'].diff()
        data['Open vs Prev Close ($)'] = data['Open'] - data['Close'].shift(1)
        data['Intraday Increase ($)'] = data['Close'] - data['Open']
    
    daily_diffs = {}
    for strategy in strategies:
        if strategies[strategy] and not data.empty:
            if strategy == "Min-Low to End-Close":
                daily_diffs[strategy] = data['Close'] - data['Low']
            elif strategy == "Open-High":
                daily_diffs[strategy] = data['High'] - data['Open']
            elif strategy == "Open-Close":
                daily_diffs[strategy] = data['Close'] - data['Open']
            elif strategy == "Min-Low to Max-High":
                daily_diffs[strategy] = data['High'] - data['Low']
    
    strategy_predictions = {}
    for strategy, diffs in daily_diffs.items():
        if len(diffs) > 1 and not diffs.isna().all():
            mean = diffs.mean()
            std = diffs.std()
            conf_interval = [mean - 1.96 * std, mean + 1.96 * std]
            conf_lower = max(conf_interval[0], 0) if strategy == "Min-Low to End-Close" else conf_interval[0]
            strategy_predictions[strategy] = {
                "Mean": mean,
                "Conf Lower": conf_lower,
                "Conf Upper": conf_interval[1],
                "Std": std
            }
    
    ml_predictions = {}
    if len(data) > 1 and any(strategies.values()) and not data.empty:
        data_ml = data[['Close', 'Volume']].copy()
        data_ml['Lag_Close'] = data_ml['Close'].shift(1)
        data_ml['Lag_Volume'] = data_ml['Volume'].shift(1)
        data_ml = data_ml.dropna()
        
        train_size = int(len(data_ml) * 0.8)
        if train_size < 2 or len(data_ml) - train_size < 1:
            st.warning("Insufficient data for ML predictions. Minimum 3 data points required.")
        else:
            train_data = data_ml[:train_size]
            test_data = data_ml[train_size:]
            
            for strategy in strategies:
                if strategies[strategy]:
                    y_full = daily_diffs[strategy].dropna()
                    if len(y_full) < 3:
                        ml_predictions[strategy] = {"Predicted Increase": 0.0, "RMSE": 0.0}
                        continue
                    
                    common_indices = y_full.index.intersection(data_ml.index)
                    y_aligned = y_full.loc[common_indices].values
                    X_aligned = data_ml.loc[common_indices, ['Lag_Close', 'Lag_Volume']].values
                    
                    train_size_strategy = int(len(y_aligned) * 0.8)
                    y_train = y_aligned[:train_size_strategy]
                    y_test = y_aligned[train_size_strategy:]
                    X_train = X_aligned[:train_size_strategy]
                    X_test = X_aligned[train_size_strategy:] if len(y_test) > 0 else np.array([])
                    
                    if len(y_train) < 1 or len(X_train) < 1:
                        ml_predictions[strategy] = {"Predicted Increase": 0.0, "RMSE": 0.0}
                        continue
                    
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # Use last available data for prediction
                    last_data = (data_ml.iloc[-1][['Lag_Close', 'Lag_Volume']].values.reshape(1, -1)
                                 if len(data_ml) > 0 else
                                 X_train[-1].reshape(1, -1))
                    
                    if len(X_test) > 0 and len(y_test) > 0:
                        ml_pred = model.predict(X_test)
                        ml_rmse = (np.sqrt(np.mean((ml_pred - y_test) ** 2))
                                   if len(ml_pred) == len(y_test) else 0.0)
                    else:
                        ml_rmse = 0.0
                    ml_next_pred = model.predict(last_data)[0]
                    ml_predictions[strategy] = {"Predicted Increase": ml_next_pred, "RMSE": ml_rmse}
    
    raw_data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Daily Increase ($)', 'Open vs Prev Close ($)', 'Intraday Increase ($)']].copy() if not data.empty else pd.DataFrame()
    if not raw_data.empty:
        raw_data['Close Color'] = raw_data.apply(
            lambda x: color_close(x['Close'], raw_data['Close'].shift(1)[x.name] if x.name in raw_data['Close'].shift(1).index else None),
            axis=1
        )
    
    volatility = data['Close'].std() if not data.empty and 'Close' in data.columns else 0
    avg_daily_range = (data['High'] - data['Low']).mean() if not data.empty and 'High' in data.columns and 'Low' in data.columns else 0
    volume_weighted_profits = {}
    for strategy, selected in strategies.items():
        if selected and f"{strategy} ($)" in daily_df.columns and not daily_df.empty:
            dollar_col = f"{strategy} ($)"
            volume_weighted_profits[strategy] = (daily_df[dollar_col] * (data['Volume'] / avg_volume)).sum() if avg_volume != 0 else 0
    
    comparison = []
    for strategy, selected in strategies.items():
        if selected and f"{strategy} ($)" in daily_df.columns and not daily_df.empty:
            dollar_col = f"{strategy} ($)"
            percent_col = f"{strategy} (%)"
            max_daily_gap = daily_df[dollar_col].max()
            max_daily_percent = daily_df[percent_col].max()
            max_day = daily_df[dollar_col].idxmax() if max_daily_gap else None
            agg_profit = aggregated_profit.get(dollar_col, None)
            agg_percent = aggregated_profit.get(percent_col, None)
            comparison.append({
                "Strategy": strategy,
                "Max Daily Gap ($)": max_daily_gap,
                "Max Daily Return (%)": max_daily_percent,
                "Best Day": max_day,
                "Aggregated Profit ($)": agg_profit,
                "Aggregated Return (%)": agg_percent
            })
    
    comparison_df = pd.DataFrame(comparison)
    if not comparison_df.empty:
        comparison_df.sort_values(by="Max Daily Gap ($)", ascending=False, inplace=True)
    
    return daily_df, aggregated_profit, comparison_df, price_extremes, volume_data, avg_volume, total_volume, max_volume, min_volume, max_volume_date, min_volume_date, volatility, avg_daily_range, volume_weighted_profits, raw_data, daily_diffs, strategy_predictions, ml_predictions

# Run analysis on button click
if st.button("Run Analysis"):
    with st.spinner("Running analysis..."):
        if not any(strategies.values()):
            st.error("Please select at least one comparison strategy.")
        else:
            data, company_details = fetch_data(ticker, start_date, end_date)
            print(f"Data after fetch: {type(data)}, {data}")
            if data is None or data.empty:
                st.error(f"Failed to fetch data for {ticker}. Please check the ticker or date range and try again.")
                st.stop()
            daily_df, aggregated_profit, comparison_df, price_extremes, volume_data, avg_volume, total_volume, max_volume, min_volume, max_volume_date, min_volume_date, volatility, avg_daily_range, volume_weighted_profits, raw_data, daily_diffs, strategy_predictions, ml_predictions = calculate_profits(data, strategies, strategy_variant, start_date, end_date)
            print(f"Data in insights: {type(data)}, {data.head()}")
            
            # Tabbed Interface
            tabs = st.tabs(["Quick Summary", "Data & Metrics", "Analysis", "Predictions", "Insights"])

            with tabs[0]:
                st.subheader("Quick Summary")
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.write(f"**Company Details for {ticker}**")
                if company_details:
                    st.write(f"- **Name**: {company_details['Name']}")
                    st.write(f"- **Sector**: {company_details['Sector']}")
                    st.write(f"- **Industry**: {company_details['Industry']}")
                    st.write(f"- **Market Cap**: ${company_details['Market Cap']:,} (USD)" if isinstance(company_details['Market Cap'], (int, float)) else f"- **Market Cap**: {company_details['Market Cap']}")
                    st.write(f"- **P/E Ratio**: {company_details['P/E Ratio']:.2f}" if isinstance(company_details['P/E Ratio'], (int, float)) else f"- **P/E Ratio**: {company_details['P/E Ratio']}")
                    st.write(f"- **EPS**: ${company_details['EPS']:.2f}" if isinstance(company_details['EPS'], (int, float)) else f"- **EPS**: {company_details['EPS']}")
                    st.write(f"- **Dividend Yield**: {company_details['Dividend Yield']*100:.2f}%" if isinstance(company_details['Dividend Yield'], (int, float)) else f"- **Dividend Yield**: {company_details['Dividend Yield']}")
                    st.write(f"- **Beta**: {company_details['Beta']:.2f}" if isinstance(company_details['Beta'], (int, float)) else f"- **Beta**: {company_details['Beta']}")
                    st.write("*Fundamentals sourced from Yahoo Finance. Verify with additional sources before making investment decisions.*")
                
                st.write(f"**Performance Summary ({start_date} to {end_date})**")
                if strategy_predictions and ml_predictions and not comparison_df.empty:
                    best_confident = max(strategy_predictions.items(), key=lambda x: x[1]["Conf Lower"]) if strategy_predictions else ("N/A", {"Conf Lower": 0, "Conf Upper": 0})
                    best_ml = max(ml_predictions.items(), key=lambda x: x[1]["Predicted Increase"]) if ml_predictions else ("N/A", {"Predicted Increase": 0})
                    best_agg = comparison_df.loc[comparison_df["Aggregated Profit ($)"].idxmax()] if not comparison_df.empty else None
                    if best_agg is not None:
                        st.write(f"- **Best Historical Strategy**: {best_agg['Strategy']} (${best_agg['Aggregated Profit ($)']:.2f}, {best_agg['Aggregated Return (%)']:.2f}% from {aggregated_profit.get(f'{best_agg['Strategy']} Buy Date', 'N/A')} to {aggregated_profit.get(f'{best_agg['Strategy']} Sell Date', 'N/A')})")
                    st.write(f"- **Most Confident Gap**: {best_confident[0]} (Expected: ${best_confident[1]['Conf Lower']:.2f} - ${best_confident[1]['Conf Upper']:.2f})")
                    st.write(f"- **ML Predicted Gap (Next Day)**: {best_ml[0]} (${best_ml[1]['Predicted Increase']:.2f})")
                    for strategy, profit in volume_weighted_profits.items():
                        st.write(f"- **Volume-Weighted Profit ({strategy})**: ${profit:.2f}")
                else:
                    st.write("No predictions available due to insufficient data.")
                st.markdown('</div>', unsafe_allow_html=True)

            with tabs[1]:
                with st.expander("Raw Stock Data", expanded=False):
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.write(f"Raw stock data for {ticker} ({start_date} to {end_date}):")
                    if not raw_data.empty:
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
                    else:
                        st.write("No raw data available.")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with st.expander("Summary Metrics"):
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Volatility (Close Price Std)", f"{volatility:.2f}")
                    col2.metric("Average Daily Range (High-Low)", f"{avg_daily_range:.2f}")
                    col3.metric("Total Volume", f"{total_volume:.0f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with st.expander("Price Extremes"):
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    if price_extremes:
                        extremes_df = pd.DataFrame(price_extremes)
                        extremes_df.set_index("Metric", inplace=True)
                        st.dataframe(extremes_df.style.format({"Highest Value": "{:.2f}", "Lowest Value": "{:.2f}"}))
                    else:
                        st.write("No price extremes data available.")
                    st.markdown('</div>', unsafe_allow_html=True)

            with tabs[2]:
                with st.expander("Daily Gap and Sentiment", expanded=True):
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.write("Daily gap from Low to Close for selected strategies (assuming 1 share) to gauge market sentiment:")
                    if not daily_df.empty:
                        st.dataframe(daily_df.style.format({col: "{:.2f}" for col in daily_df.columns}).applymap(color_profit_loss, subset=[col for col in daily_df.columns if col.endswith("($)")]))
                    
                        if any(col.endswith("($)") for col in daily_df.columns):
                            profit_cols = [col for col in daily_df.columns if col.endswith("($)")]
                            pivot_df = daily_df[profit_cols].T
                            fig_heatmap = px.imshow(pivot_df,
                                                   labels=dict(x="Date", y="Strategy", color="Gap ($)"),
                                                   color_continuous_scale="RdYlGn",
                                                   aspect="auto",
                                                   title=f"Heatmap of Daily Gap by Strategy for {ticker} ({start_date} to {end_date})")
                            fig_heatmap.update_layout(coloraxis_colorbar_title="Gap ($)")
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                        if "Min-Low to End-Close ($)" in daily_df.columns:
                            daily_df['Sentiment'] = pd.cut(daily_df["Min-Low to End-Close ($)"],
                                                          bins=[-float('inf'), 0, 2, 5, float('inf')],
                                                          labels=['Neutral', 'Weak Bullish', 'Bullish', 'Strong Bullish'])
                            sentiment_volume_df = pd.concat([daily_df[['Min-Low to End-Close ($)', 'Sentiment']], volume_data['Volume']], axis=1)
                            sentiment_volume_df['Volume Change'] = sentiment_volume_df['Volume'].pct_change() * 100
                            st.write("Market Sentiment and Volume Correlation based on Min-Low to End-Close Gap:")
                            st.dataframe(sentiment_volume_df.style.format({
                                "Min-Low to End-Close ($)": "{:.2f}",
                                "Volume": "{:.0f}",
                                "Volume Change": "{:.2f}%"
                            }))
                            
                            all_profitable = daily_df[[col for col in daily_df.columns if col.endswith("($)")]].apply(lambda x: x >= 0, axis=1).all(axis=1)
                            profitable_days = daily_df[all_profitable].index
                            if len(profitable_days) > 0:
                                common_indices = profitable_days.intersection(volume_data.index)
                                if not common_indices.empty:
                                    mean_volume_profitable = volume_data.loc[common_indices, 'Volume'].mean()
                                    st.write(f"Volume on these days (mean): {mean_volume_profitable:.0f} shares")
                                else:
                                    st.write("No volume data available for profitable days.")
                                st.write("Days with All Strategies Profitable:")
                                st.dataframe(daily_df.loc[profi
