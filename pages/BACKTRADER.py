import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# Streamlit page configuration
st.set_page_config(page_title="Stock Price Comparison Dashboard", page_icon="📊")
st.title("Stock Price Comparison Dashboard")

# User inputs
ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
start_date = st.date_input("Start Date", value=datetime(2024, 6, 1))
end_date = st.date_input("End Date", value=datetime(2024, 6, 30))
st.write("**Note**: Select a date range with trading days (e.g., avoid weekends). End date is the sell date for aggregated analysis.")

# Validate date range
if start_date >= end_date:
    st.error("End date must be after start date.")
    st.stop()

# Strategy selection
st.subheader("Select Comparison Strategies")
strategies = {
    "Low-Close": st.checkbox("Low to Close (Buy at Low, Sell at Close)", value=True),
    "Open-High": st.checkbox("Open to High (Buy at Open, Sell at High)", value=True),
    "Open-Close": st.checkbox("Open to Close (Buy at Open, Sell at Close)", value=True),
    "Low-High": st.checkbox("Low to High (Buy at Low, Sell at High)", value=True)
}

# Function to fetch data from yfinance
def fetch_data(ticker, start_date, end_date):
    try:
        # Adjust end date to include the last trading day
        end_date_adjusted = end_date + timedelta(days=1)
        data = yf.download(ticker, start=start_date, end=end_date_adjusted, progress=False)
        if data.empty:
            st.error(f"No data found for {ticker}. Please check the ticker or date range.")
            return None
        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        # Ensure required columns
        required_columns = ['Open', 'High', 'Low', 'Close']
        actual_columns = data.columns.tolist()
        if not all(col in actual_columns for col in required_columns):
            st.error(f"Data for {ticker} missing required columns. Expected: {required_columns}, Found: {actual_columns}")
            return None
        # Capitalize column names
        data.columns = [col.capitalize() for col in data.columns]
        data.index = pd.to_datetime(data.index)
        # Filter to exact date range
        data = data.loc[start_date:end_date]
        if data.empty:
            st.error(f"No trading data available between {start_date} and {end_date}.")
            return None
        # Log DataFrame for debugging
        st.write(f"DataFrame columns: {data.columns.tolist()}")
        st.write(f"DataFrame sample:\n{data.head()}")
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Function to color-code profit/loss
def color_profit_loss(val):
    color = 'background-color: lightgreen' if val > 0 else 'background-color: lightcoral' if val < 0 else ''
    return color

# Function to calculate profit/loss and percentage returns
def calculate_profits(data, strategies, start_date, end_date):
    daily_results = []
    aggregated_results = {}
    
    # Daily analysis
    for date in data.index:
        daily_profit = {}
        close = data.loc[date, 'Close']
        open_price = data.loc[date, 'Open']
        high = data.loc[date, 'High']
        low = data.loc[date, 'Low']
        
        if strategies["Low-Close"]:
            profit = close - low
            daily_profit["Low-Close ($)"] = profit
            daily_profit["Low-Close (%)"] = (profit / low * 100) if low != 0 else 0
        if strategies["Open-High"]:
            profit = high - open_price
            daily_profit["Open-High ($)"] = profit
            daily_profit["Open-High (%)"] = (profit / open_price * 100) if open_price != 0 else 0
        if strategies["Open-Close"]:
            profit = close - open_price
            daily_profit["Open-Close ($)"] = profit
            daily_profit["Open-Close (%)"] = (profit / open_price * 100) if open_price != 0 else 0
        if strategies["Low-High"]:
            profit = high - low
            daily_profit["Low-High ($)"] = profit
            daily_profit["Low-High (%)"] = (profit / low * 100) if low != 0 else 0
        
        daily_results.append({
            "Date": date,
            **daily_profit
        })
    
    # Aggregated analysis
    aggregated_profit = {}
    first_open = data['Open'].iloc[0]
    last_close = data['Close'].iloc[-1]
    period_low = data['Low'].min()
    period_high = data['High'].max()
    
    if strategies["Low-Close"]:
        profit = last_close - period_low
        aggregated_profit["Low-Close ($)"] = profit
        aggregated_profit["Low-Close (%)"] = (profit / period_low * 100) if period_low != 0 else 0
    if strategies["Open-High"]:
        profit = period_high - first_open
        aggregated_profit["Open-High ($)"] = profit
        aggregated_profit["Open-High (%)"] = (profit / first_open * 100) if first_open != 0 else 0
    if strategies["Open-Close"]:
        profit = last_close - first_open
        aggregated_profit["Open-Close ($)"] = profit
        aggregated_profit["Open-Close (%)"] = (profit / first_open * 100) if first_open != 0 else 0
    if strategies["Low-High"]:
        profit = period_high - period_low
        aggregated_profit["Low-High ($)"] = profit
        aggregated_profit["Low-High (%)"] = (profit / period_low * 100) if period_low != 0 else 0
    
    # Convert daily results to DataFrame
    daily_df = pd.DataFrame(daily_results)
    daily_df.set_index("Date", inplace=True)
    
    # Comparison table
    comparison = []
    for strategy, selected in strategies.items():
        if selected:
            dollar_col = f"{strategy} ($)"
            percent_col = f"{strategy} (%)"
            if dollar_col in daily_df.columns:
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
    
    return daily_df, aggregated_profit, comparison_df

# Run analysis on button click
if st.button("Run Analysis"):
    with st.spinner("Running analysis..."):
        if not any(strategies.values()):
            st.error("Please select at least one comparison strategy.")
        else:
            data = fetch_data(ticker, start_date, end_date)
            if data is not None:
                daily_df, aggregated_profit, comparison_df = calculate_profits(data, strategies, start_date, end_date)
                
                # Display daily results
                st.subheader("Daily Profit/Loss")
                st.write("Profit/loss per day for selected strategies (assuming 1 share):")
                st.dataframe(daily_df.style.format({col: "{:.2f}" for col in daily_df.columns}).applymap(color_profit_loss, subset=[col for col in daily_df.columns if col.endswith("($)")]))
                
                # Display aggregated results
                st.subheader(f"Aggregated Profit/Loss ({start_date} to {end_date})")
                agg_df = pd.DataFrame([aggregated_profit], index=[f"{start_date} to {end_date}"])
                st.dataframe(agg_df.style.format({col: "{:.2f}" for col in agg_df.columns}).applymap(color_profit_loss, subset=[col for col in agg_df.columns if col.endswith("($)")]))
                
                # Display comparison
                st.subheader("Comparison of Strategies")
                st.write("Comparing max daily profit vs. aggregated profit (sorted by aggregated return):")
                st.dataframe(comparison_df.style.format({
                    "Max Daily Profit ($)": "{:.2f}",
                    "Max Daily Return (%)": "{:.2f}",
                    "Aggregated Profit ($)": "{:.2f}",
                    "Aggregated Return (%)": "{:.2f}"
                }).applymap(color_profit_loss, subset=["Max Daily Profit ($)", "Aggregated Profit ($)"]))
                
                # Plot daily profits
                if not daily_df.empty:
                    dollar_cols = [col for col in daily_df.columns if col.endswith("($)")]
                    fig = px.line(daily_df, x=daily_df.index, y=dollar_cols,
                                 title=f"Daily Profit/Loss for {ticker}",
                                 labels={"value": "Profit/Loss ($)", "Date": "Date", "variable": "Strategy"})
                    st.plotly_chart(fig)
                
                # Highlight most profitable strategy
                if not comparison_df.empty:
                    best_daily = comparison_df.loc[comparison_df["Max Daily Profit ($)"].idxmax()]
                    best_agg = comparison_df.loc[comparison_df["Aggregated Profit ($)"].idxmax()]
                    st.write(f"**Most Profitable Daily Strategy**: {best_daily['Strategy']} on {best_daily['Best Day']} "
                             f"(${best_daily['Max Daily Profit ($)']:.2f}, {best_daily['Max Daily Return (%)']:.2f}%)")
                    st.write(f"**Most Profitable Aggregated Strategy**: {best_agg['Strategy']} "
                             f"(${best_agg['Aggregated Profit ($)']:.2f}, {best_agg['Aggregated Return (%)']:.2f}%)")
