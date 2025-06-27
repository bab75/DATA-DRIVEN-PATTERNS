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
start_date = st.date_input("Start Date", value=datetime(2025, 6, 26))
end_date = st.date_input("End Date", value=datetime(2025, 6, 29))
st.write("**Note**: Ensure the date range includes at least one trading day. For multi-day analysis, the end date is the sell date.")

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
        # Capitalize column names for consistency
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

# Function to calculate profit/loss for strategies
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
            daily_profit["Low-Close"] = close - low
        if strategies["Open-High"]:
            daily_profit["Open-High"] = high - open_price
        if strategies["Open-Close"]:
            daily_profit["Open-Close"] = close - open_price
        if strategies["Low-High"]:
            daily_profit["Low-High"] = high - low
        
        daily_results.append({
            "Date": date,
            **daily_profit
        })
    
    # Aggregated analysis (start date to end date)
    aggregated_profit = {}
    first_open = data['Open'].iloc[0]
    last_close = data['Close'].iloc[-1]
    period_low = data['Low'].min()
    period_high = data['High'].max()
    
    if strategies["Low-Close"]:
        aggregated_profit["Low-Close"] = last_close - period_low
    if strategies["Open-High"]:
        aggregated_profit["Open-High"] = period_high - first_open
    if strategies["Open-Close"]:
        aggregated_profit["Open-Close"] = last_close - first_open
    if strategies["Low-High"]:
        aggregated_profit["Low-High"] = period_high - period_low
    
    # Convert daily results to DataFrame
    daily_df = pd.DataFrame(daily_results)
    daily_df.set_index("Date", inplace=True)
    
    # Find most profitable day/strategy
    comparison = []
    for strategy, selected in strategies.items():
        if selected and strategy in daily_df.columns:
            max_daily_profit = daily_df[strategy].max()
            max_day = daily_df[strategy].idxmax() if max_daily_profit else None
            agg_profit = aggregated_profit.get(strategy, None)
            comparison.append({
                "Strategy": strategy,
                "Max Daily Profit": max_daily_profit,
                "Best Day": max_day,
                "Aggregated Profit": agg_profit
            })
    
    comparison_df = pd.DataFrame(comparison)
    
    return daily_df, aggregated_profit, comparison_df

# Run analysis on button click
if st.button("Run Analysis"):
    with st.spinner("Running analysis..."):
        data = fetch_data(ticker, start_date, end_date)
        if data is not None:
            if not any(strategies.values()):
                st.error("Please select at least one comparison strategy.")
            else:
                daily_df, aggregated_profit, comparison_df = calculate_profits(data, strategies, start_date, end_date)
                
                # Display daily results
                st.subheader("Daily Profit/Loss")
                st.write("Profit/loss per day for selected strategies (assuming 1 share):")
                st.dataframe(daily_df)
                
                # Display aggregated results
                st.subheader("Aggregated Profit/Loss (Start to End Date)")
                agg_df = pd.DataFrame([aggregated_profit], index=[f"{start_date} to {end_date}"])
                st.dataframe(agg_df)
                
                # Display comparison
                st.subheader("Comparison of Strategies")
                st.write("Comparing max daily profit vs. aggregated profit:")
                st.dataframe(comparison_df)
                
                # Plot daily profits
                if not daily_df.empty:
                    fig = px.line(daily_df, x=daily_df.index, y=daily_df.columns,
                                 title=f"Daily Profit/Loss for {ticker}",
                                 labels={"value": "Profit/Loss ($)", "Date": "Date", "variable": "Strategy"})
                    st.plotly_chart(fig)
                
                # Highlight most profitable strategy
                if not comparison_df.empty:
                    best_strategy = comparison_df.loc[comparison_df["Max Daily Profit"].idxmax()]["Strategy"]
                    best_day = comparison_df.loc[comparison_df["Max Daily Profit"].idxmax()]["Best Day"]
                    best_agg = comparison_df.loc[comparison_df["Aggregated Profit"].idxmax()]["Strategy"]
                    st.write(f"**Most Profitable Daily Strategy**: {best_strategy} on {best_day}")
                    st.write(f"**Most Profitable Aggregated Strategy**: {best_agg} for {start_date} to {end_date}")
