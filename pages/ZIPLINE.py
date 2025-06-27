import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from zipline import run_algorithm
from zipline.api import order_target_percent, record, symbol
from zipline.data.bundles import register, csvdir
from zipline.utils.calendars import get_calendar
import os
from datetime import datetime
import uuid

# Streamlit app title
st.title("Trading Strategy Backtester")

# User inputs
ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
end_date = st.date_input("End Date", value=datetime(2023, 12, 31))
capital_base = st.number_input("Capital Base ($)", value=10000.0, min_value=1000.0)

# Function to fetch data from yfinance
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        st.error(f"No data found for {ticker}. Please check the ticker or date range.")
        return None
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.index = pd.to_datetime(data.index)
    return data

# Function to prepare and save data for Zipline
def prepare_zipline_data(ticker, start_date, end_date):
    data = fetch_data(ticker, start_date, end_date)
    if data is None:
        return None
    # Create directory for Zipline bundle
    bundle_dir = os.path.join(os.getcwd(), 'data', 'csvdir', ticker)
    os.makedirs(bundle_dir, exist_ok=True)
    # Save data to CSV
    csv_path = os.path.join(bundle_dir, f"{ticker}.csv")
    data.to_csv(csv_path)
    return data

# Register custom CSV bundle for Zipline
def register_bundle(ticker):
    bundle_dir = os.path.join(os.getcwd(), 'data', 'csvdir')
    register(
        'csvdir',
        csvdir.csvdir_equities([bundle_dir]),
        calendar_name='NYSE'
    )

# Zipline strategy: Moving Average Crossover
def initialize(context):
    context.asset = symbol(ticker)
    context.short_window = 50
    context.long_window = 200

def handle_data(context, data):
    hist = data.history(context.asset, 'price', context.long_window + 1, '1d')
    short_mavg = hist[-context.short_window:].mean()
    long_mavg = hist[-context.long_window:].mean()

    if short_mavg > long_mavg:
        order_target_percent(context.asset, 1.0)  # Buy
    elif short_mavg < long_mavg:
        order_target_percent(context.asset, 0.0)  # Sell

    record(price=data.current(context.asset, 'price'), short_mavg=short_mavg, long_mavg=long_mavg)

# Run Zipline backtest
def run_zipline(ticker, start_date, end_date, capital_base):
    data = prepare_zipline_data(ticker, start_date, end_date)
    if data is None:
        return None
    register_bundle(ticker)
    try:
        result = run_algorithm(
            start=pd.to_datetime(start_date),
            end=pd.to_datetime(end_date),
            initialize=initialize,
            handle_data=handle_data,
            capital_base=capital_base,
            data_frequency='daily',
            bundle='csvdir'
        )
        return result
    except Exception as e:
        st.error(f"Backtest failed: {str(e)}")
        return None

# Run backtest on button click
if st.button("Run Backtest"):
    with st.spinner("Running backtest..."):
        result = run_zipline(ticker, start_date, end_date, capital_base)
        if result is not None:
            # Calculate key metrics
            total_returns = (result['portfolio_value'][-1] / capital_base) - 1
            st.subheader("Backtest Results")
            st.write(f"**Total Returns**: {{:.2%}}".format(total_returns))
            
            # Display results table
            st.dataframe(result[['portfolio_value', 'returns', 'price', 'short_mavg', 'long_mavg']])
            
            # Plot portfolio value
            fig = px.line(result, x=result.index, y='portfolio_value', title=f"Portfolio Value for {ticker}")
            st.plotly_chart(fig)
