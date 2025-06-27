import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit.components.v1 as components

# Custom CSS for Tailwind styling
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

# Function to fetch and clean data
def fetch_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if data is None or data.empty or len(data) < 50:
            st.error(f"Insufficient or no data for {symbol}. Need at least 50 days.")
            return None
        # Ensure Close column exists and is numeric
        if 'Close' not in data.columns:
            st.error(f"No 'Close' column in data for {symbol}.")
            return None
        # Convert to numeric and drop NaN
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data = data.dropna()
        # Check for valid prices
        if len(data) < 50:
            st.error(f"Insufficient valid data for {symbol} after cleaning ({len(data)} days).")
            return None
        if (data['Close'] <= 0).any():
            st.error(f"Invalid data for {symbol}: Contains zero or negative prices.")
            return None
        # Select relevant columns and ensure float type
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# SMA Crossover Strategy
def sma_crossover_backtest(data, short_period=20, long_period=50):
    data = data.copy()
    if len(data) < long_period:
        st.warning(f"Insufficient data for SMA Crossover ({len(data)} days, need {long_period}).")
        return [], 10000, [10000] * len(data)
    data['SMA_short'] = data['Close'].rolling(window=short_period).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_period).mean()
    position = 0
    trades = []
    cash = 10000
    shares = 0
    equity = [10000]
    
    for i in range(1, len(data)):
        if pd.isna(data['SMA_short'].iloc[i]) or pd.isna(data['SMA_long'].iloc[i]):
            equity.append(cash + (shares * data['Close'].iloc[i] if position == 1 else 0))
            continue
        close_price = data['Close'].iloc[i]
        if not pd.isna(close_price) and close_price > 0:
            if data['SMA_short'].iloc[i] > data['SMA_long'].iloc[i] and data['SMA_short'].iloc[i-1] <= data['SMA_long'].iloc[i-1]:
                if position == 0:
                    shares = cash / close_price
                    cash = 0
                    position = 1
                    trades.append(('BUY', data.index[i], close_price, shares))
            elif data['SMA_short'].iloc[i] < data['SMA_long'].iloc[i] and data['SMA_short'].iloc[i-1] >= data['SMA_long'].iloc[i-1]:
                if position == 1:
                    cash = shares * close_price
                    shares = 0
                    position = 0
                    trades.append(('SELL', data.index[i], close_price, shares))
        equity.append(cash + (shares * close_price if position == 1 else 0))
    
    final_value = cash + (shares * data['Close'].iloc[-1] if position == 1 else 0)
    return trades, final_value, equity

# RSI Mean Reversion Strategy
def rsi_mean_reversion_backtest(data, rsi_period=14, oversold=30, overbought=70):
    data = data.copy()
    if len(data) < rsi_period:
        st.warning(f"Insufficient data for RSI Mean Reversion ({len(data)} days, need {rsi_period}).")
        return [], 10000, [10000] * len(data)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    position = 0
    trades = []
    cash = 10000
    shares = 0
    equity = [10000]
    
    for i in range(1, len(data)):
        if pd.isna(data['RSI'].iloc[i]):
            equity.append(cash + (shares * data['Close'].iloc[i] if position == 1 else 0))
            continue
        close_price = data['Close'].iloc[i]
        if not pd.isna(close_price) and close_price > 0:
            if data['RSI'].iloc[i] < oversold and position == 0:
                shares = cash / close_price
                cash = 0
                position = 1
                trades.append(('BUY', data.index[i], close_price, shares))
            elif data['RSI'].iloc[i] > overbought and position == 1:
                cash = shares * close_price
                shares = 0
                position = 0
                trades.append(('SELL', data.index[i], close_price, shares))
        equity.append(cash + (shares * close_price if position == 1 else 0))
    
    final_value = cash + (shares * data['Close'].iloc[-1] if position == 1 else 0)
    return trades, final_value, equity

# Bollinger Bands Strategy
def bollinger_bands_backtest(data, period=20, std_dev=2):
    data = data.copy()
    if len(data) < period:
        st.warning(f"Insufficient data for Bollinger Bands ({len(data)} days, need {period}).")
        return [], 10000, [10000] * len(data)
    data['SMA'] = data['Close'].rolling(window=period).mean()
    data['STD'] = data['Close'].rolling(window=period).std()
    data['Upper'] = data['SMA'] + (data['STD'] * std_dev)
    data['Lower'] = data['SMA'] - (data['STD'] * std_dev)
    position = 0
    trades = []
    cash = 10000
    shares = 0
    equity = [10000]
    
    for i in range(1, len(data)):
        if pd.isna(data['Upper'].iloc[i]) or pd.isna(data['Lower'].iloc[i]) or pd.isna(data['STD'].iloc[i]):
            equity.append(cash + (shares * data['Close'].iloc[i] if position == 1 else 0))
            continue
        close_price = data['Close'].iloc[i]
        if not pd.isna(close_price) and close_price > 0:
            if data['Close'].iloc[i] < data['Lower'].iloc[i] and position == 0:
                shares = cash / close_price
                cash = 0
                position = 1
                trades.append(('BUY', data.index[i], close_price, shares))
            elif data['Close'].iloc[i] > data['Upper'].iloc[i] and position == 1:
                cash = shares * close_price
                shares = 0
                position = 0
                trades.append(('SELL', data.index[i], close_price, shares))
        equity.append(cash + (shares * close_price if position == 1 else 0))
    
    final_value = cash + (shares * data['Close'].iloc[-1] if position == 1 else 0)
    return trades, final_value, equity

# Buy and Hold Strategy
def buy_and_hold_backtest(data):
    if len(data) < 1:
        st.warning("No data available for Buy and Hold strategy.")
        return [], 10000, []
    close_price_first = data['Close'].iloc[0]
    if pd.isna(close_price_first) or close_price_first <=ഗ

System: * Today's date and time is 06:37 PM EDT on Friday, June 27, 2025.
