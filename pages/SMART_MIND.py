import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from datetime import datetime, timedelta

# --- Streamlit Config ---
st.set_page_config(page_title="Smart Pattern Analyzer", layout="centered")
st.title("📊 Smart Pattern Analyzer for Day Traders")

# --- User Inputs ---
symbol = st.text_input("Enter Stock Symbol", value="AAPL")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=60))
with col2:
    end_date = st.date_input("End Date", value=datetime.today())

comparison = st.radio(
    "Select metric to compare:",
    ["Open", "High", "Low", "Close", "Volume", "All"],
    index=5
)

# --- On Submit ---
if st.button("🚀 Analyze Pattern"):

    # Download Data
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    if df.empty:
        st.warning("No data found. Check the symbol or date range.")
        st.stop()

    # Clean & Prepare
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    # Create previous day columns
    df["Prev_Open"] = df["Open"].shift(1)
    df["Prev_High"] = df["High"].shift(1)
    df["Prev_Low"] = df["Low"].shift(1)
    df["Prev_Close"] = df["Close"].shift(1)
    df["Prev_Volume"] = df["Volume"].shift(1)

    # Recovery detection
    df["Low_Diff"] = df["Open"] - df["Low"]
    df["Recovered"] = np.where(df["Close"] >= df["Open"], "Yes", "No")

    # Metric Comparison
    metric_list = ["Open", "High", "Low", "Close", "Volume"] if comparison == "All" else [comparison]

    for metric in metric_list:
        prev_col = f"Prev_{metric}"
        if metric in df.columns and prev_col in df.columns:
            try:
                df[f"{metric}_Change_vs_Yest"] = df[metric] - df[prev_col]
            except Exception as e:
                st.warning(f"Couldn't compute difference for {metric}: {e}")
        else:
            st.warning(f"Missing data for {metric}. Skipping.")

    # Show Results
    st.success(f"📈 Analysis complete for {symbol.upper()} from {start_date} to {end_date}")
    st.dataframe(df.tail(25), use_container_width=True)

    st.subheader("📉 Price Trend Overview")
    st.line_chart(df.set_index("Date")[["Open", "High", "Low", "Close"]])
