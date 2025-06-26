import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from datetime import datetime, timedelta

# --- Streamlit Config ---
st.set_page_config(page_title="Smart Pattern Analyzer", layout="centered")
st.title("📊 Smart Pattern Analyzer for Day Traders")

# --- Inputs ---
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

# --- Analyze Button ---
if st.button("🚀 Analyze Pattern"):

    try:
        df = yf.download(symbol, start=start_date, end=end_date)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    if df.empty:
        st.warning("No data retrieved. Please check your symbol or date range.")
        st.stop()

    # --- Clean and Format ---
    df = df.reset_index()
    df.columns = [str(col).strip().replace(" ", "_") for col in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    # --- Previous Day Columns ---
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[f"Prev_{col}"] = df[col].shift(1)

    # --- Recovery Pattern Flag ---
    df["Low_Diff"] = df["Open"] - df["Low"]
    df["Recovered"] = np.where(df["Close"] >= df["Open"], "Yes", "No")

    # --- Metric Comparisons ---
    selected_metrics = ["Open", "High", "Low", "Close", "Volume"] if comparison == "All" else [comparison]

    for metric in selected_metrics:
        try:
            df[f"{metric}_Change_vs_Yest"] = df[metric] - df[f"Prev_{metric}"]
        except Exception as e:
            st.warning(f"Could not compute difference for {metric}: {e}")

    # --- Display Results ---
    st.success(f"✅ Analysis complete for {symbol.upper()} from {start_date} to {end_date}")
    st.subheader("📋 Recent Pattern Data")
    st.dataframe(df.tail(25), use_container_width=True)

    # --- Price Trend Chart ---
    st.subheader("📈 Price Trend Chart")
    try:
        st.line_chart(df.set_index("Date")[["Open", "High", "Low", "Close"]])
    except Exception as e:
        st.warning(f"Could not plot chart: {e}")
