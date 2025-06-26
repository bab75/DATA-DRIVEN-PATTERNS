import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from datetime import datetime, timedelta

st.set_page_config(page_title="Pattern Analyzer", layout="centered")

st.title("📊 Historical Pattern Explorer for Day Traders")

# --- Sidebar Inputs ---
symbol = st.text_input("Enter Stock Symbol", value="AAPL")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=60))
with col2:
    end_date = st.date_input("End Date", value=datetime.today())

comparison = st.radio(
    "Choose comparison metric:",
    ["Open", "High", "Low", "Close", "Volume", "All"],
    index=5
)

# --- Button to trigger analysis ---
if st.button("🚀 Analyze Pattern"):

    # Download data
    df = yf.download(symbol, start=start_date, end=end_date)

    if df.empty:
        st.error("No data found. Check the symbol or date range.")
    else:
        df.reset_index(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df["Prev Close"] = df["Close"].shift(1)
        df["Prev Open"] = df["Open"].shift(1)
        df["Prev Low"] = df["Low"].shift(1)
        df["Prev High"] = df["High"].shift(1)
        df["Prev Volume"] = df["Volume"].shift(1)

        df["Low_Diff"] = df["Open"] - df["Low"]
        df["Recovered"] = np.where(df["Close"] >= df["Open"], "Yes", "No")

        # Smart previous day difference logic (holiday-aware)
        df["Prev_Date"] = df["Date"].apply(lambda d: (d - BDay(1)).date())

        if comparison != "All":
    col_name = comparison
    prev_col = f"Prev {col_name}"
    if col_name in df.columns and prev_col in df.columns:
        df[f"{col_name}_Change_vs_Yest"] = df[col_name] - df[prev_col]
    else:
        st.warning(f"Column {col_name} or {prev_col} not found in data.")
else:
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        prev_col = f"Prev {col}"
        if col in df.columns and prev_col in df.columns:
            df[f"{col}_Change_vs_Yest"] = df[col] - df[prev_col]
        else:
            st.warning(f"Skipping {col}: column or previous column missing.")

        st.success(f"Showing pattern analysis for {symbol.upper()} from {start_date} to {end_date}")
        st.dataframe(df.tail(25), use_container_width=True)

        # Optional: Show some visualizations
        st.line_chart(df.set_index("Date")[["Open", "Close", "Low", "High"]])
