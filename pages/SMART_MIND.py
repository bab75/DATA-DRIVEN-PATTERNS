import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(page_title="Smart Pattern Analyzer", layout="centered")
st.title("📈 Smart Pattern Analyzer for Day Traders")

# --- Inputs ---
symbol = st.text_input("Enter Stock Symbol", value="AAPL")

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=60))
with col2:
    end_date = st.date_input("End Date", value=datetime.today())

comparison = st.radio(
    "Select which metric to compare:",
    ["Open", "High", "Low", "Close", "Volume", "All"],
    index=5
)

# --- Submit Button ---
if st.button("Analyze Pattern"):

    # --- Data Fetching ---
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        st.stop()

    if df.empty:
        st.error("No data retrieved. Please check the symbol or date range.")
    else:
        df.reset_index(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)

        # --- Lagged Columns for Comparison ---
        df["Prev Open"] = df["Open"].shift(1)
        df["Prev High"] = df["High"].shift(1)
        df["Prev Low"] = df["Low"].shift(1)
        df["Prev Close"] = df["Close"].shift(1)
        df["Prev Volume"] = df["Volume"].shift(1)

        # --- Recovery Pattern Tag ---
        df["Low_Diff"] = df["Open"] - df["Low"]
        df["Recovered"] = np.where(df["Close"] >= df["Open"], "Yes", "No")

        # --- Dynamic Day-over-Day Comparisons ---
        cols_to_compare = ["Open", "High", "Low", "Close", "Volume"] if comparison == "All" else [comparison]
        
        for col in cols_to_compare:
            curr_col = col
            prev_col = f"Prev {col}"
            if curr_col in df.columns and prev_col in df.columns:
                df[f"{col}_Change_vs_Yest"] = df[curr_col] - df[prev_col]
            else:
                st.warning(f"Could not compute difference for {col} due to missing data.")

        st.success(f"Analysis completed for {symbol.upper()} from {start_date} to {end_date}.")
        st.subheader("🔍 Pattern Summary")
        st.dataframe(df.tail(25), use_container_width=True)

        # --- Optional Visualization ---
        st.subheader("📊 Price Movement Overview")
        st.line_chart(df.set_index("Date")[["Open", "Close", "Low", "High"]])
