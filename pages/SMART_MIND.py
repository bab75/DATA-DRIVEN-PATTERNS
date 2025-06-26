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
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # Extract the first level of the MultiIndex and map to clean names
        column_mapping = {
            ('Date', ''): 'Date',
            ('Open', symbol): 'Open',
            ('High', symbol): 'High',
            ('Low', symbol): 'Low',
            ('Close', symbol): 'Close',
            ('Volume', symbol): 'Volume',
            ('Adj Close', symbol): 'Adj_Close'  # Handle Adj Close if present
        }
        df.columns = [column_mapping.get(col, col[0]) for col in df.columns]
    else:
        # Clean column names for single-level index
        df.columns = [str(col).strip().replace(" ", "_") for col in df.columns]

    # Verify if 'Date' column exists
    if 'Date' not in df.columns:
        st.error("Date column not found in the data. Available columns: " + ", ".join(df.columns))
        st.stop()
    
    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)

    # --- Previous Day Columns ---
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[f"Prev_{col}"] = df[col].shift(1)
        else:
            st.warning(f"Column {col} not found in the data.")

    # --- Recovery Pattern Flag ---
    if all(col in df.columns for col in ["Open", "Low", "Close"]):
        df["Low_Diff"] = df["Open"] - df["Low"]
        df["Recovered"] = np.where(df["Close"] >= df["Open"], "Yes", "No")
    else:
        st.warning("Required columns for recovery pattern (Open, Low, Close) not found.")

    # --- Metric Comparisons ---
    selected_metrics = ["Open", "High", "Low", "Close", "Volume"] if comparison == "All" else [comparison]

    for metric in selected_metrics:
        try:
            if metric in df.columns and f"Prev_{metric}" in df.columns:
                df[f"{metric}_Change_vs_Yest"] = df[metric] - df[f"Prev_{metric}"]
            else:
                st.warning(f"Could not compute difference for {metric}: Required columns not found.")
        except Exception as e:
            st.warning(f"Could not compute difference for {metric}: {e}")

    # --- Display Results ---
    st.success(f"✅ Analysis complete for {symbol.upper()} from {start_date} to {end_date}")
    st.subheader("📋 Recent Pattern Data")
    st.dataframe(df.tail(25), use_container_width=True)

    # --- Price Trend Chart ---
    st.subheader("📈 Price Trend Chart")
    try:
        if all(col in df.columns for col in ["Date", "Open", "High", "Low", "Close"]):
            st.line_chart(df.set_index("Date")[["Open", "High", "Low", "Close"]])
        else:
            st.warning("Required columns for plotting (Date, Open, High, Low, Close) not found.")
    except Exception as e:
        st.warning(f"Could not plot chart: {e}")
