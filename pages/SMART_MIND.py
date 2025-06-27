import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- Streamlit Config ---
st.set_page_config(page_title="Smart Pattern Analyzer", layout="centered")
st.title("📊 Smart Pattern Analyzer for Day Traders")

# --- Inputs ---
with st.form(key='stock_form'):
    symbol = st.text_input("Enter Stock Symbol", value="AAPL")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2025, 4, 28))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2025, 6, 27))
    comparison = st.radio(
        "Select metric to compare:",
        ["Open", "High", "Low", "Close", "Volume", "All"],
        index=5
    )
    submit_button = st.form_submit_button(label="🚀 Analyze Pattern")

# --- Analyze Button Logic ---
if submit_button:
    # Validate user input
    if symbol and start_date and end_date:
        try:
            # Fetch stock data using yfinance
            df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)
            
            # Check if data is returned
            if not df.empty:
                st.success(f"✅ Analysis complete for {symbol.upper()} from {start_date} to {end_date}")
                
                # Debug: Print actual date range
                df_reset = df.reset_index()
                min_date = df_reset['Date'].min()
                max_date = df_reset['Date'].max()
                st.write(f"Actual data range: {min_date} to {max_date}")

                # --- Clean and Format ---
                df = df.reset_index()
                
                # Handle MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    column_mapping = {
                        ('Date', ''): 'Date',
                        ('Open', symbol): 'Open',
                        ('High', symbol): 'High',
                        ('Low', symbol): 'Low',
                        ('Close', symbol): 'Close',
                        ('Volume', symbol): 'Volume',
                        ('Adj Close', symbol): 'Adj_Close'
                    }
                    df.columns = [column_mapping.get(col, col[0]) for col in df.columns]
                else:
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

                # --- Recovery Pattern Flag and Profit Analysis ---
                if all(col in df.columns for col in ["Open", "Low", "Close", "Prev_Close"]):
                    df["Low_Diff"] = df["Open"] - df["Low"]
                    df["Recovered"] = np.where(df["Close"] >= df["Open"], "Yes", "No")
                    df["Profit_Low_to_Close"] = df["Close"] - df["Low"]
                    df["Profit_Percent"] = (df["Profit_Low_to_Close"] / df["Low"]) * 100
                    # Additional Metrics
                    df["Volatility"] = df["High"] - df["Low"]
                    df["True_Range"] = np.maximum.reduce([
                        df["High"] - df["Low"],
                        abs(df["High"] - df["Prev_Close"]),
                        abs(df["Low"] - df["Prev_Close"])
                    ])
                else:
                    st.warning("Required columns for recovery pattern and profit analysis (Open, Low, Close, Prev_Close) not found.")

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
                st.subheader("📋 Recent Pattern Data")
                st.dataframe(df.tail(25), use_container_width=True)

                # --- Profit Analysis Summary ---
                if "Profit_Low_to_Close" in df.columns:
                    st.subheader("💰 Profit Analysis (Buy at Low, Sell at Close)")
                    profitable_days = len(df[df["Profit_Low_to_Close"] > 0])
                    total_days = len(df)
                    avg_profit = df["Profit_Low_to_Close"].mean()
                    avg_profit_percent = df["Profit_Percent"].mean()
                    total_profit = df["Profit_Low_to_Close"].sum()
                    win_days = df[df["Profit_Low_to_Close"] > 0]
                    loss_days = df[df["Profit_Low_to_Close"] <= 0]
                    win_loss_ratio = len(win_days) / len(loss_days) if len(loss_days) > 0 else float('inf')
                    avg_win_profit = win_days["Profit_Low_to_Close"].mean() if not win_days.empty else 0
                    avg_loss = loss_days["Profit_Low_to_Close"].mean() if not loss_days.empty else 0
                    recovery_rate = (len(df[df["Recovered"] == "Yes"]) / total_days) * 100
                    avg_volatility = df["Volatility"].mean()
                    avg_atr = df["True_Range"].mean()
                    cumulative_profit = df["Profit_Low_to_Close"].cumsum()
                    max_drawdown = (cumulative_profit - cumulative_profit.cummax()).min()

                    # Create table for profit analysis
                    profit_metrics = pd.DataFrame({
                        "Metric": [
                            "Profitable Days",
                            "Average Profit per Day ($)",
                            "Average Profit per Day (%)",
                            "Total Profit Over Period ($)",
                            "Win/Loss Ratio",
                            "Average Profit on Winning Days ($)",
                            "Average Loss on Losing Days ($)",
                            "Intraday Recovery Rate (%)",
                            "Average Daily Volatility ($)",
                            "Average True Range ($)",
                            "Maximum Drawdown ($)"
                        ],
                        "Value": [
                            f"{profitable_days} out of {total_days} ({(profitable_days/total_days)*100:.2f}%)",
                            f"{avg_profit:.2f}",
                            f"{avg_profit_percent:.2f}%",
                            f"{total_profit:.2f}",
                            f"{win_loss_ratio:.2f}" if win_loss_ratio != float('inf') else "No Losses",
                            f"{avg_win_profit:.2f}",
                            f"{avg_loss:.2f}" if avg_loss != 0 else "No Losses",
                            f"{recovery_rate:.2f}%",
                            f"{avg_volatility:.2f}",
                            f"{avg_atr:.2f}",
                            f"{max_drawdown:.2f}"
                        ]
                    })
                    st.table(profit_metrics)

                    # Plot profit over time
                    st.subheader("📈 Profit Trend (Low to Close)")
                    st.line_chart(df.set_index("Date")["Profit_Low_to_Close"])

                # --- Price Trend Chart ---
                st.subheader("📈 Price Trend Chart")
                try:
                    if all(col in df.columns for col in ["Date", "Open", "High", "Low", "Close"]):
                        st.line_chart(df.set_index("Date")[["Open", "High", "Low", "Close"]])
                    else:
                        st.warning("Required columns for plotting (Date, Open, High, Low, Close) not found.")
                except Exception as e:
                    st.warning(f"Could not plot chart: {e}")

                # Option to download as CSV
                csv = df.to_csv()
                st.download_button(
                    label="Download Data as CSV",
                    data=csv,
                    file_name=f"{symbol}_data.csv",
                    mime="text/csv"
                )
            else:
                st.error("No data found for the given parameters.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please fill in all fields.")
