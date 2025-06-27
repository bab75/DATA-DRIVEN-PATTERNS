import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
    col3, col4 = st.columns(2)
    with col3:
        min_profit_pct = st.number_input("Minimum Profit % for Filter", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    with col4:
        min_loss_pct = st.number_input("Minimum Loss % for Filter", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
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

                # --- Handle Missing Data ---
                if df.isnull().any().any():
                    st.warning("Some data points are missing. Filling with previous valid values where possible.")
                    df = df.fillna(method='ffill').fillna(method='bfill')

                # --- Previous Day Columns ---
                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    if col in df.columns:
                        df[f"Prev_{col}"] = df[col].shift(1)
                    else:
                        st.warning(f"Column {col} not found in the data.")

                # --- Technical Indicators ---
                if all(col in df.columns for col in ["Close", "High", "Low", "Volume"]):
                    # Moving Averages
                    df['SMA_20'] = df['Close'].rolling(window=20).mean()
                    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
                    
                    # Bollinger Bands
                    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
                    df['BB_Std'] = df['Close'].rolling(window=20).std()
                    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
                    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
                    
                    # RSI
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['RSI'] = 100 - (100 / (1 + rs))
                    
                    # On-Balance Volume
                    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
                else:
                    st.warning("Required columns for technical indicators (Close, High, Low, Volume) not found.")

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

                # --- Apply Profit/Loss Filters ---
                if "Profit_Percent" in df.columns:
                    filtered_df = df[
                        (df["Profit_Percent"] >= min_profit_pct) | 
                        (df["Profit_Percent"] <= -min_loss_pct)
                    ]
                    if filtered_df.empty:
                        st.warning("No data matches the profit/loss filter criteria.")
                else:
                    filtered_df = df
                    st.warning("Profit percentage data not available for filtering.")

                # --- Display Results ---
                st.subheader("📋 Recent Pattern Data (Filtered)")
                st.dataframe(filtered_df, use_container_width=True)

                # --- Profit Analysis Summary ---
                if "Profit_Low_to_Close" in filtered_df.columns:
                    st.subheader("💰 Profit Analysis (Buy at Low, Sell at Close)")
                    profitable_days = len(filtered_df[filtered_df["Profit_Low_to_Close"] > 0])
                    total_days = len(filtered_df)
                    avg_profit = filtered_df["Profit_Low_to_Close"].mean()
                    avg_profit_percent = filtered_df["Profit_Percent"].mean()
                    total_profit = filtered_df["Profit_Low_to_Close"].sum()
                    win_days = filtered_df[filtered_df["Profit_Low_to_Close"] > 0]
                    loss_days = filtered_df[filtered_df["Profit_Low_to_Close"] <= 0]
                    win_loss_ratio = len(win_days) / len(loss_days) if len(loss_days) > 0 else float('inf')
                    avg_win_profit = win_days["Profit_Low_to_Close"].mean() if not win_days.empty else 0
                    avg_loss = loss_days["Profit_Low_to_Close"].mean() if not loss_days.empty else 0
                    recovery_rate = (len(filtered_df[filtered_df["Recovered"] == "Yes"]) / total_days) * 100 if total_days > 0 else 0
                    avg_volatility = filtered_df["Volatility"].mean()
                    avg_atr = filtered_df["True_Range"].mean()
                    cumulative_profit = filtered_df["Profit_Low_to_Close"].cumsum()
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
                            f"{profitable_days} out of {total_days} ({(profitable_days/total_days)*100:.2f}%)" if total_days > 0 else "No data",
                            f"{avg_profit:.2f}" if not pd.isna(avg_profit) else "N/A",
                            f"{avg_profit_percent:.2f}%" if not pd.isna(avg_profit_percent) else "N/A",
                            f"{total_profit:.2f}",
                            f"{win_loss_ratio:.2f}" if win_loss_ratio != float('inf') else "No Losses",
                            f"{avg_win_profit:.2f}" if not pd.isna(avg_win_profit) else "N/A",
                            f"{avg_loss:.2f}" if avg_loss != 0 and not pd.isna(avg_loss) else "No Losses",
                            f"{recovery_rate:.2f}%",
                            f"{avg_volatility:.2f}" if not pd.isna(avg_volatility) else "N/A",
                            f"{avg_atr:.2f}" if not pd.isna(avg_atr) else "N/A",
                            f"{max_drawdown:.2f}" if not pd.isna(max_drawdown) else "N/A"
                        ]
                    })
                    st.table(profit_metrics)

                    # Plot profit over time
                    st.subheader("📈 Profit Trend (Low to Close)")
                    st.line_chart(filtered_df.set_index("Date")["Profit_Low_to_Close"])

                # --- Price Trend Chart with Indicators ---
                st.subheader("📈 Price Trend Chart")
                try:
                    if all(col in filtered_df.columns for col in ["Date", "Open", "High", "Low", "Close", "SMA_20", "BB_Upper", "BB_Lower"]):
                        fig = go.Figure(data=[
                            go.Candlestick(x=filtered_df['Date'],
                                          open=filtered_df['Open'],
                                          high=filtered_df['High'],
                                          low=filtered_df['Low'],
                                          close=filtered_df['Close'],
                                          increasing_line_color='green',
                                          decreasing_line_color='red'),
                            go.Scatter(x=filtered_df['Date'], y=filtered_df['SMA_20'], name='SMA 20', line=dict(color='blue')),
                            go.Scatter(x=filtered_df['Date'], y=filtered_df['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dash')),
                            go.Scatter(x=filtered_df['Date'], y=filtered_df['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dash'))
                        ])
                        fig.update_layout(
                            title=f"{symbol.upper()} Candlestick Chart with Indicators",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            xaxis_rangeslider_visible=False,
                            hovermode='x unified',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Required columns for plotting (Date, Open, High, Low, Close, SMA_20, BB_Upper, BB_Lower) not found.")
                except Exception as e:
                    st.warning(f"Could not plot chart: {e}")

                # --- RSI Chart ---
                if 'RSI' in filtered_df.columns:
                    st.subheader("📉 RSI Trend")
                    fig_rsi = go.Figure(data=[go.Scatter(x=filtered_df['Date'], y=filtered_df['RSI'], mode='lines', name='RSI')])
                    fig_rsi.update_layout(
                        title=f"{symbol.upper()} RSI",
                        xaxis_title="Date",
                        yaxis_title="RSI",
                        yaxis_range=[0, 100],
                        xaxis_rangeslider_visible=False,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_rsi, use_container_width=True)

                # Option to download as CSV
                csv = filtered_df.to_csv()
                st.download_button(
                    label="Download Filtered Data as CSV",
                    data=csv,
                    file_name=f"{symbol}_filtered_data.csv",
                    mime="text/csv"
                )
            else:
                st.error("No data found for the given parameters.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please fill in all fields.")
