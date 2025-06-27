import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from datetime import datetime

# Streamlit page configuration
st.set_page_config(page_title="Backtesting.py Backtester", page_icon="📈")
st.title("Trading Strategy Backtester")

# User inputs
ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
end_date = st.date_input("End Date", value=datetime(2023, 12, 31))
capital_base = st.number_input("Capital Base ($)", value=10000.0, min_value=1000.0)

# Function to fetch data from yfinance
def fetch_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.error(f"No data found for {ticker}. Please check the ticker or date range.")
            return None
        # Ensure required columns and capitalize for Backtesting.py
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            st.error(f"Data for {ticker} missing required columns: {required_columns}")
            return None
        # Backtesting.py expects capitalized column names
        data.columns = [col.capitalize() for col in data.columns]
        st.write(f"DataFrame columns: {data.columns.tolist()}")
        st.write(f"DataFrame sample:\n{data.head()}")
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Backtesting.py strategy: Moving Average Crossover
class MACrossover(Strategy):
    short_window = 50
    long_window = 200

    def init(self):
        close = self.data.Close
        self.short_ma = self.I(pd.Series.rolling, close, self.short_window, min_periods=1).mean()
        self.long_ma = self.I(pd.Series.rolling, close, self.long_window, min_periods=1).mean()

    def next(self):
        if crossover(self.short_ma, self.long_ma):
            self.buy()
        elif crossover(self.long_ma, self.short_ma):
            self.sell()

# Function to run Backtesting.py backtest
def run_backtest(ticker, start_date, end_date, capital_base):
    data = fetch_data(ticker, start_date, end_date)
    if data is None:
        return None, None
    
    try:
        bt = Backtest(data, MACrossover, cash=capital_base, commission=0.001)
        stats = bt.run()
        results = pd.DataFrame({
            'Date': data.index,
            'Portfolio Value': stats['_equity_curve']['Equity'],
            'Price': data['Close'],
            'Short MA': data['Close'].rolling(window=50).mean(),
            'Long MA': data['Close'].rolling(window=200).mean()
        })
        final_value = stats['Equity Final [$]']
        return results, final_value
    except Exception as e:
        st.error(f"Backtest failed: {str(e)}")
        return None, None

# Run backtest on button click
if st.button("Run Backtest"):
    with st.spinner("Running backtest..."):
        results, final_value = run_backtest(ticker, start_date, end_date, capital_base)
        if results is not None and final_value is not None:
            # Calculate key metrics
            total_returns = (final_value / capital_base) - 1
            st.subheader("Backtest Results")
            st.write(f"**Total Returns**: {total_returns:.2%}")
            
            # Display results table
            st.dataframe(results[['Date', 'Portfolio Value', 'Price', 'Short MA', 'Long MA']])
            
            # Plot portfolio value
            fig = px.line(results, x='Date', y='Portfolio Value', title=f"Portfolio Value for {ticker}")
            st.plotly_chart(fig)
