import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import backtrader as bt
from datetime import datetime

# Streamlit page configuration
st.set_page_config(page_title="Backtrader Backtester", page_icon="📈")
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
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            st.error(f"Data for {ticker} missing required columns: {required_columns}")
            return None
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

# Backtrader strategy: Moving Average Crossover
class MACrossover(bt.Strategy):
    params = (
        ('short_window', 50),
        ('long_window', 200),
    )

    def __init__(self):
        self.short_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.short_window)
        self.long_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.long_window)
        self.price = self.data.close
        self.order = None
        self.portfolio_value = []  # Track portfolio value

    def next(self):
        self.portfolio_value.append(self.broker.getvalue())
        if self.order:
            return
        if not self.position:
            if self.short_ma[0] > self.long_ma[0]:
                self.order = self.buy()
        else:
            if self.short_ma[0] < self.long_ma[0]:
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None

# Function to run Backtrader backtest
def run_backtrader(ticker, start_date, end_date, capital_base):
    data = fetch_data(ticker, start_date, end_date)
    if data is None:
        return None, None
    
    # Create Backtrader data feed
    try:
        data_feed = bt.feeds.PandasData(dataname=data)
    except Exception as e:
        st.error(f"Error creating Backtrader data feed: {str(e)}")
        return None, None
    
    # Initialize Backtrader cerebro engine
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MACrossover)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(capital_base)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission
    
    # Run backtest
    try:
        strats = cerebro.run()
        final_value = cerebro.broker.getvalue()
        strategy = strats[0]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Date': data.index[:len(strategy.portfolio_value)],
            'Portfolio Value': strategy.portfolio_value,
            'Price': data['Close'],
            'Short MA': data['Close'].rolling(window=50).mean(),
            'Long MA': data['Close'].rolling(window=200).mean()
        })
        return results, final_value
    except Exception as e:
        st.error(f"Backtest failed: {str(e)}")
        return None, None

# Run backtest on button click
if st.button("Run Backtest"):
    with st.spinner("Running backtest..."):
        results, final_value = run_backtrader(ticker, start_date, end_date, capital_base)
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
