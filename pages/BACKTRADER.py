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
        data.index = pd.to_datetime(data.index)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
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

    def next(self):
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
    data_feed = bt.feeds.PandasData(dataname=data)
    
    # Initialize Backtrader cerebro engine
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MACrossover)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(capital_base)
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission
    
    # Run backtest
    cerebro.run()
    
    # Extract results
    final_value = cerebro.broker.getvalue()
    trades = cerebro.runstrats[0][0].trades if cerebro.runstrats else []
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Date': data.index,
        'Portfolio Value': [capital_base] * len(data),  # Simplified; adjust for actual portfolio value
        'Price': data['Close'],
        'Short MA': data['Close'].rolling(window=50).mean(),
        'Long MA': data['Close'].rolling(window=200).mean()
    })
    
    return results, final_value

# Run backtest on button click
if st.button("Run Backtest"):
    with st.spinner("Running backtest..."):
        results, final_value = run_backtrader(ticker, start_date, end_date, capital_base)
        if results is not None:
            # Calculate key metrics
            total_returns = (final_value / capital_base) - 1
            st.subheader("Backtest Results")
            st.write(f"**Total Returns**: {total_returns:.2%}")
            
            # Display results table
            st.dataframe(results[['Date', 'Portfolio Value', 'Price', 'Short MA', 'Long MA']])
            
            # Plot portfolio value
            fig = px.line(results, x='Date', y='Portfolio Value', title=f"Portfolio Value for {ticker}")
            st.plotly_chart(fig)
