import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit.components.v1 as components

# Custom CSS for Tailwind styling
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
""", unsafe_allow_html=True)

# Function to fetch and clean data
def fetch_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if data.empty or len(data) < 50:  # Ensure enough data for strategies
            st.error(f"Insufficient data for {symbol}. Need at least 50 days.")
            return None
        # Clean data: remove NaN and ensure numeric values
        data = data.dropna()
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# SMA Crossover Strategy
def sma_crossover_backtest(data, short_period=20, long_period=50):
    data = data.copy()
    if len(data) < long_period:
        st.warning(f"Insufficient data for SMA Crossover ({len(data)} days, need {long_period}).")
        return [], 10000, [10000] * len(data)
    data['SMA_short'] = data['Close'].rolling(window=short_period).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_period).mean()
    position = 0
    trades = []
    cash = 10000
    shares = 0
    equity = [10000]
    
    for i in range(1, len(data)):
        if pd.isna(data['SMA_short'].iloc[i]) or pd.isna(data['SMA_long'].iloc[i]):
            equity.append(cash + (shares * data['Close'].iloc[i] if position == 1 else 0))
            continue
        if data['SMA_short'].iloc[i] > data['SMA_long'].iloc[i] and data['SMA_short'].iloc[i-1] <= data['SMA_long'].iloc[i-1]:
            if position == 0:
                if data['Close'].iloc[i] > 0:  # Avoid division by zero
                    shares = cash / data['Close'].iloc[i]
                    cash = 0
                    position = 1
                    trades.append(('BUY', data.index[i], data['Close'].iloc[i], shares))
        elif data['SMA_short'].iloc[i] < data['SMA_long'].iloc[i] and data['SMA_short'].iloc[i-1] >= data['SMA_long'].iloc[i-1]:
            if position == 1:
                cash = shares * data['Close'].iloc[i]
                shares = 0
                position = 0
                trades.append(('SELL', data.index[i], data['Close'].iloc[i], shares))
        equity.append(cash + (shares * data['Close'].iloc[i] if position == 1 else 0))
    
    final_value = cash + (shares * data['Close'].iloc[-1] if position == 1 else 0)
    return trades, final_value, equity

# RSI Mean Reversion Strategy
def rsi_mean_reversion_backtest(data, rsi_period=14, oversold=30, overbought=70):
    data = data.copy()
    if len(data) < rsi_period:
        st.warning(f"Insufficient data for RSI Mean Reversion ({len(data)} days, need {rsi_period}).")
        return [], 10000, [10000] * len(data)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    position = 0
    trades = []
    cash = 10000
    shares = 0
    equity = [10000]
    
    for i in range(1, len(data)):
        if pd.isna(data['RSI'].iloc[i]):
            equity.append(cash + (shares * data['Close'].iloc[i] if position == 1 else 0))
            continue
        if data['RSI'].iloc[i] < oversold and position == 0:
            if data['Close'].iloc[i] > 0:
                shares = cash / data['Close'].iloc[i]
                cash = 0
                position = 1
                trades.append(('BUY', data.index[i], data['Close'].iloc[i], shares))
        elif data['RSI'].iloc[i] > overbought and position == 1:
            cash = shares * data['Close'].iloc[i]
            shares = 0
            position = 0
            trades.append(('SELL', data.index[i], data['Close'].iloc[i], shares))
        equity.append(cash + (shares * data['Close'].iloc[i] if position == 1 else 0))
    
    final_value = cash + (shares * data['Close'].iloc[-1] if position == 1 else 0)
    return trades, final_value, equity

# Bollinger Bands Strategy
def bollinger_bands_backtest(data, period=20, std_dev=2):
    data = data.copy()
    if len(data) < period:
        st.warning(f"Insufficient data for Bollinger Bands ({len(data)} days, need {period}).")
        return [], 10000, [10000] * len(data)
    data['SMA'] = data['Close'].rolling(window=period).mean()
    data['STD'] = data['Close'].rolling(window=period).std()
    data['Upper'] = data['SMA'] + (data['STD'] * std_dev)
    data['Lower'] = data['SMA'] - (data['STD'] * std_dev)
    position = 0
    trades = []
    cash = 10000
    shares = 0
    equity = [10000]
    
    for i in range(1, len(data)):
        if pd.isna(data['Upper'].iloc[i]) or pd.isna(data['Lower'].iloc[i]) or pd.isna(data['STD'].iloc[i]):
            equity.append(cash + (shares * data['Close'].iloc[i] if position == 1 else 0))
            continue
        if data['Close'].iloc[i] < data['Lower'].iloc[i] and position == 0:
            if data['Close'].iloc[i] > 0:
                shares = cash / data['Close'].iloc[i]
                cash = 0
                position = 1
                trades.append(('BUY', data.index[i], data['Close'].iloc[i], shares))
        elif data['Close'].iloc[i] > data['Upper'].iloc[i] and position == 1:
            cash = shares * data['Close'].iloc[i]
            shares = 0
            position = 0
            trades.append(('SELL', data.index[i], data['Close'].iloc[i], shares))
        equity.append(cash + (shares * data['Close'].iloc[i] if position == 1 else 0))
    
    final_value = cash + (shares * data['Close'].iloc[-1] if position == 1 else 0)
    return trades, final_value, equity

# Buy and Hold Strategy
def buy_and_hold_backtest(data):
    if len(data) < 1:
        st.warning("No data available for Buy and Hold strategy.")
        return [], 10000, []
    cash = 10000
    if data['Close'].iloc[0] <= 0:
        st.warning("Invalid price data for Buy and Hold strategy.")
        return [], 10000, [10000] * len(data)
    shares = cash / data['Close'].iloc[0]
    trades = [('BUY', data.index[0], data['Close'].iloc[0], shares)]
    final_value = shares * data['Close'].iloc[-1]
    equity = [shares * data['Close'].iloc[i] for i in range(len(data))]
    return trades, final_value, equity

# Performance Metrics
def calculate_metrics(equity, data):
    if not equity or len(equity) < 2:
        st.warning("Equity list is empty or too short for metrics calculation.")
        return {
            'Total Return (%)': 0,
            'Annualized Return (%)': 0,
            'Volatility (%)': 0,
            'Sharpe Ratio': 0,
            'Max Drawdown (%)': 0
        }
    
    try:
        equity = np.array(equity, dtype=float)
        if np.any(np.isnan(equity)) or np.any(np.isinf(equity)):
            st.warning("Equity contains NaN or inf values.")
            return {
                'Total Return (%)': 0,
                'Annualized Return (%)': 0,
                'Volatility (%)': 0,
                'Sharpe Ratio': 0,
                'Max Drawdown (%)': 0
            }
        
        returns = np.diff(equity) / equity[:-1]
        total_return = (equity[-1] - 10000) / 10000 * 100
        annualized_return = ((equity[-1] / 10000) ** (252 / len(data)) - 1) * 100
        volatility = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 0 else 0
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        equity_series = pd.Series(equity)
        rolling_max = equity_series.cummax()
        drawdown = (rolling_max - equity_series) / rolling_max
        max_drawdown = drawdown.max() * 100 if len(drawdown) > 0 else 0
        
        return {
            'Total Return (%)': total_return,
            'Annualized Return (%)': annualized_return,
            'Volatility (%)': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown
        }
    except Exception as e:
        st.error(f"Error in metrics calculation: {e}")
        return {
            'Total Return (%)': 0,
            'Annualized Return (%)': 0,
            'Volatility (%)': 0,
            'Sharpe Ratio': 0,
            'Max Drawdown (%)': 0
        }

# Plot Price with Signals
def plot_price(data, trades, strategy_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price', line=dict(color='blue')))
    buy_signals = [(t[1], t[2]) for t in trades if t[0] == 'BUY' and isinstance(t[2], (int, float))]
    sell_signals = [(t[1], t[2]) for t in trades if t[0] == 'SELL' and isinstance(t[2], (int, float))]
    fig.add_trace(go.Scatter(x=[t[0] for t in buy_signals], y=[t[1] for t in buy_signals], mode='markers', name='Buy', marker=dict(symbol='triangle-up', size=10, color='green')))
    fig.add_trace(go.Scatter(x=[t[0] for t in sell_signals], y=[t[1] for t in sell_signals], mode='markers', name='Sell', marker=dict(symbol='triangle-down', size=10, color='red')))
    fig.update_layout(title=f'{strategy_name} - Price with Signals', xaxis_title='Date', yaxis_title='Price')
    return fig

# Plot Equity Curves
def plot_equity_curves(results, data):
    fig = go.Figure()
    for strategy_name, result in results.items():
        if result['equity'] and len(result['equity']) == len(data):
            fig.add_trace(go.Scatter(x=data.index, y=result['equity'], name=strategy_name))
    fig.update_layout(title='Equity Curves Comparison', xaxis_title='Date', yaxis_title='Portfolio Value ($)')
    return fig

# Streamlit App
st.markdown("""
    <div class="bg-gray-100 p-6 rounded-lg shadow-md">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-4">Stock Backtesting Dashboard</h1>
        <p class="text-center text-gray-600 mb-6">Test and compare trading strategies with historical stock data.</p>
    </div>
""", unsafe_allow_html=True)

# Input Form
with st.form("backtest_form"):
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Stock Symbol (e.g., AAPL)", "AAPL")
        start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2023, 12, 31))
        st.write("")  # Spacer
    
    st.markdown("<h3 class='text-lg font-semibold text-gray-700'>Select Strategies</h3>", unsafe_allow_html=True)
    strategies = {
        "Simple Moving Average Crossover": st.checkbox("Simple Moving Average Crossover", value=True),
        "RSI Mean Reversion": st.checkbox("RSI Mean Reversion"),
        "Bollinger Bands": st.checkbox("Bollinger Bands"),
        "Buy and Hold": st.checkbox("Buy and Hold")
    }
    
    submitted = st.form_submit_button("Run Backtest", use_container_width=True)

# Run Backtest
if submitted:
    if not any(strategies.values()):
        st.error("Please select at least one strategy.")
    elif start_date >= end_date:
        st.error("Start date must be before end date.")
    else:
        data = fetch_data(symbol, start_date, end_date)
        if data is not None and not data.empty:
            results = {}
            for strategy_name, selected in strategies.items():
                if selected:
                    try:
                        if strategy_name == "Simple Moving Average Crossover":
                            trades, final_value, equity = sma_crossover_backtest(data)
                        elif strategy_name == "RSI Mean Reversion":
                            trades, final_value, equity = rsi_mean_reversion_backtest(data)
                        elif strategy_name == "Bollinger Bands":
                            trades, final_value, equity = bollinger_bands_backtest(data)
                        elif strategy_name == "Buy and Hold":
                            trades, final_value, equity = buy_and_hold_backtest(data)
                        
                        if not equity or len(equity) != len(data):
                            st.warning(f"Invalid equity data for {strategy_name}. Length: {len(equity)}")
                            continue
                        
                        results[strategy_name] = {
                            'trades': trades,
                            'final_value': final_value,
                            'equity': equity,
                            'metrics': calculate_metrics(equity, data)
                        }
                    except Exception as e:
                        st.error(f"Error in {strategy_name}: {e}")
            
            if not results:
                st.error("No valid results generated. Please check inputs or data.")
            else:
                # Display Results
                st.markdown("<h2 class='text-2xl font-bold text-gray-800 mt-6'>Backtest Results</h2>", unsafe_allow_html=True)
                for strategy_name, result in results.items():
                    st.markdown(f"<h3 class='text-xl font-semibold text-gray-700'>{strategy_name}</h3>", unsafe_allow_html=True)
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write("**Performance Metrics**")
                        metrics_df = pd.DataFrame(result['metrics'].items(), columns=['Metric', 'Value'])
                        metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.2f}")
                        st.table(metrics_df)
                        st.write("**Trades**")
                        trades_df = pd.DataFrame(result['trades'], columns=['Action', 'Date', 'Price', 'Shares'])
                        # Ensure Price is numeric before formatting
                        trades_df['Price'] = pd.to_numeric(trades_df['Price'], errors='coerce')
                        if not trades_df.empty and not trades_df['Price'].isna().all():
                            trades_df['Price'] = trades_df['Price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                            trades_df['Shares'] = trades_df['Shares'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                            st.dataframe(trades_df)
                        else:
                            st.write("No trades executed.")
                    with col2:
                        st.plotly_chart(plot_price(data, result['trades'], strategy_name))
                
                # Equity Curve Comparison
                if len(results) > 1:
                    st.markdown("<h3 class='text-xl font-semibold text-gray-700'>Equity Curves Comparison</h3>", unsafe_allow_html=True)
                    st.plotly_chart(plot_equity_curves(results, data))
