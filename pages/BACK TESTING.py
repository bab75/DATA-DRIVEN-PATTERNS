import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit.components.v1 as components

# Custom CSS for Tailwind styling with enhancements
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        .custom-container { padding: 1.5rem; background-color: #f9fafb; border-radius: 0.5rem; margin-bottom: 1.5rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .custom-table { max-width: 100%; overflow-x: auto; }
        .custom-table table { width: 100%; border-collapse: collapse; }
        .custom-table th, .custom-table td { padding: 0.75rem; border: 1px solid #e5e7eb; }
        .custom-table th { background-color: #e5e7eb; font-weight: 600; }
        .suggestion-box { background-color: #e0f2fe; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem; }
        h2, h3 { color: #1f2937; }
    </style>
""", unsafe_allow_html=True)

# Function to fetch and clean data
def fetch_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if data is None or data.empty or len(data) < 50:
            st.error(f"Insufficient or no data for {symbol}. Need at least 50 days.")
            return None
        
        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.error(f"Missing required columns for {symbol}: {missing_columns}")
            return None
        
        # Convert all columns to numeric, handle non-numeric values
        for col in required_columns:
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            except Exception as e:
                st.error(f"Error converting '{col}' to numeric for {symbol}: {e}")
                return None
        
        # Drop rows with NaN values and ensure sufficient data
        data = data.dropna()
        if len(data) < 50:
            st.error(f"Insufficient valid data for {symbol} after cleaning ({len(data)} days).")
            return None
        
        # Check for valid prices (no zero or negative values)
        if (data['Close'] <= 0).any():
            st.error(f"Invalid data for {symbol}: Contains zero or negative prices.")
            return None
        
        # Select only required columns and ensure float type
        data = data[required_columns].astype(float)
        return data
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
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
        close_price = data['Close'].iloc[i]
        if not pd.isna(close_price) and close_price > 0:
            if data['SMA_short'].iloc[i] > data['SMA_long'].iloc[i] and data['SMA_short'].iloc[i-1] <= data['SMA_long'].iloc[i-1]:
                if position == 0:
                    shares = cash / close_price
                    cash = 0
                    position = 1
                    trades.append(('BUY', data.index[i], close_price, shares))
            elif data['SMA_short'].iloc[i] < data['SMA_long'].iloc[i] and data['SMA_short'].iloc[i-1] >= data['SMA_long'].iloc[i-1]:
                if position == 1:
                    cash = shares * close_price
                    shares = 0
                    position = 0
                    trades.append(('SELL', data.index[i], close_price, shares))
        equity_value = cash + (shares * close_price if position == 1 else 0)
        equity.append(max(equity_value, 0))
    
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
        close_price = data['Close'].iloc[i]
        if not pd.isna(close_price) and close_price > 0:
            if data['RSI'].iloc[i] < oversold and position == 0:
                shares = cash / close_price
                cash = 0
                position = 1
                trades.append(('BUY', data.index[i], close_price, shares))
            elif data['RSI'].iloc[i] > overbought and position == 1:
                cash = shares * close_price
                shares = 0
                position = 0
                trades.append(('SELL', data.index[i], close_price, shares))
        equity_value = cash + (shares * close_price if position == 1 else 0)
        equity.append(max(equity_value, 0))
    
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
        close_price = data['Close'].iloc[i]
        if not pd.isna(close_price) and close_price > 0:
            if data['Close'].iloc[i] < data['Lower'].iloc[i] and position == 0:
                shares = cash / close_price
                cash = 0
                position = 1
                trades.append(('BUY', data.index[i], close_price, shares))
            elif data['Close'].iloc[i] > data['Upper'].iloc[i] and position == 1:
                cash = shares * close_price
                shares = 0
                position = 0
                trades.append(('SELL', data.index[i], close_price, shares))
        equity_value = cash + (shares * close_price if position == 1 else 0)
        equity.append(max(equity_value, 0))
    
    final_value = cash + (shares * data['Close'].iloc[-1] if position == 1 else 0)
    return trades, final_value, equity

# Buy and Hold Strategy
def buy_and_hold_backtest(data):
    if len(data) < 1:
        st.warning("No data available for Buy and Hold strategy.")
        return [], 10000, []
    close_price_first = data['Close'].iloc[0]
    if pd.isna(close_price_first) or close_price_first <= 0:
        st.warning("Invalid price data for Buy and Hold strategy.")
        return [], 10000, [10000] * len(data)
    cash = 10000
    shares = cash / close_price_first
    trades = [('BUY', data.index[0], close_price_first, shares)]
    final_value = shares * data['Close'].iloc[-1]
    equity = [max(shares * data['Close'].iloc[i], 0) for i in range(len(data))]
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
        equity = equity[np.isfinite(equity)]
        
        if len(equity) < 2:
            st.warning("Insufficient valid equity data after cleaning.")
            return {
                'Total Return (%)': 0,
                'Annualized Return (%)': 0,
                'Volatility (%)': 0,
                'Sharpe Ratio': 0,
                'Max Drawdown (%)': 0
            }
        
        equity_clean = equity[equity > 0]
        if len(equity_clean) < 2:
            st.warning("Equity contains zero or negative values.")
            return {
                'Total Return (%)': 0,
                'Annualized Return (%)': 0,
                'Volatility (%)': 0,
                'Sharpe Ratio': 0,
                'Max Drawdown (%)': 0
            }
        
        returns = []
        for i in range(1, len(equity_clean)):
            if equity_clean[i-1] > 0:
                returns.append((equity_clean[i] - equity_clean[i-1]) / equity_clean[i-1])
        
        if len(returns) == 0:
            returns = [0]
        
        returns = np.array(returns)
        total_return = (equity[-1] - 10000) / 10000 * 100 if equity[-1] > 0 else 0
        days_in_period = len(data) if len(data) > 0 else 1
        annualized_return = ((equity[-1] / 10000) ** (252 / days_in_period) - 1) * 100 if equity[-1] > 0 and days_in_period > 0 else 0
        volatility = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 0 else 0
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        equity_series = pd.Series(equity)
        rolling_max = equity_series.cummax()
        drawdown = (rolling_max - equity_series) / rolling_max
        max_drawdown = drawdown.max() * 100 if len(drawdown) > 0 else 0
        
        return {
            'Total Return (%)': round(total_return, 2),
            'Annualized Return (%)': round(annualized_return, 2),
            'Volatility (%)': round(volatility, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
            'Max Drawdown (%)': round(max_drawdown, 2)
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
    buy_signals = [(t[1], t[2]) for t in trades if t[0] == 'BUY' and isinstance(t[2], (int, float)) and not pd.isna(t[2])]
    sell_signals = [(t[1], t[2]) for t in trades if t[0] == 'SELL' and isinstance(t[2], (int, float)) and not pd.isna(t[2])]
    fig.add_trace(go.Scatter(x=[t[0] for t in buy_signals], y=[t[1] for t in buy_signals], mode='markers', name='Buy', marker=dict(symbol='triangle-up', size=10, color='green')))
    fig.add_trace(go.Scatter(x=[t[0] for t in sell_signals], y=[t[1] for t in sell_signals], mode='markers', name='Sell', marker=dict(symbol='triangle-down', size=10, color='red')))
    fig.update_layout(
        title=f'{strategy_name} - Price with Signals',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# Plot Equity Curves
def plot_equity_curves(results, data):
    fig = go.Figure()
    for strategy_name, result in results.items():
        if result['equity'] and len(result['equity']) == len(data):
            fig.add_trace(go.Scatter(x=data.index, y=result['equity'], name=strategy_name))
    fig.update_layout(
        title='Equity Curves Comparison',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        template='plotly_white',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# Generate Suggestions
def generate_suggestions(results):
    suggestions = []
    if not results:
        return ["No results available to generate suggestions."]
    
    # Find best-performing strategy by Total Return
    best_strategy = max(results.items(), key=lambda x: x[1]['metrics']['Total Return (%)'], default=(None, {'metrics': {'Total Return (%)': 0}}))[0]
    if best_strategy:
        suggestions.append(f"The {best_strategy} strategy achieved the highest Total Return. Consider focusing on this strategy.")
    
    # Check for high volatility
    for strategy_name, result in results.items():
        volatility = result['metrics']['Volatility (%)']
        if volatility > 30:
            suggestions.append(f"{strategy_name} has high volatility ({volatility:.2f}%). Consider reducing risk by adjusting parameters or diversifying.")
        
        # Check for low Sharpe Ratio
        sharpe = result['metrics']['Sharpe Ratio']
        if sharpe < 1 and sharpe != 0:
            suggestions.append(f"{strategy_name} has a low Sharpe Ratio ({sharpe:.2f}). Explore parameter optimization or alternative strategies.")
    
    if not suggestions:
        suggestions.append("All strategies performed within expected parameters. Try testing with different periods or symbols for further insights.")
    
    return suggestions

# Streamlit App
st.markdown("""
    <div class="custom-container">
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
        st.write("")
    
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
            # Debug: Display data summary
            st.write(f"Data fetched: {len(data)} rows, first few Close prices: {data['Close'].head().tolist()}")
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
                # Dynamic Heading
                st.markdown(
                    f"<h2 class='text-2xl font-bold text-gray-800 mt-6'>Backtest Results for {symbol} ({start_date} to {end_date})</h2>",
                    unsafe_allow_html=True
                )
                
                # Display Results in Expanders
                for strategy_name, result in results.items():
                    with st.expander(f"{strategy_name}", expanded=True):
                        st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
                        col1, col2 = st.columns([1, 2], gap="medium")
                        with col1:
                            st.markdown("<h4 class='text-lg font-semibold text-gray-700'>Performance Metrics</h4>", unsafe_allow_html=True)
                            metrics_df = pd.DataFrame(result['metrics'].items(), columns=['Metric', 'Value'])
                            st.markdown(
                                f'<div class="custom-table">{metrics_df.to_html(index=False, classes=["table-auto"])}</div>',
                                unsafe_allow_html=True
                            )
                            st.markdown("<h4 class='text-lg font-semibold text-gray-700 mt-4'>Trades</h4>", unsafe_allow_html=True)
                            trades_df = pd.DataFrame(result['trades'], columns=['Action', 'Date', 'Price', 'Shares'])
                            if not trades_df.empty:
                                trades_df['Price'] = pd.to_numeric(trades_df['Price'], errors='coerce')
                                if not trades_df['Price'].isna().all():
                                    trades_df['Price'] = trades_df['Price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N ποιοSystem: * Today's date and time is 06:00 PM EDT on Saturday, June 28, 2025.)
