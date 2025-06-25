```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime, timedelta
import base64
import ta
import openpyxl
import re
import calendar
import json
from sklearn.linear_model import LinearRegression
import pytz
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check Plotly version
if plotly.__version__ < '5.0.0':
    st.warning(f"Plotly version {plotly.__version__} detected. Please upgrade to Plotly 5.x or higher with: `pip install plotly --upgrade`")

# Streamlit page config
st.set_page_config(page_title="Stock Investment Analysis", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #ffffff; color: #000000; }
    .sidebar .sidebar-content { background-color: #f0f0f0; color: #000000; }
    .stButton>button { background-color: #4CAF50; color: #ffffff; border-radius: 5px; }
    .stFileUploader label { color: #000000; }
    .stTextInput label { color: #000000; }
    h1, h2, h3 { color: #0288d1; font-family: 'Arial', sans-serif; }
    .stExpander { background-color: #f5f5f5; border-radius: 5px; }
    .metric-box { background-color: #e0e0e0; padding: 10px; border-radius: 5px; color: #000000; }
    .trade-details { background-color: #f0f0f0; padding: 10px; border-radius: 5px; color: #000000; }
    .alert-box { background-color: #fff3e0; padding: 10px; border-radius: 5px; color: #000000; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
st.session_state.setdefault('trade_details', None)
st.session_state.setdefault('data_loaded', False)
st.session_state.setdefault('data_processed', False)
st.session_state.setdefault('symbol', 'AAPL')
if 'aapl_df' not in st.session_state:
    st.session_state.aapl_df = pd.DataFrame()
if 'pl_df' not in st.session_state:
    st.session_state.pl_df = pd.DataFrame()

# Title
st.title("📊 Stock Analysis: Consolidation & Breakout")

# Sidebar
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Select Data Source", ["Upload CSV/XLSX", "Fetch Real-Time (Yahoo Finance)"], key="data_source")
symbol = st.sidebar.text_input("Stock Symbol (e.g., AAPL)", value=st.session_state.symbol, key="symbol_input")

# File uploaders
primary_file = st.sidebar.file_uploader("Upload Stock Data (CSV/XLSX)", type=["csv", "xlsx"], key="primary_file")
secondary_file = st.sidebar.file_uploader("Upload Benchmark Data (CSV/XLSX)", type=["csv", "xlsx"], key="secondary_file")

# Auto-update date range
if primary_file and data_source == "Upload CSV/XLSX":
    try:
        temp_df = pd.read_csv(primary_file) if primary_file.name.endswith('.csv') else pd.read_excel(primary_file)
        if temp_df.empty:
            st.sidebar.error("Uploaded file is empty. Please upload a valid file.")
        else:
            temp_df['date'] = pd.to_datetime(temp_df['date'], errors='coerce')
            if not temp_df['date'].isna().all():
                file_min_date = temp_df['date'].min()
                file_max_date = temp_df['date'].max()
                st.session_state['auto_min_date'] = file_min_date
                st.session_state['auto_max_date'] = file_max_date
            else:
                st.session_state['auto_min_date'] = pd.to_datetime('2020-01-01')
                st.session_state['auto_max_date'] = pd.to_datetime('2025-06-25')
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded file: {str(e)}. Please check the file format.")
        st.session_state['auto_min_date'] = pd.to_datetime('2020-01-01')
        st.session_state['auto_max_date'] = pd.to_datetime('2025-06-25')
else:
    st.session_state['auto_min_date'] = pd.to_datetime('2020-01-01')
    st.session_state['auto_max_date'] = pd.to_datetime('2025-06-25')

# Dynamic date inputs
if 'aapl_df' in st.session_state and not st.session_state.aapl_df.empty:
    valid_dates = st.session_state.aapl_df['date'].dropna()
    min_date = valid_dates.min() if not valid_dates.empty else st.session_state['auto_min_date']
    max_date = valid_dates.max() if not valid_dates.empty else st.session_state['auto_max_date']
else:
    min_date = st.session_state['auto_min_date']
    max_date = st.session_state['auto_max_date']

from_date = st.sidebar.date_input("From Date", value=min_date, min_value=min_date, max_value=max_date, key="from_date_input", format="MM-DD-YYYY")
to_date = st.sidebar.date_input("To Date", value=max_date, min_value=min_date, max_value=max_date, key="to_date_input", format="MM-DD-YYYY")

st.sidebar.header("Chart Settings")
show_indicators = st.sidebar.multiselect(
    "Select Indicators",
    ["Bollinger Bands", "Ichimoku Cloud", "RSI", "MACD", "Stochastic", "ADX", "Fibonacci", "RVOL"],
    default=["Bollinger Bands", "RSI", "MACD"],
    key="indicators"
)
subplot_order = st.sidebar.multiselect(
    "Customize Subplot Order",
    ["Candlestick", "RSI", "MACD & Stochastic", "ADX & Volatility", "Volume", "Win/Loss Distribution"],
    default=["Candlestick", "RSI", "MACD & Stochastic", "ADX & Volatility", "Volume", "Win/Loss Distribution"],
    key="subplot_order"
)
html_report_type = st.sidebar.radio("HTML Report Type", ["Interactive (with Hover)", "Static Images"], key="html_report_type")

# Submit and Clear buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    submit = st.button("Submit")
with col2:
    clear = st.button("Clear")

# Handle Clear button
if clear:
    st.session_state.clear()
    st.cache_data.clear()
    st.session_state.data_loaded = False
    st.session_state.data_processed = False
    st.session_state.symbol = 'AAPL'
    st.session_state.aapl_df = pd.DataFrame()
    st.session_state.pl_df = pd.DataFrame()
    st.rerun()

# Validate symbol
def validate_symbol(symbol):
    return bool(re.match(r'^[A-Za-z0-9.]+$', symbol.strip()))

# Load data
@st.cache_data
def load_data(primary_file, data_source, symbol, start_date, end_date, secondary_file=None):
    aapl_df = pd.DataFrame()
    pl_df = pd.DataFrame()
    
    if data_source == "Upload CSV/XLSX" and primary_file:
        try:
            if primary_file.name.endswith('.csv'):
                aapl_df = pd.read_csv(primary_file)
            elif primary_file.name.endswith('.xlsx'):
                aapl_df = pd.read_excel(primary_file)
            
            if aapl_df.empty:
                st.error("Uploaded file is empty. Please upload a valid CSV/XLSX file.")
                return pd.DataFrame(), pd.DataFrame()
            
            aapl_df.columns = aapl_df.columns.str.lower().str.strip()
            
            benchmark_cols = ['year', 'start date', 'end date', 'profit/loss (percentage)', 'profit/loss (value)']
            if any(col in aapl_df.columns for col in benchmark_cols):
                st.error(
                    "The uploaded file appears to be benchmark data. Please upload it as 'Benchmark Data' or upload a stock data file with columns: date, open, high, low, close, volume."
                )
                return pd.DataFrame(), pd.DataFrame()
            
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            actual_cols = aapl_df.columns.tolist()
            missing_cols = [col for col in required_cols if col not in actual_cols]
            if missing_cols:
                st.error(
                    f"Missing required columns in stock data: {', '.join(missing_cols)}. Please upload a file with columns: date, open, high, low, close, volume."
                )
                st.write("Available columns:", actual_cols)
                st.write("Sample data (first 5 rows):", aapl_df.head())
                return pd.DataFrame(), pd.DataFrame()
            
            # Enhanced date parsing
            aapl_df['date_original'] = aapl_df['date']
            date_formats = ['%m/%d/%Y', '%m-%d-%Y', '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y']
            aapl_df['date'] = pd.to_datetime(aapl_df['date'], errors='coerce')
            if aapl_df['date'].isna().all():
                for fmt in date_formats:
                    aapl_df['date'] = pd.to_datetime(aapl_df['date_original'], format=fmt, errors='coerce')
                    if not aapl_df['date'].isna().all():
                        break
                if aapl_df['date'].isna().all():
                    st.error("No valid dates found in the uploaded file. Please ensure the 'date' column contains valid dates (e.g., MM-DD-YYYY, YYYY-MM-DD).")
                    st.write("Sample data (first 5 rows):", aapl_df.head())
                    return pd.DataFrame(), pd.DataFrame()
            
            aapl_df = aapl_df.dropna(subset=['date'])
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                aapl_df[col] = pd.to_numeric(aapl_df[col], errors='coerce')
            
            if aapl_df[numeric_cols].isna().any().any():
                st.warning("Missing values in numeric columns. Interpolating...")
                aapl_df = aapl_df.interpolate(method='linear', limit_direction='both')
            
            if not aapl_df['date'].empty:
                min_date = aapl_df['date'].min()
                max_date = aapl_df['date'].max()
                st.sidebar.write(f"File date range: {min_date.strftime('%m-%d-%Y')} to {max_date.strftime('%m-%d-%Y')}")
                
                if start_date < min_date or end_date > max_date:
                    st.error(f"Selected data range ({start_date.strftime('%m-%d-%Y')} to {end_date.strftime('%m-%d-%Y')}) is outside the file's range ({min_date.strftime('%m-%d-%Y')} to {max_date.strftime('%m-%d-%Y')}).")
                    return pd.DataFrame(), pd.DataFrame()
                
                aapl_df = aapl_df[(aapl_df['date'] >= start_date) & (aapl_df['date'] <= end_date)]
                if aapl_df.empty:
                    st.error(f"No data available for the selected data range ({start_date.strftime('%m-%d-%Y')} to {end_date.strftime('%m-%d-%Y')}). Please adjust the date range.")
                    return pd.DataFrame(), pd.DataFrame()
                
                if len(aapl_df) < 52:
                    st.error(f"Insufficient data points ({len(aapl_df)}) in selected data range. Please select a range with at least 52 trading days.")
                    return pd.DataFrame(), pd.DataFrame()
            
            if 'vwap' not in aapl_df.columns:
                st.warning("VWAP column is missing. VWAP plot will be skipped (optional).")
        
        except Exception as e:
            st.error(f"Error loading stock data: {str(e)}. Please check the file format and content.")
            logger.error(f"File upload error: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    elif data_source == "Fetch Real-Time (Yahoo Finance)":
        try:
            symbol = symbol.strip()
            if not validate_symbol(symbol):
                st.error(f"Invalid symbol '{symbol}'. Please enter a single valid stock symbol (e.g., AAPL, MSFT, BRK.B).")
                return pd.DataFrame(), pd.DataFrame()
            
            aapl_df = yf.download(symbol, start=start_date, end=end_date + timedelta(days=1), progress=False)
            if aapl_df.empty:
                st.error(f"Failed to fetch {symbol} data from Yahoo Finance. Please check the symbol, date range, or internet connection.")
                return pd.DataFrame(), pd.DataFrame()
            
            aapl_df = aapl_df.reset_index().rename(columns={
                'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            })
            aapl_df['date'] = pd.to_datetime(aapl_df['date'], errors='coerce')
            aapl_df = aapl_df.dropna(subset=['date'])
            
            if aapl_df.empty:
                st.error(f"No valid data fetched for {symbol}. Please try a different symbol or date range.")
                return pd.DataFrame(), pd.DataFrame()
            
            aapl_df = aapl_df.interpolate(method='linear', limit_direction='both')
            
            if len(aapl_df) < 52:
                st.error(f"Insufficient data points ({len(aapl_df)}) for {symbol}. Please select a wider date range (at least 52 trading days).")
                return pd.DataFrame(), pd.DataFrame()
        
        except Exception as e:
            st.error(f"Error fetching {symbol} data from Yahoo Finance: {str(e)}. Please check the symbol, date range, or try uploading a file.")
            logger.error(f"Yahoo Finance fetch error: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    if secondary_file:
        try:
            if secondary_file.name.endswith('.csv'):
                pl_df = pd.read_csv(secondary_file)
            elif secondary_file.name.endswith('.xlsx'):
                pl_df = pd.read_excel(secondary_file)
            pl_df['Start Date'] = pd.to_datetime(pl_df['Start Date'], errors='coerce')
            pl_df['End Date'] = pd.to_datetime(pl_df['End Date'], errors='coerce')
            if pl_df[['Start Date', 'End Date', 'Profit/Loss (Percentage)']].isnull().any().any():
                st.warning("Benchmark data contains null values. Proceeding without benchmark.")
                pl_df = pd.DataFrame()
        except Exception as e:
            st.warning(f"Error loading benchmark data: {str(e)}. Proceeding without benchmark.")
    
    return aapl_df, pl_df

# Load data on submit
if submit and not st.session_state.data_processed:
    st.session_state.data_loaded = True
    st.session_state.symbol = st.session_state.symbol_input
    st.session_state.start_date = pd.to_datetime(from_date)
    st.session_state.end_date = pd.to_datetime(to_date)
    aapl_df, pl_df = load_data(primary_file, data_source, st.session_state.symbol, st.session_state.start_date, st.session_state.end_date, secondary_file)
    st.session_state.aapl_df = aapl_df
    st.session_state.pl_df = pl_df
    st.session_state.data_processed = True
elif not st.session_state.data_loaded:
    st.info("Please enter a symbol, select a data source, select a date range, and click 'Submit' to load data.")
    st.stop()

if st.session_state.aapl_df.empty:
    st.error(f"Failed to load valid data for {st.session_state.symbol}. Please check the file, symbol, or date range.")
    st.stop()

# Calculate metrics
@st.cache_data
def calculate_metrics(df):
    df['daily_return'] = df['close'].pct_change().fillna(0)
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
    
    average_return = df['daily_return'].mean() * 100
    volatility = df['daily_return'].std() * np.sqrt(252) * 100
    win_ratio = (df['daily_return'] > 0).mean() * 100
    annualized_return = ((1 + df['cumulative_return'].iloc[-1]) ** (252 / len(df))) - 1 if len(df) > 0 else 0
    sharpe_ratio = (annualized_return - 0.03) / (volatility / 100) if volatility > 0 else 0
    downside_returns = df['daily_return'][df['daily_return'] < 0]
    sortino_ratio = (annualized_return - 0.03) / (downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else 0
    drawdowns = df['close'] / df['close'].cummax() - 1
    max_drawdown = drawdowns.min() * 100
    largest_loss = df['daily_return'].min() * 100
    largest_loss_date = df.loc[df['daily_return'].idxmin(), 'date'].strftime('%m-%d-%Y') if not df['daily_return'].empty else "N/A"
    largest_gain = df['daily_return'].max() * 100
    largest_gain_date = df.loc[df['daily_return'].idxmax(), 'date'].strftime('%m-%d-%Y') if not df['daily_return'].empty else "N/A"
    
    return {
        'Average Return': average_return,
        'Volatility': volatility,
        'Win Ratio': win_ratio,
        'CAGR': annualized_return * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Max Drawdown': max_drawdown,
        'Largest Loss': largest_loss,
        'Largest Loss Date': largest_loss_date,
        'Largest Gain': largest_gain,
        'Largest Gain Date': largest_gain_date
    }

if 'aapl_metrics' not in st.session_state or submit:
    st.session_state.aapl_metrics = calculate_metrics(st.session_state.aapl_df)

# Detect consolidation and breakout
@st.cache_data
def detect_consolidation_breakout(df):
    try:
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['std_dev'] = df['close'].rolling(window=20).std()
        df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close']).rsi()
        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['signal'] = macd.macd_signal()
        df['macd_diff'] = df['macd'] - df['signal']
        stochastic = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stochastic_k'] = stochastic.stoch()
        df['stochastic_d'] = stochastic.stoch_signal()
        df['adx'] = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx()
        df['rvol'] = df['volume'] / df['volume'].rolling(window=20).mean()
        df['is_consolidation'] = (df['atr'] < df['atr'].mean() * 0.8) & (df['adx'] < 20)
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        high_20 = df['high'].rolling(window=20).max()
        low_20 = df['low'].rolling(window=20).min()
        df['fib_236'] = low_20 + 0.236 * (high_20 - low_20)
        df['fib_382'] = low_20 + 0.382 * (high_20 - low_20)
        df['fib_50'] = low_20 + 0.5 * (high_20 - low_20)
        df['fib_618'] = low_20 + 0.618 * (high_20 - low_20)
        ichimoku = ta.trend.IchimokuIndicator(high=df['high'], low=df['low'])
        df['tenkan_sen'] = ichimoku.ichimoku_conversion_line()
        df['kijun_sen'] = ichimoku.ichimoku_base_line()
        df['senkou_span_a'] = ichimoku.ichimoku_a()
        df['senkou_span_b'] = ichimoku.ichimoku_b()
        close_exceeds_resistance = df['close'] > df['resistance'].shift(1)
        volume_condition = df['volume'] > df['volume'].mean() * 0.8
        rsi_condition = (df['rsi'] > 30) & (df['rsi'] < 80)
        macd_condition = df['macd'] > df['signal']
        stochastic_condition = df['stochastic_k'] > df['stochastic_d']
        df['buy_signal'] = close_exceeds_resistance & volume_condition & rsi_condition & macd_condition & stochastic_condition
        close_below_support = df['close'] < df['support'].shift(1)
        df['sell_signal'] = close_below_support & volume_condition & (df['rsi'] < 70) & (df['macd'] < df['signal']) & (df['stochastic_k'] < df['stochastic_d'])
        df['stop_loss'] = df['close'] - 1.5 * df['atr']
        df['take_profit'] = df['close'] + 2 * 1.5 * df['atr']
        return df
    except Exception as e:
        st.error(f"Error calculating technical indicators: {str(e)}. Please check data integrity.")
        logger.error(f"Indicator calculation error: {str(e)}")
        return df

if 'aapl_df_processed' not in st.session_state or submit:
    st.session_state.aapl_df = detect_consolidation_breakout(st.session_state.aapl_df)
    st.session_state.aapl_df_processed = st.session_state.aapl_df.copy()

# Backtesting
@st.cache_data
def backtest_strategy(df):
    trades = []
    position = None
    for i in range(1, len(df)):
        if df['buy_signal'].iloc[i-1]:
            if position is None:
                stop_loss = df['stop_loss'].iloc[i] if pd.notna(df['stop_loss'].iloc[i]) else df['close'].iloc[i] * 0.80
                take_profit = df['take_profit'].iloc[i] if pd.notna(df['take_profit'].iloc[i]) else df['close'].iloc[i] * 1.20
                position = {
                    'entry_date': df['date'].iloc[i],
                    'entry_price': df['close'].iloc[i],
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
        elif position:
            if pd.notna(df['low'].iloc[i]) and pd.notna(df['high'].iloc[i]):
                if df['low'].iloc[i] <= position['stop_loss']:
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': df['date'].iloc[i],
                        'entry_price': position['entry_price'],
                        'exit_price': position['stop_loss'],
                        'return': (position['stop_loss'] - position['entry_price']) / position['entry_price'] * 100
                    })
                    position = None
                elif df['high'].iloc[i] >= position['take_profit']:
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': df['date'].iloc[i],
                        'entry_price': position['entry_price'],
                        'exit_price': position['take_profit'],
                        'return': (position['take_profit'] - position['entry_price']) / position['entry_price'] * 100
                    })
                    position = None
    
    if not trades:
        return {'Win Rate': 0, 'Profit Factor': 0, 'Total Return': 0, 'Trades': 0}
    
    trades_df = pd.DataFrame(trades)
    win_rate = (trades_df['return'] > 0).mean() * 100 if not trades_df.empty else 0
    gross_profit = trades_df[trades_df['return'] > 0]['return'].sum() if (trades_df['return'] > 0).any() else 0
    gross_loss = -trades_df[trades_df['return'] < 0]['return'].sum() if (trades_df['return'] < 0).any() else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    total_return = trades_df['return'].sum() if not trades_df.empty else 0
    return {
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Total Return': total_return,
        'Trades': len(trades)
    }

if 'backtest_results' not in st.session_state or submit:
    st.session_state.backtest_results = backtest_strategy(st.session_state.aapl_df)

# Technical signals
@st.cache_data
def get_signals(df):
    signals = {
        'RSI': 'Buy' if df['rsi'].iloc[-1] < 40 else 'Sell' if df['rsi'].iloc[-1] > 70 else 'Neutral',
        'MACD': 'Buy' if df['macd'].iloc[-1] > df['signal'].iloc[-1] else 'Sell',
        'Stochastic': 'Buy' if (df['stochastic_k'].iloc[-1] < 20 and df['stochastic_k'].iloc[-1] > df['stochastic_d'].iloc[-1]) else 'Sell' if (df['stochastic_k'].iloc[-1] > 80) else 'Neutral',
        'Ichimoku': 'Buy' if (df['close'].iloc[-1] > df['senkou_span_a'].iloc[-1] and df['close'].iloc[-1] > df['senkou_span_b'].iloc[-1]) else 'Sell',
        'ADX': 'Strong Trend' if df['adx'].iloc[-1] > 25 else 'Weak Trend'
    }
    return signals

if 'signals' not in st.session_state or submit:
    st.session_state.signals = get_signals(st.session_state.aapl_df)

# Breakout timeframe
@st.cache_data
def get_breakout_timeframe(df):
    if df['is_consolidation'].iloc[-1]:
        return "Breakout expected within 1-5 days"
    elif df['buy_signal'].iloc[-1]:
        return "Breakout detected; confirm within 1-3 days"
    else:
        return "No breakout expected soon"

if 'breakout_timeframe' not in st.session_state or submit:
    st.session_state.breakout_timeframe = get_breakout_timeframe(st.session_state.aapl_df)

# Scoring system
@st.cache_data
def calculate_score(metrics, signals):
    performance_score = min(metrics['CAGR'], 30)
    risk_score = min((metrics['Sharpe Ratio'] * 5 + metrics['Sortino Ratio'] * 5), 20)
    technical_score = sum([10 if s in ['Buy', 'Strong Trend'] else 0 for s in signals.values()])
    volume_score = 20 if st.session_state.aapl_df['volume'].iloc[-1] > st.session_state.aapl_df['volume'].mean() else 10
    total_score = performance_score + risk_score + technical_score + volume_score
    recommendation = 'Buy' if total_score > 70 else 'Hold' if total_score > 50 else 'Underperformance'
    return {
        'Performance': performance_score,
        'Risk': risk_score,
        'Technical': technical_score,
        'Volume': volume_score,
        'Total': total_score,
        'Recommendation': recommendation
    }

if 'score' not in st.session_state or submit:
    st.session_state.score = calculate_score(st.session_state.aapl_metrics, st.session_state.signals)

# Price prediction
@st.cache_data
def predict_price(df):
    X = np.arange(len(df['close'])).reshape(-1, 1)
    y = df['close'].values
    model = LinearRegression()
    model.fit(X, y)
    next_days = np.arange(len(df['close']), len(df['close']) + 5).reshape(-1, 1)
    predicted_prices = model.predict(next_days)
    last_date = df['date'].iloc[-1]
    if pd.isna(last_date):
        last_date = pd.Timestamp.now(tz='America/New_York')
    date_range = pd.date_range(start=last_date, periods=5, freq='B')
    return pd.DataFrame({
        'date': date_range,
        'predicted_close': predicted_prices
    })

if 'price_prediction' not in st.session_state or submit:
    st.session_state.price_prediction = predict_price(st.session_state.aapl_df)

# Alerts
@st.cache_data
def get_alerts(df):
    df['daily_change'] = df['close'].pct_change() * 100
    alerts = []
    for idx, row in df.iterrows():
        if abs(row['daily_change']) > 2:
            signal_strength = 'High' if abs(row['daily_change']) > 5 else 'Moderate'
            alert_type = 'BUY' if row['buy_signal'] else 'SELL' if row['sell_signal'] else 'PRICE MOVE'
            description = f"{row['daily_change']:.2f}% change detected"
            alerts.append({
                'date': row['date'].strftime('%m-%d-%Y'),
                'type': alert_type,
                'price': row['close'],
                'volume': row['volume'],
                'strength': signal_strength,
                'description': description
            })
    return alerts if alerts else [{"date": "N/A", "type": "NONE", "price": 0, "volume": 0, "strength": "N/A", "description": "No significant price movements (>2%) detected."}]

if 'alerts' not in st.session_state or submit:
    st.session_state.alerts = get_alerts(st.session_state.aapl_df)

# Candlestick chart
subplot_titles = [s for s in subplot_order]
row_heights = [0.35 if s == "Candlestick" else 0.15 if s == "Win/Loss Distribution" else 0.1 for s in subplot_order]
fig = make_subplots(rows=len(subplot_order), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    subplot_titles=subplot_titles, row_heights=row_heights)

def add_candlestick_trace(fig, df, row):
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'rvol']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns for candlestick chart: {missing_cols}. Please ensure data is correctly processed.")
        logger.error(f"Missing columns for candlestick: {missing_cols}")
        return
    
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date'] = df['date'].fillna(pd.NaT)

    # Handle NaN values
    df[['volume', 'rsi', 'rvol']] = df[['volume', 'rsi', 'rvol']].fillna(0)
    
    signal_labels = []
    hovertemplate_custom = []
    for _, row in df.iterrows():
        if row.get('buy_signal', False):
            signal_labels.append('Buy Signal')
            hovertemplate_custom.append('<b style="color:darkgreen">🟢 BUY SIGNAL</b><br>')
        elif row.get('sell_signal', False):
            signal_labels.append('Sell Signal')
            hovertemplate_custom.append('<b style="color:darkred">🔴 SELL SIGNAL</b><br>')
        else:
            signal_labels.append('')
            hovertemplate_custom.append('')

    customdata = np.array(
        [signal_labels, df['volume'], df['rsi'], df['rvol']],
        dtype=object
    ).T

    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Price",
        increasing_line_color='#4CAF50',
        decreasing_line_color='#f44336',
        customdata=customdata,
        hovertemplate='%{customdata[0]}' +
                      'Date: %{x|%m-%d-%Y}<br>' +
                      'Month: %{x|%B}<br>' +
                      'Open: $%{open:.2f}<br>' +
                      'High: $%{high:.2f}<br>' +
                      'Low: $%{low:.2f}<br>' +
                      'Close: $%{close:.2f}<br>' +
                      'Volume: %{customdata[1]:,.0f}<br>' +
                      'RSI: %{customdata[2]:.2f}<br>' +
                      'RVOL: %{customdata[3]:.2f}<extra></extra>'
    ), row=row, col=1)

    if "Bollinger Bands" in show_indicators and 'ma20' in df.columns and 'std_dev' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['ma20'] + 2*df['std_dev'], name="Bollinger Upper", line=dict(color="#0288d1")), row=row, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['ma20'] - 2*df['std_dev'], name="Bollinger Lower", line=dict(color="#0288d1"), fill='tonexty', fillcolor='rgba(2,136,209,0.1)'), row=row, col=1)
    if "Ichimoku Cloud" in show_indicators:
        fig.add_trace(go.Scatter(x=df['date'], y=df['senkou_span_a'], name="Senkou Span A", line=dict(color="#4CAF50"), fill='tonexty', fillcolor='rgba(76,175,80,0.2)'), row=row, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['senkou_span_b'], name="Senkou Span B", line=dict(color="#f44336"), fill='toself', fillcolor='rgba(244,67,54,0.2)'), row=row, col=1)
    if "Fibonacci" in show_indicators:
        for level, color in [('fib_236', '#ff9800'), ('fib_382', '#e91e63'), ('fib_50', '#9c27b0'), ('fib_618', '#3f51b5')]:
            fig.add_trace(go.Scatter(x=df['date'], y=df[level], name=f"Fib {level[-3:]}%", line=dict(color=color, dash='dash')), row=row, col=1)
    buy_signals = df[df['buy_signal'] == True]
    for _, signal in buy_signals.iterrows():
        fig.add_annotation(x=signal['date'], y=signal['high'], text="Buy", showarrow=True, arrowhead=2, ax=0, ay=-30, font=dict(color="#000000"), row=row, col=1)
    sell_signals = df[df['sell_signal'] == True]
    for _, signal in sell_signals.iterrows():
        fig.add_annotation(x=signal['date'], y=signal['low'], text="Sell", showarrow=True, arrowhead=2, ax=0, ay=30, font=dict(color="#000000"), row=row, col=1)
    if not buy_signals.empty:
        latest_buy = buy_signals.iloc[-1]
        risk = latest_buy['close'] - latest_buy['stop_loss']
        reward = latest_buy['take_profit'] - latest_buy['close']
        rr_ratio = reward / risk if risk > 0 else 'N/A'
        fig.add_hline(y=latest_buy['stop_loss'], line_dash="dash", line_color="#f44336", annotation_text="Stop-Loss", row=row, col=1)
        fig.add_hline(y=latest_buy['take_profit']]=, line_dash="dash", line_color="#4CAF50", annotation_text="Take-Profit", row=row, col=1)
        fig.add_trace(go.Scatter(x=[latest_buy['date'], latest_buy['date']], y=[latest_buy['stop_loss'], latest_buy['take_profit']],
                                 mode='lines', line=dict(color='rgba(76,175,80,0.2)'), fill='toself', fillcolor='rgba(76,175,80,0.2)',
                                 hovertext=[f"Risk-Reward Ratio: {rr_ratio:.2f}" if isinstance(rr_ratio, float) else f"Risk-Reward Ratio: {rr_ratio}"], hoverinfo='text'), row=row, col=1)

def add_rsi_trace(fig, df, row):
    if 'rsi' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['rsi'], name="RSI", line=dict(color="#9c27b0")), row=row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#f44336", row=row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#4CAF50", row=row, col=1)

def add_macd_trace(fig, df, row):
    if 'macd' in df.columns and 'signal' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['macd'], name="MACD", line=dict(color="#0288d1")), row=row, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['signal'], name="MACD Signal", line=dict(color="#ff9800")), row=row, col=1)
        fig.add_trace(go.Bar(x=df['date'], y=df['macd_diff'], name="MACD Histogram", marker_color="#607d8b"), row=row, col=1)
    if "Stochastic" in show_indicators and 'stochastic_k' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['stochastic_k'], name="Stochastic %K", line=dict(color="#388e3c"), yaxis="y2"), row=row, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['stochastic_d'], name="Stochastic %D", line=dict(color="#ff5722"), yaxis="y2"), row=row, col=1)
        fig.update_layout(yaxis2=dict(overlaying='y', side='right'))

def add_adx_volatility_trace(fig, df, row):
    if 'adx' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['adx'], name="ADX", line=dict(color="#3f51b5")), row=row, col=1)
        fig.add_hline(y=25, line_dash="dash", line_color="#4CAF50", row=row, col=1)
    if 'rvol' in df.columns:
        fig.add_trace(go.Scatter(x=df['date'], y=df['rvol'], name="RVOL", line=dict(color="#2196f3"), yaxis="y3"), row=row, col=1)
        fig.update_layout(yaxis3=dict(overlaying='y', side='right'))

def add_volume_trace(fig, df, row):
    if 'volume' in df.columns:
        fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name="Volume", marker_color="#607d8b"), row=row, col=1)
        if 'vwap' in df.columns:
            fig.add_trace(go.Scatter(x=df['date'], y=df['vwap'], name="VWAP", line=dict(color="#0288d1")), row=row, col=1)

def add_win_loss_trace(fig, df, row):
    if 'daily_return' in df.columns:
        returns = df['daily_return'].dropna()
        if not returns.empty:
            fig.add_trace(go.Histogram(x=returns*100, name="Returns", marker_color="#607d8b"), row=row, col=1)

for i, subplot in enumerate(subplot_order, 1):
    if subplot == "Candlestick":
        add_candlestick_trace(fig, st.session_state.aapl_df, i)
    elif subplot == "RSI":
        add_rsi_trace(fig, st.session_state.aapl_df, i)
    elif subplot == "MACD & Stochastic":
        add_macd_stochastic_trace(fig, st.session_state.aapl_df, i)
    elif subplot == "ADX & Volatility":
        add_adx_volatility_trace(fig, st.session_state.aapl_df, i)
    elif subplot == "Volume":
        add_volume_trace(fig, st.session_state.aapl_df, i)
    elif subplot == "Win/Loss Distribution":
        add_win_loss_trace(fig, st.session_state.aapl_df, i)

fig.update_layout(height=200 * len(subplot_order), title=f"{st.session_state.symbol} Analysis", showlegend=True)
fig.update_xaxes(rangeslider_visible=True, tickformat='%Y-%m-%d')

def on_click(trace, points, state):
    if points.point:
        idx = points.point[0]
        row = st.session_state.aapl_df.iloc[idx]
        st.session_state.trade_details = {
            'Date': row['date'].strftime('%Y-%m-%d'),
            'Close': row['close'],
            'Stop-Loss': row['stop_loss'],
            'Take-Profit': row['take_profit'],
            'Buy Signal': row['buy_signal']
        }

for trace in fig.data:
    trace.on_click(on_click)

# Display metrics
st.header("Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Average Return", value=f"{st.session_state.aapl_metrics['avg_return']:.2f}%")
with col2:
    st.metric(label="Volatility", value=f"{st.session_state.aapl_metrics['volatility']:.2f}%")
with col3:
    st.metric(label="Win Ratio", value=f"{st.session_state.aapl_metrics['win_ratio']:.2f}%")
with col4:
    st.metric(label="Max Drawdown", value=f"{st.session_state.aapl_metrics['max_drawdown']:.2f}%")

# Alerts
st.header("Alerts")
with st.expander("View Alerts"):
    alerts_df = pd.DataFrame(st.session_state.alerts)
    st.table(alerts_df)

# Backtest results
st.header("Backtest Results")
col1, col4 = st.columns(4)
with col1:
    st.metric(label="Win Rate", value=f"{st.session_state.backtest_results['win_rate']:.2f}%")
with col2:
    st.metric(label="Profit Factor", value=f"{st.session_state.backtest_results['profit_factor']:.2f}")
with col3:
    st.metric(label="Total Return", value=f"{st.session_state.backtest_results['total_return']:.2f}%")
with col4:
    st.metric(label="Trades", value=f"{st.session_state.backtest_results['trades']}")

# Price prediction
st.header("Price Prediction")
pred_fig = go.Figure()
pred_fig.add_trace(go.Scatter(x=st.session_state.price_prediction['date'], y=st.session_state.price_prediction['predicted_close'], mode='lines+markers', name="Predicted")))
pred_fig.update_layout(height=400, title="Price Prediction")
st.plotly_chart(pred_fig)

# Decision dashboard
st.header("Decision Dashboard")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Recommendation", value=st.session_state.score['Recommendation'])
with col2:
    st.metric(label="Total Score", value=f"{st.session_state.score['Total']:.2f}")
with col3:
    st.metric(label="Breakout Timeframe", value=st.session_state.breakout_timeframe)

# Trade details
if st.session_state.trade_details:
    st.header("Trade Details")
    details = st.session_state.trade_details
    try:
        rr_ratio = (details['Take-Profit'] - details['Close']) / (details['Close'] - details['Stop-Loss']) if (details['Close'] - details['Stop-Loss']) > 0 else 'N/A'
        st.markdown(
            <div class='trade-details'>
                <b>Date:</b> {details['Date']}<br>
                <b>Close:</b> ${details['Close']:.2f}<br>
                <b>Stop-Loss:</b> ${details['Stop-Loss']:.2f}<br>
                <b>Take-Profit:</b> ${details['Take-Profit']:.2f}<br>
                <b>Buy Signal:</b> {details['Buy Signal']}<br>
                <b>Risk-Reward Ratio:</b> {rr_ratio if isinstance(rr_ratio, float) else 'N/A'}
            </div>",
            unsafe_html=True
        )
    except Exception as e:
        st.error(f"Error displaying trade details: {str(e)}")
    st.markdown()

# Chart display
st.plotly_chart(fig)

# Benchmark comparison
if not st.session_state.pl_df.empty:
    st.header("Benchmark Comparison")
    bench_fig = go.Figure()
    bench_fig.add_trace(go.Scatter(x=st.session_state.aapl_df['date'], y=st.session_state.aapl_df['cumulative_return'], name="Actual"))
    bench_fig.add_trace(go.Scatter(x=st.session_state.pl_df['End Date'], y=(1 + st.session_state.pl_df['Profit/Loss (Percentage)'] / 100).cumprod() - 1, name="Benchmark"))
    bench_fig.update_layout(height=300, title="Benchmark Comparison")
    st.plotly_chart(bench_fig)

# Seasonality
st.header("Seasonality")
st.session_state.aapl_df['month'] = st.session_state.aapl_df['date'].dt.month
st.session_state.aapl_df['year'] = st.session_state.aapl_df['date'].dt.year
monthly_returns = st.session_state.aapl_df.groupby(['year'], ['month'])['daily_return'].mean().unstack()
heatmap_fig = go.Figure(data=go.Heatmap(
    z=monthly_returns.values,
    x=[calendar.month_name[i] for i in monthly_returns.columns],
    y=monthly_returns.index,
    colorscale='Plasma'
))
heatmap_fig.update_layout(height=300, title="Seasonality Heatmap")
st.plotly_chart(heatmap)

# Export data as CSV and Excel
st.header("Export Data and Reports")
if not st.session_state.aapl_df.empty:
    valid_dates = st.session_state.aapl_df['date'].dropna()
    if not valid_dates.empty:
        min_date = valid_dates.min().strftime('%m-%d-%Y')
        max_date = valid_dates.max().strftime('%m-%d-%Y')
    else:
        min_date = '01-01-2020'
        max_date = '06-24-2025'
    csv_buffer = io.StringIO()
    st.session_state.aapl_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    st.download_button("Download Stock Data (CSV)", csv_buffer.getvalue(), file_name=f"{st.session_state.symbol}_analysis_data_{min_date}_to_{max_date}.csv", mime="text/csv")

    excel_buffer = io.BytesIO()
    st.session_state.aapl_df.to_excel(excel_buffer, index=False, engine='openpyxl')
    excel_buffer.seek(0)
    st.download_button("Download Stock Data (Excel)", excel_buffer, file_name=f"{st.session_state.symbol}_analysis_data_{min_date}_to_{max_date}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Export PDF report
if not st.session_state.aapl_df.empty:
    valid_dates = st.session_state.aapl_df['date'].dropna()
    if not valid_dates.empty:
        min_date = valid_dates.min().strftime('%m-%d-%Y')
        max_date = valid_dates.max().strftime('%m-%d-%Y')
    else:
        min_date = '01-01-2020'
        max_date = '06-24-2025'
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(50, 750, f"{st.session_state.symbol} Stock Analysis Report ({min_date} to {max_date})")
    c.drawString(50, 730, f"Date: {datetime.now(pytz.timezone('America/New_York')).strftime('%m-%d-%Y %I:%M %p EDT')}")
    c.drawString(50, 710, f"Recommendation: {st.session_state.score['Recommendation']}")
    c.drawString(50, 690, "Scores:")
    c.drawString(70, 670, f"- Performance: {st.session_state.score['Performance']:.1f}/30")
    c.drawString(70, 650, f"- Risk: {st.session_state.score['Risk']:.1f}/20")
    c.drawString(70, 630, f"- Technical: {st.session_state.score['Technical']:.1f}/30")
    c.drawString(70, 610, f"- Volume: {st.session_state.score['Volume']:.1f}/20")
    c.drawString(70, 590, f"- Total: {st.session_state.score['Total']:.1f}/100")
    c.drawString(50, 570, "Key Metrics:")
    c.drawString(70, 550, f"- Average Return: {st.session_state.aapl_metrics['Average Return']:.2f}%")
    c.drawString(70, 530, f"- Volatility: {st.session_state.aapl_metrics['Volatility']:.2f}%")
    c.drawString(70, 510, f"- Win Ratio: {st.session_state.aapl_metrics['Win Ratio']:.2f}%")
    c.drawString(70, 490, f"- Max Drawdown: {st.session_state.aapl_metrics['Max Drawdown']:.2f}%")
    c.drawString(70, 470, f"- Largest Loss: {st.session_state.aapl_metrics['Largest Loss']:.2f}% on {st.session_state.aapl_metrics['Largest Loss Date']}")
    c.drawString(70, 450, f"- Largest Gain: {st.session_state.aapl_metrics['Largest Gain']:.2f}% on {st.session_state.aapl_metrics['Largest Gain Date']}")
    c.drawString(50, 430, "Latest Trade Setup:")
    stop_loss_value = st.session_state.aapl_df['stop_loss'].iloc[-1] if 'stop_loss' in st.session_state.aapl_df.columns and not st.session_state.aapl_df['stop_loss'].iloc[-1] is None else 0.0
    take_profit_value = st.session_state.aapl_df['take_profit'].iloc[-1] if 'take_profit' in st.session_state.aapl_df.columns and not st.session_state.aapl_df['take_profit'].iloc[-1] is None else 0.0
    c.drawString(70, 410, f"- Date: {st.session_state.aapl_df['date'].iloc[-1].strftime('%m-%d-%Y')}")
    c.drawString(70, 390, f"- Entry: ${st.session_state.aapl_df['close'].iloc[-1]:.2f}")
    c.drawString(70, 370, f"- Stop-Loss: ${stop_loss_value:.2f}")
    c.drawString(70, 350, f"- Take-Profit: ${take_profit_value:.2f}")
    c.drawString(50, 330, "Backtesting Results:")
    c.drawString(70, 310, f"- Win Rate: {st.session_state.backtest_results['Win Rate']:.2f}%")
    c.drawString(70, 290, f"- Profit Factor: {st.session_state.backtest_results['Profit Factor']:.2f}")
    c.drawString(70, 270, f"- Total Return: {st.session_state.backtest_results['Total Return']:.2f}%")
    c.drawString(70, 250, f"- Trades: {st.session_state.backtest_results['Trades']}")
    c.showPage()
    c.save()
    pdf_buffer.seek(0)
    st.download_button("Download PDF Report", pdf_buffer, file_name=f"{st.session_state.symbol}_investment_report_{min_date}_to_{max_date}.pdf", mime="application/pdf")

# Generate HTML alerts table
def generate_alerts_table_html(alerts_data):
    if not alerts_data or alerts_data[0]['type'] == 'NONE':
        return "<p>No price movement alerts detected.</p>"
    
    html = """
    <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; width: 100%;">
        <thead style="background-color: #f0f0f0;">
            <tr>
                <th>Date</th>
                <th>Alert Type</th>
                <th>Price</th>
                <th>Volume</th>
                <th>Signal Strength</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for alert in alerts_data:
        row_color = "#e8f5e8" if alert['type'] == 'BUY' else "#ffe8e8" if alert['type'] == 'SELL' else "#fff3e0"
        html += f"""
            <tr style="background-color: {row_color};">
                <td>{alert['date']}</td>
                <td><strong>{alert['type']}</strong></td>
                <td>${alert['price']:.2f}</td>
                <td>{alert['volume']:,.0f}</td>
                <td>{alert['strength']}</td>
                <td>{alert['description']}</td>
            </tr>
        """
    
    html += """
        </tbody>
    </table>
    """
    return html

# Export HTML report
if not st.session_state.aapl_df.empty:
    valid_dates = st.session_state.aapl_df['date'].dropna()
    if not valid_dates.empty:
        min_date = valid_dates.min().strftime('%m-%d-%Y')
        max_date = valid_dates.max().strftime('%m-%d-%Y')
    else:
        min_date = '01-01-2020'
        max_date = '06-24-2025'
    if html_report_type == "Interactive (with Hover)":
        candlestick_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
        bench_html = fig_bench.to_html(include_plotlyjs='cdn', full_html=False) if fig_bench else ""
        heatmap_html = fig_heatmap.to_html(include_plotlyjs='cdn', full_html=False)
        pred_html = fig_pred.to_html(include_plotlyjs='cdn', full_html=False)
    else:
        candlestick_img = fig.to_image(format="png")
        candlestick_img_b64 = base64.b64encode(candlestick_img).decode()
        bench_img_b64 = fig_bench.to_image(format="png") if fig_bench else None
        bench_img_b64 = base64.b64encode(bench_img_b64).decode() if bench_img_b64 else ""
        heatmap_img = fig_heatmap.to_image(format="png")
        heatmap_img_b64 = base64.b64encode(heatmap_img).decode()
        pred_img = fig_pred.to_image(format="png")
        pred_img_b64 = base64.b64encode(pred_img).decode()
        candlestick_html = f'<img src="data:image/png;base64,{candlestick_img_b64}" alt="Candlestick Chart">'
        bench_html = f'<img src="data:image/png;base64,{bench_img_b64}" alt="Benchmark Chart">' if bench_img_b64 else ""
        heatmap_html = f'<img src="data:image/png;base64,{heatmap_img_b64}" alt="Seasonality Heatmap">'
        pred_html = f'<img src="data:image/png;base64,{pred_img_b64}" alt="Price Prediction">'

    alerts_table_html = generate_alerts_table_html(st.session_state.alerts)
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{symbol} Stock Analysis Report ({start_date} to {end_date})</title>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #ffffff; color: #000000; margin: 20px; }}
            h1, h2 {{ color: #0288d1; }}
            .metric-box {{ background-color: #e0e0e0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .section {{ margin-bottom: 20px; }}
            .plotly-graph-div {{ max-width: 100%; }}
            .alert-box {{ background-color: #fff3e0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f0f0f0; }}
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>{symbol} Stock Analysis Report ({start_date} to {end_date})</h1>
        <p><b>Date:</b> {date}</p>
        
        <div class="section">
            <h2>Recommendation</h2>
            <div class="metric-box">
                <p><b>Recommendation:</b> {recommendation}</p>
                <p><b>Total Score:</b> {total_score:.1f}/100</p>
                <p><b>Breakout Timeframe:</b> {breakout_timeframe}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Key Metrics</h2>
            <div class="metric-box">
                <p><b>Average Return:</b> {average_return:.2f}%</p>
                <p><b>Volatility:</b> {volatility:.2f}%</p>
                <p><b>Win Ratio:</b> {win_ratio:.2f}%</p>
                <p><b>Max Drawdown:</b> {max_drawdown:.2f}%</p>
                <p><b>Largest Loss:</b> {largest_loss:.2f}% on {largest_loss_date}</p>
                <p><b>Largest Gain:</b> {largest_gain:.2f}% on {largest_gain_date}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Price Movement Alerts</h2>
            {alerts_table_html}
        </div>
        
        <div class="section">
            <h2>Backtesting Results</h2>
            <div class="metric-box">
                <p><b>Win Rate:</b> {win_rate:.2f}%</p>
                <p><b>Profit Factor:</b> {profit_factor:.2f}</p>
                <p><b>Total Return:</b> {total_return:.2f}%</p>
                <p><b>Trades:</b> {trades}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Latest Trade Setup</h2>
            <div class="metric-box">
                <p><b>Date:</b> {latest_date}</p>
                <p><b>Entry:</b> ${entry:.2f}</p>
                <p><b>Stop-Loss:</b> ${stop_loss:.2f}</p>
                <p><b>Take-Profit:</b> ${take_profit:.2f}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Price Prediction</h2>
            {pred_html}
        </div>
        
        <div class="section">
            <h2>Candlestick & Technical Analysis</h2>
            {candlestick_html}
        </div>
        
        <div class="section">
            <h2>Benchmark Comparison</h2>
            {bench_html}
        </div>
        
        <div class="section">
            <h2>Seasonality Analysis</h2>
            {heatmap_html}
        </div>
    </body>
    </html>
    """.format(
        symbol=st.session_state.symbol,
        start_date=min_date,
        end_date=max_date,
        date=datetime.now(pytz.timezone('America/New_York')).strftime('%m-%d-%Y %I:%M %p EDT'),
        recommendation=st.session_state.score['Recommendation'],
        total_score=st.session_state.score['Total'],
        breakout_timeframe=st.session_state.breakout_timeframe,
        average_return=st.session_state.aapl_metrics['Average Return'],
        volatility=st.session_state.aapl_metrics['Volatility'],
        win_ratio=st.session_state.aapl_metrics['Win Ratio'],
        max_drawdown=st.session_state.aapl_metrics['Max Drawdown'],
        largest_loss=st.session_state.aapl_metrics['Largest Loss'],
        largest_loss_date=st.session_state.aapl_metrics['Largest Loss Date'],
        largest_gain=st.session_state.aapl_metrics['Largest Gain'],
        largest_gain_date=st.session_state.aapl_metrics['Largest Gain Date'],
        win_rate=st.session_state.backtest_results['Win Rate'],
        profit_factor=st.session_state.backtest_results['Profit Factor'],
        total_return=st.session_state.backtest_results['Total Return'],
        trades=st.session_state.backtest_results['Trades'],
        latest_date=st.session_state.aapl_df['date'].iloc[-1].strftime('%m-%d-%Y') if not st.session_state.aapl_df.empty else 'N/A',
        entry=st.session_state.aapl_df['close'].iloc[-1] if not st.session_state.aapl_df.empty else 0,
        stop_loss=stop_loss_value,
        take_profit=take_profit_value,
        alerts_table_html=alerts_table_html,
        candlestick_html=candlestick_html,
        bench_html=bench_html,
        heatmap_html=heatmap_html,
        pred_html=pred_html
    )
    html_buffer = io.StringIO()
    html_buffer.write(html_content)
    html_buffer.seek(0)
    st.download_button("Download HTML Report", html_buffer.getvalue(), file_name=f"{st.session_state.symbol}_analysis_report_{min_date}_to_{max_date}.html", mime="text/html")

# Export JSON report
if not st.session_state.aapl_df.empty:
    valid_dates = st.session_state.aapl_df['date'].dropna()
    if not valid_dates.empty:
        min_date = valid_dates.min().strftime('%m-%d-%Y')
        max_date = valid_dates.max().strftime('%m-%d-%Y')
    else:
        min_date = '01-01-2020'
        max_date = '06-24-2025'
    price_prediction_dict = st.session_state.price_prediction.to_dict(orient='records')
    for pred in price_prediction_dict:
        pred['date'] = pred['date'].isoformat()
    json_data = {
        "symbol": st.session_state.symbol,
        "date_range": {"from": min_date, "to": max_date},
        "metrics": st.session_state.aapl_metrics,
        "backtest_results": st.session_state.backtest_results,
        "signals": st.session_state.signals,
        "score": st.session_state.score,
        "price_prediction": price_prediction_dict,
        "alerts": st.session_state.alerts
    }
    json_buffer = io.StringIO()
    json.dump(json_data, json_buffer)
    json_buffer.seek(0)
    st.download_button("Download JSON Report", json_buffer.getvalue(), file_name=f"{st.session_state.symbol}_analysis_report_{min_date}_to_{max_date}.json", mime="application/json")

# Help section
with st.expander("📚 Help: How the Analysis Works"):
    help_text = """
    ### Step-by-Step Analysis Explanation
    This app analyzes {symbol} stock data to identify consolidation, breakouts, and trading setups. Below is the process with a real-time example based on current data.

    #### 1. Data Collection
    - **What**: Use OHLC, volume, and technical indicators (RSI, MACD, Stochastic, Ichimoku, ADX, ATR, Fibonacci, RVOL) from uploaded file or Yahoo Finance.
    - **How**: 
      - **Upload Mode**: Upload a CSV/XLSX file with columns: date, open, high, low, close, volume. Select a date range within the file’s data (at least 52 trading days).
      - **Real-Time Mode**: Fetch data via Yahoo Finance with a single valid symbol (e.g., AAPL) and user-specified date range (at least 52 trading days).
    - **Example**: Close: $196.45, RSI: 52.30, Stochastic %K: 7.12, %D: 8.00, ADX: 31.79, Volume: 51.4M, ATR: $4.30, RVOL: 1.2.

    #### 2. Candlestick Analysis & Breakout Detection
    - **What**: Visualize price movements to identify consolidation and breakouts.
    - **How**:
      - **Candlesticks**: Green (#4CAF50) for bullish, red (#f44336) for bearish.
      - **Consolidation**: Low ATR (< mean * 0.8), ADX < 20.
      - **Breakout**: Close > 20-day high, volume > average, RSI 40-70, MACD > signal, Stochastic %K > %D.
      - **Stop-Loss/Take-Profit**: Stop-loss = close - 1.5 * ATR; take-profit = close + 3 * ATR (1:2 risk-reward).
      - **Fibonacci**: Levels (23.6%, 38.2%, 50%, 61.8%) based on 20-day high/low.
      - **RVOL**: Volume / 20-day average volume.
      - **MACD Histogram**: Visualizes MACD - Signal difference to enhance breakout confirmation.
    - **Example**: Price ($196.45) below resistance (~$200). Buy if breaks $200 with volume > 50M, RSI 40-70, Stochastic %K > %D. Stop-loss: $193.55, take-profit: $212.90.

    #### 3. Profit/Loss Analysis
    - **What**: Calculate performance metrics.
    - **How**:
      - **Average Return**: Mean daily return (%).
      - **Volatility**: Annualized standard deviation (%).
      - **Win Ratio**: Percentage of positive return days.
      - **Max Drawdown**: Maximum peak-to-trough decline (%).
      - **Significant Events**: Largest loss/gain and dates.
    - **Example**: Average Return: -0.08%, Volatility: 4.57%, Win Ratio: 51.52%, Max Drawdown: 21.27%, Largest Loss: -9.25% on 04-10-2025.

    #### 4. Backtesting
    - **What**: Simulate trades based on buy signals, stop-loss, and take-profit.
    - **How**: Enter on buy signal, exit at stop-loss or take-profit, calculate win rate and profit factor.
    - **Example**: Win Rate: 60%, Profit Factor: 1.5, Total Return: 10%, Trades: 5.

    #### 5. Breakout Timeframe Prediction
    - **What**: Estimate breakout timing.
    - **How**: Consolidation → 1-5 days; breakout → confirm in 1-3 days.
    - **Example**: Consolidation on 06-24-2025, breakout expected by 06-29-2025.

    #### 6. Scoring System & Recommendation
    - **What**: Combine performance, risk, technical signals, and volume to determine a recommendation.
    - **How**: Total = Performance (30) + Risk (20) + Technical (30) + Volume (20). Recommendation: Buy if >70, Hold if >50, Market if ≤50.
    - **Example**: Total: 75/100, Recommendation: Buy.

    #### 7. Price Prediction
    - **What**: Predict next 5 trading days' closing prices.
    - **How**: Use linear regression on historical closes.
    - **Example**: 06-25-2025: $197.10, 06-26-2025: $197.85, etc.

    #### 8. Alert System
    - **What**: Notify significant price movements (>2% daily change).
    - **How**: Check daily percentage change within date range.
    - **Example**: Alert on 06-20-2025: 2.5% increase.

    #### 9. Visualization
    - **What**: Candlestick chart with Bollinger Bands, Ichimoku, RSI, MACD, Stochastic, ADX, RVOL, volume, and win/loss distribution.
    - **How**: Plotly charts with hover text and clickable trade details.
    - **Example**: Hover shows Date: 06-24-2025, Month: June, Close: $196.45, RSI: 52.30, Volume: 51.4M. Click candlestick for trade setup.

    #### 10. Benchmark Comparison
    - **What**: Compare {symbol} to benchmark (if uploaded) with 4 decimal place precision.
    - **Example**: {symbol}'s 20% outperforms benchmark's 10.1234%.

    #### 11. Seasonality Analysis
    - **What**: Identify monthly performance trends.
    - **How**: Heatmap of monthly returns.
    - **Example**: April 2025: -9.25% loss.

    #### 12. Trade Setups
    - **Consolidation Detection**: Identifies periods where the stock price is moving sideways with low volatility, indicating a potential buildup before a breakout. Calculated by checking if ATR is less than 80% of its 20-day mean and ADX < 20.
    """
    st.write(help_text.format(symbol=st.session_state.symbol))
