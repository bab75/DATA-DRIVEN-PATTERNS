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
import uuid
import openpyxl
import re
import calendar
import json
from sklearn.linear_model import LinearRegression
import pytz

# Check Plotly version
if plotly.__version__ < '5.0.0':
    st.warning(f"Plotly version {plotly.__version__} detected. Please upgrade to Plotly 5.x or higher with: `pip install plotly --upgrade`")

# Streamlit page config
st.set_page_config(page_title="Stock Investment Analysis", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for white background and readable text
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

# Initialize session state with defaults
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

# Sidebar for data source and settings
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Select Data Source", ["Upload CSV/XLSX", "Fetch Real-Time (Yahoo Finance)"], key="data_source")
symbol = st.sidebar.text_input("Stock Symbol (e.g., AAPL)", value=st.session_state.symbol, key="symbol_input")

# File uploaders
primary_file = st.sidebar.file_uploader("Upload Stock Data (CSV/XLSX)", type=["csv", "xlsx"], key="primary_file")
secondary_file = st.sidebar.file_uploader("Upload Benchmark Data (CSV/XLSX)", type=["csv", "xlsx"], key="secondary_file")

# Dynamic date inputs based on loaded data
# Initialize default dates
default_min_date = pd.to_datetime('01-01-2020', format='%m-%d-%Y')
default_max_date = pd.to_datetime(datetime.now().date(), format='%m-%d-%Y')  # Current system date

# Set min and max dates based on data source
if data_source == "Upload CSV/XLSX" and primary_file:
    # Load the file to check date range without processing the entire dataset
    try:
        if primary_file.name.endswith('.csv'):
            temp_df = pd.read_csv(primary_file)
        elif primary_file.name.endswith('.xlsx'):
            temp_df = pd.read_excel(primary_file)
        
        temp_df.columns = temp_df.columns.str.lower().str.strip()
        if 'date' not in temp_df.columns:
            st.error("Uploaded file must contain a 'date' column.")
            st.stop()
        
        temp_df['date'] = pd.to_datetime(temp_df['date'], errors='coerce', infer_datetime_format=True)
        valid_dates = temp_df['date'].dropna()
        
        if not valid_dates.empty:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
        else:
            st.error("No valid dates found in the uploaded file. Please ensure the 'date' column contains valid dates in formats like MM-DD-YYYY or YYYY-MM-DD.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading uploaded file: {str(e)}. Please check the file format.")
        st.stop()
else:
    # For Yahoo Finance, use default dates
    min_date = default_min_date
    max_date = default_max_date

# Ensure dates are timezone-naive for date_input
min_date = min_date if pd.api.types.is_datetime64_any_dtype(min_date) else pd.to_datetime(min_date)
max_date = max_date if pd.api.types.is_datetime64_any_dtype(max_date) else pd.to_datetime(max_date)
min_date = min_date.tz_localize(None) if min_date.tzinfo is not None else min_date
max_date = max_date.tz_localize(None) if max_date.tzinfo is not None else max_date

# Sidebar date inputs
st.sidebar.header("Date Range")
from_date = st.sidebar.date_input(
    "From Date",
    value=min_date.date() if data_source == "Upload CSV/XLSX" else default_min_date.date(),
    min_value=min_date.date(),
    max_value=max_date.date(),
    key="from_date_input",
    format="MM-DD-YYYY"
)
to_date = st.sidebar.date_input(
    "To Date",
    value=max_date.date(),
    min_value=min_date.date(),
    max_value=max_date.date() if data_source == "Upload CSV/XLSX" else datetime.now().date(),
    key="to_date_input",
    format="MM-DD-YYYY"
)

# Display date range for uploaded file
if data_source == "Upload CSV/XLSX" and primary_file:
    st.sidebar.write(f"File date range: {min_date.strftime('%m-%d-%Y')} to {max_date.strftime('%m-%d-%Y')}")

#########
st.sidebar.header("Chart Settings")
show_indicators = st.sidebar.multiselect(
    "Select Indicators",
    ["Bollinger Bands", "Ichimoku Cloud", "RSI", "MACD", "Stochastic", "ADX", "Fibonacci", "RVOL"],
    default=["Bollinger Bands", "RSI", "MACD","ADX"],
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

# Validate symbol format
def validate_symbol(symbol):
    return bool(re.match(r'^[A-Za-z0-9.]+$', symbol.strip()))

# Load and validate data
@st.cache_data
@st.cache_data
def load_data(primary_file, data_source, symbol, start_date, end_date, secondary_file=None, benchmark_symbol=None):
    aapl_df = pd.DataFrame()
    pl_df = pd.DataFrame()
    
    if data_source == "Upload CSV/XLSX" and primary_file:
        try:
            # Check file size
            primary_file.seek(0, 2)  # Move to end of file
            file_size = primary_file.tell()
            primary_file.seek(0)  # Reset to start
            if file_size == 0:
                st.error("Uploaded file is empty. Please upload a valid CSV/XLSX file with stock data.")
                return pd.DataFrame(), pd.DataFrame()
            
            # Read the file
            if primary_file.name.endswith('.csv'):
                # Try reading with different delimiters if comma fails
                try:
                    aapl_df = pd.read_csv(primary_file, sep=',')
                except pd.errors.EmptyDataError:
                    st.error("CSV file is empty or contains no valid data. Please check the file content.")
                    return pd.DataFrame(), pd.DataFrame()
                except pd.errors.ParserError:
                    # Try alternative delimiters (e.g., semicolon, tab)
                    primary_file.seek(0)
                    try:
                        aapl_df = pd.read_csv(primary_file, sep=';')
                    except:
                        primary_file.seek(0)
                        aapl_df = pd.read_csv(primary_file, sep='\t')
            elif primary_file.name.endswith('.xlsx'):
                aapl_df = pd.read_excel(primary_file)
            
            # Check if DataFrame is empty
            if aapl_df.empty:
                st.error("No data could be loaded from the file. Please ensure the file contains valid data with columns: date, open, high, low, close, volume.")
                return pd.DataFrame(), pd.DataFrame()
            
            # Standardize column names
            aapl_df.columns = aapl_df.columns.str.lower().str.strip()
            
            # Check for benchmark columns
            benchmark_cols = ['year', 'start date', 'end date', 'profit/loss (percentage)', 'profit/loss (value)']
            if any(col in aapl_df.columns for col in benchmark_cols):
                st.error(
                    "The uploaded file appears to be benchmark data. Please upload it as 'Benchmark Data' or upload a stock data file with columns: date, open, high, low, close, volume."
                )
                return pd.DataFrame(), pd.DataFrame()
            
            # Check for required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            actual_cols = aapl_df.columns.tolist()
            missing_cols = [col for col in required_cols if col not in actual_cols]
            if missing_cols:
                st.error(
                    f"Missing required columns in stock data: {', '.join(missing_cols)}. Required columns: date, open, high, low, close, volume."
                )
                st.write("Available columns:", actual_cols)
                st.write("Sample data (first 5 rows):", aapl_df.head() if not aapl_df.empty else "No data loaded")
                st.write("Data types:", aapl_df.dtypes if not aapl_df.empty else "No data loaded")
                return pd.DataFrame(), pd.DataFrame()
            
            # Flexible date parsing
            aapl_df['date'] = pd.to_datetime(aapl_df['date'], errors='coerce', infer_datetime_format=True)
            if aapl_df['date'].isna().all():
                st.error("All dates in the 'date' column are invalid. Supported formats include M-D-YYYY (e.g., 1-2-2025), MM-DD-YYYY (e.g., 01-02-2025), and YYYY-MM-DD (e.g., 2025-01-02).")
                st.write("Sample data (first 5 rows):", aapl_df.head() if not aapl_df.empty else "No data loaded")
                return pd.DataFrame(), pd.DataFrame()
            
            aapl_df = aapl_df.dropna(subset=['date'])  # Remove rows with NaT dates
            aapl_df['date'] = aapl_df['date'].dt.strftime('%m-%d-%Y')  # Standardize to MM-DD-YYYY
            aapl_df['date'] = pd.to_datetime(aapl_df['date'], format='%m-%d-%Y')  # Re-convert to datetime
            
            # Check numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                aapl_df[col] = pd.to_numeric(aapl_df[col], errors='coerce')
                if aapl_df[col].isna().any():
                    st.warning(f"Non-numeric values found in '{col}' column. Interpolating missing values.")
                    aapl_df[col] = aapl_df[col].interpolate(method='linear', limit_direction='both')
            
            # Check data sufficiency
            if aapl_df.empty or len(aapl_df) < 10:
                st.error(f"Insufficient data points ({len(aapl_df)}) after processing. Please upload a file with at least 10 trading days.")
                st.write("Sample data (first 5 rows):", aapl_df.head() if not aapl_df.empty else "No data loaded")
                return pd.DataFrame(), pd.DataFrame()
            
            min_date = aapl_df['date'].min()
            max_date = aapl_df['date'].max()
            st.sidebar.write(f"File date range: {min_date.strftime('%m-%d-%Y')} to {max_date.strftime('%m-%d-%Y')}")
            
            # Adjust selected date range to fit file’s range
            adjusted_start_date = max(start_date, min_date)
            adjusted_end_date = min(end_date, max_date)
            if adjusted_start_date > adjusted_end_date:
                st.error(
                    f"Adjusted date range ({adjusted_start_date.strftime('%m-%d-%Y')} to {adjusted_end_date.strftime('%m-%d-%Y')}) is invalid. "
                    f"File range is {min_date.strftime('%m-%d-%Y')} to {max_date.strftime('%m-%d-%Y')}. Adjust the date range."
                )
                return pd.DataFrame(), pd.DataFrame()
            
            aapl_df = aapl_df[(aapl_df['date'] >= adjusted_start_date) & (aapl_df['date'] <= adjusted_end_date)]
            if aapl_df.empty:
                st.error(f"No data available for the adjusted date range ({adjusted_start_date.strftime('%m-%d-%Y')} to {adjusted_end_date.strftime('%m-%d-%Y')}).")
                return pd.DataFrame(), pd.DataFrame()
            
            if len(aapl_df) < 10:
                st.error(f"Insufficient data points ({len(aapl_df)}) in adjusted date range. Select a range with at least 10 trading days.")
                return pd.DataFrame(), pd.DataFrame()
            
            if 'vwap' not in aapl_df.columns:
                st.warning("VWAP column is missing. VWAP plot will be skipped (optional).")
        
        except Exception as e:
            st.error(f"Error loading stock data: {str(e)}. Please check the file format, content, or try a different file.")
            st.write("Sample data (first 5 rows):", aapl_df.head() if not aapl_df.empty else "No data loaded")
            return pd.DataFrame(), pd.DataFrame()
    
    elif data_source == "Fetch Real-Time (Yahoo Finance)":
        try:
            symbol = symbol.strip()
            if not validate_symbol(symbol):
                st.error(f"Invalid symbol '{symbol}'. Use a valid stock symbol (e.g., AAPL, MSFT, BRK.B).")
                return pd.DataFrame(), pd.DataFrame()
            
            with st.spinner(f"Fetching data for {symbol} from Yahoo Finance..."):
                aapl_df = yf.download(symbol, start=start_date, end=end_date + timedelta(days=1), progress=False)
            
            if aapl_df.empty:
                st.error(
                    f"No data returned for {symbol} from Yahoo Finance. Possible causes:\n"
                    f"- Invalid symbol (check '{symbol}').\n"
                    f"- Date range ({start_date.strftime('%m-%d-%Y')} to {end_date.strftime('%m-%d-%Y')}) is invalid or too short.\n"
                    f"- Network issues or Yahoo Finance API downtime."
                )
                return pd.DataFrame(), pd.DataFrame()
            
            if isinstance(aapl_df, pd.DataFrame) and aapl_df.columns.nlevels > 1:
                try:
                    aapl_df = aapl_df.xs(symbol, level=1, axis=1, drop_level=True)
                except KeyError:
                    st.error(f"Unexpected multi-index data for {symbol}. Ensure a single valid symbol is entered.")
                    return pd.DataFrame(), pd.DataFrame()
            
            aapl_df = aapl_df.reset_index().rename(columns={
                'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            })
            aapl_df['date'] = pd.to_datetime(aapl_df['date'], errors='coerce', infer_datetime_format=True)
            aapl_df['date'] = aapl_df['date'].dt.strftime('%m-%d-%Y')
            aapl_df['date'] = pd.to_datetime(aapl_df['date'], format='%m-%d-%Y')
            
            aapl_df = aapl_df.dropna(subset=['date'])
            aapl_df = aapl_df.interpolate(method='linear', limit_direction='both')
            
            if aapl_df.empty or len(aapl_df) < 10:
                st.error(
                    f"Insufficient data points ({len(aapl_df)}) for {symbol}. Select a wider date range "
                    f"(e.g., 01-01-2020 to {datetime.now().strftime('%m-%d-%Y')})."
                )
                return pd.DataFrame(), pd.DataFrame()
        
        except Exception as e:
            st.error(
                f"Error fetching {symbol} data from Yahoo Finance: {str(e)}. Possible causes:\n"
                f"- Network connectivity issues.\n"
                f"- Invalid symbol or date range.\n"
                f"- Yahoo Finance API issues. Try uploading a file or a different symbol."
            )
            return pd.DataFrame(), pd.DataFrame()
    
    if secondary_file:
        try:
            if secondary_file.name.endswith('.csv'):
                pl_df = pd.read_csv(secondary_file)
            elif secondary_file.name.endswith('.xlsx'):
                pl_df = pd.read_excel(secondary_file)
            pl_df['Start Date'] = pd.to_datetime(pl_df['Start Date'], errors='coerce', infer_datetime_format=True)
            pl_df['Start Date'] = pl_df['Start Date'].dt.strftime('%m-%d-%Y')
            pl_df['Start Date'] = pd.to_datetime(pl_df['Start Date'], format='%m-%d-%Y')
            pl_df['End Date'] = pd.to_datetime(pl_df['End Date'], errors='coerce', infer_datetime_format=True)
            pl_df['End Date'] = pl_df['End Date'].dt.strftime('%m-%d-%Y')
            pl_df['End Date'] = pd.to_datetime(pl_df['End Date'], format='%m-%d-%Y')
            if pl_df[['Start Date', 'End Date', 'Profit/Loss (Percentage)']].isnull().any().any():
                st.warning("Benchmark data contains null values. Proceeding without benchmark.")
                pl_df = pd.DataFrame()
        except Exception as e:
            st.warning(f"Error loading benchmark data: {str(e)}. Proceeding without benchmark.")
    
    return aapl_df, pl_df

# Load data only if Submit is pressed and not already processed
if submit and not st.session_state.data_processed:
    st.session_state.data_loaded = True
    st.session_state.symbol = st.session_state.symbol_input
    st.session_state.start_date = pd.to_datetime(from_date, format='%m-%d-%Y')
    st.session_state.end_date = pd.to_datetime(to_date, format='%m-%d-%Y')
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

# Display searched symbols in UI
symbol_display = st.session_state.symbol
if hasattr(st.session_state, 'benchmark_symbol') and st.session_state.benchmark_symbol:
    symbol_display += f" vs {st.session_state.benchmark_symbol}"
st.markdown(f"### Analysis for {symbol_display}")

# Calculate daily return and metrics
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
    largest_loss_date = df.loc[df['daily_return'].idxmin(), 'date'].strftime('%m-%d-%Y') if not df['daily_return'].empty and not np.isnan(df['daily_return'].min()) else "N/A"
    largest_gain = df['daily_return'].max() * 100
    largest_gain_date = df.loc[df['daily_return'].idxmax(), 'date'].strftime('%m-%d-%Y') if not df['daily_return'].empty and not np.isnan(df['daily_return'].max()) else "N/A"
    
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
    df['stop_loss'] = df['close'] - 1.5 * df['atr']
    df['take_profit'] = df['close'] + 2 * 1.5 * df['atr']
    return df

if 'aapl_df_processed' not in st.session_state or submit:
    st.session_state.aapl_df = detect_consolidation_breakout(st.session_state.aapl_df)
    st.session_state.aapl_df_processed = st.session_state.aapl_df.copy()

# Backtesting framework
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

# Breakout timeframe prediction
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

# Price prediction with linear regression
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
        last_date = pd.Timestamp.now(tz='America/New_York')  # Fallback to current date/time if last date is NaT
    date_range = pd.date_range(start=last_date, periods=5, freq='B')
    return pd.DataFrame({
        'date': date_range,
        'predicted_close': predicted_prices
    })

if 'price_prediction' not in st.session_state or submit:
    st.session_state.price_prediction = predict_price(st.session_state.aapl_df)

# Alert system
@st.cache_data
def get_alerts(df):
    df['daily_change'] = df['close'].pct_change() * 100
    alerts = df[(df['daily_change'].abs() > 2)].copy()
    alerts['alert'] = alerts.apply(lambda row: f"{row['date'].strftime('%m-%d-%Y')}: {row['daily_change']:.2f}% change", axis=1)
    return alerts['alert'].tolist() if not alerts.empty else ["No significant price movements (>2%) detected."]

if 'alerts' not in st.session_state or submit:
    st.session_state.alerts = get_alerts(st.session_state.aapl_df)

# Plotly candlestick chart with customizable subplots
# Plotly candlestick chart with customizable subplots
if not subplot_order:
    st.warning("Please select at least one subplot in 'Customize Subplot Order' to display the chart.")
else:
    subplot_titles = [s for s in subplot_order]
    row_heights = [0.35 if s == "Candlestick" else 0.15 if s == "Win/Loss Distribution" else 0.1 for s in subplot_order]
    fig = make_subplots(rows=len(subplot_order), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=subplot_titles, row_heights=row_heights)

    def add_candlestick_trace(fig, df, row):
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce', format='%m-%d-%Y')
        df['date'] = df['date'].fillna(pd.NaT)

        hover_texts = [
            "Date: {date}<br>Month: {month}<br>Open: ${open:.2f}<br>High: ${high:.2f}<br>Low: ${low:.2f}<br>Close: ${close:.2f}<br>Volume: {volume:,.0f}<br>RSI: {rsi:.2f}<br>RVOL: {rvol:.2f}<br><b style='color:#006400'>Buy Signal: {buy_signal}</b>".format(
                date=getattr(r, 'date').strftime('%m-%d-%Y') if pd.notna(getattr(r, 'date')) else 'N/A',
                month=getattr(r, 'date').strftime('%B') if pd.notna(getattr(r, 'date')) else 'N/A',
                open=getattr(r, 'open'), high=getattr(r, 'high'), low=getattr(r, 'low'),
                close=getattr(r, 'close'), volume=getattr(r, 'volume'), rsi=getattr(r, 'rsi'), rvol=getattr(r, 'rvol'),
                buy_signal='Yes' if getattr(r, 'buy_signal') else 'No'
            )
            for r in df.itertuples()
        ]
        fig.add_trace(go.Candlestick(
            x=df['date'],
            open=df['open'], high=df['high'], low=df['low'], close=df['close'],
            name="Candlestick",
            increasing_line_color='#4CAF50', decreasing_line_color='#f44336',
            hovertext=hover_texts,
            hoverinfo='text',
            customdata=df.index
        ), row=row, col=1)
        if "Bollinger Bands" in show_indicators and 'ma20' in df.columns and 'std_dev' in df.columns:
            fig.add_trace(go.Scatter(x=df['date'], y=df['ma20'] + 2*df['std_dev'], name="Bollinger Upper", line=dict(color="#0288d1")), row=row, col=1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['ma20'] - 2*df['std_dev'], name="Bollinger Lower", line=dict(color="#0288d1"), fill='tonexty', fillcolor='rgba(2,136,209,0.1)'), row=row, col=1)
        if "Ichimoku Cloud" in show_indicators:
            fig.add_trace(go.Scatter(x=df['date'], y=df['senkou_span_a'], name="Senkou Span A", line=dict(color="#4CAF50"), fill='tonexty', fillcolor='rgba(76,175,80,0.2)'), row=row, col=1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['senkou_span_b'], name="Senkou Span B", line=dict(color="#f44336"), fill='tonexty', fillcolor='rgba(244,67,54,0.2)'), row=row, col=1)
        if "Fibonacci" in show_indicators:
            for level, color in [('fib_236', '#ff9800'), ('fib_382', '#e91e63'), ('fib_50', '#9c27b0'), ('fib_618', '#3f51b5')]:
                fig.add_trace(go.Scatter(x=df['date'], y=df[level], name=f"Fib {level[-3:]}%", line=dict(color=color, dash='dash'),
                                        hovertext=[f"Fib {level[-3:]}%: ${x:.2f}" for x in df[level]], hoverinfo='text+x'), row=row, col=1)
        buy_signals = df[df['buy_signal'] == True]
        for _, signal_row in buy_signals.iterrows():
            fig.add_annotation(x=signal_row['date'], y=signal_row['high'], text="Buy", showarrow=True, arrowhead=2, ax=0, ay=-30, font=dict(color="#000000"), row=row, col=1)
        if not buy_signals.empty:
            latest_buy = buy_signals.iloc[-1]
            risk = latest_buy['close'] - latest_buy['stop_loss']
            reward = latest_buy['take_profit'] - latest_buy['close']
            rr_ratio = reward / risk if risk > 0 else 'N/A'
            fig.add_hline(y=latest_buy['stop_loss'], line_dash="dash", line_color="#f44336", annotation_text="Stop-Loss", annotation_font_color="#000000", row=row, col=1)
            fig.add_hline(y=latest_buy['take_profit'], line_dash="dash", line_color="#4CAF50", annotation_text="Take-Profit", annotation_font_color="#000000", row=row, col=1)
            fig.add_trace(go.Scatter(x=[latest_buy['date'], latest_buy['date']], y=[latest_buy['stop_loss'], latest_buy['take_profit']],
                                    mode='lines', line=dict(color='rgba(76,175,80,0.2)'), fill='toself', fillcolor='rgba(76,175,80,0.2)',
                                    hovertext=[f"Risk-Reward Ratio: {rr_ratio:.2f}" if isinstance(rr_ratio, float) else f"Risk-Reward Ratio: {rr_ratio}"], hoverinfo='text', showlegend=False), row=row, col=1)

    def add_rsi_trace(fig, df, row):
        fig.add_trace(go.Scatter(x=df['date'], y=df['rsi'], name="RSI", line=dict(color="#9c27b0"),
                                hovertext=[f"RSI: {x:.2f}" for x in df['rsi']], hoverinfo='text+x'), row=row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#f44336", row=row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#4CAF50", row=row, col=1)

    def add_macd_stochastic_trace(fig, df, row):
        if "MACD" in show_indicators and 'macd' in df.columns and 'signal' in df.columns:
            fig.add_trace(go.Scatter(x=df['date'], y=df['macd'], name="MACD", line=dict(color="#0288d1"),
                                    hovertext=[f"Date: {date.strftime('%m-%d-%Y')}<br>MACD: {macd:.2f}" for date, macd in zip(df['date'], df['macd'])], hoverinfo='text'), row=row, col=1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['signal'], name="Signal Line", line=dict(color="#ff9800"),
                                    hovertext=[f"Date: {date.strftime('%m-%d-%Y')}<br>Signal: {signal:.2f}" for date, signal in zip(df['date'], df['signal'])], hoverinfo='text'), row=row, col=1)
            fig.add_trace(go.Bar(x=df['date'], y=df['macd_diff'], name="MACD Histogram", marker_color="#607d8b",
                                hovertext=[f"Date: {date.strftime('%m-%d-%Y')}<br>MACD Diff: {diff:.2f}" for date, diff in zip(df['date'], df['macd_diff'])], hoverinfo='text'), row=row, col=1)
        if "Stochastic" in show_indicators:
            fig.add_trace(go.Scatter(x=df['date'], y=df['stochastic_k'], name="Stochastic %K", line=dict(color="#e91e63"), yaxis="y2",
                                    hovertext=[f"Date: {date.strftime('%m-%d-%Y')}<br>Stochastic %K: {k:.2f}" for date, k in zip(df['date'], df['stochastic_k'])], hoverinfo='text'), row=row, col=1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['stochastic_d'], name="Stochastic %D", line=dict(color="#ff5722"), yaxis="y2",
                                    hovertext=[f"Date: {date.strftime('%m-%d-%Y')}<br>Stochastic %D: {d:.2f}" for date, d in zip(df['date'], df['stochastic_d'])], hoverinfo='text'), row=row, col=1)
            fig.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 100]))

    def add_adx_volatility_trace(fig, df, row):
        if "ADX" in show_indicators:
            fig.add_trace(go.Scatter(x=df['date'], y=df['adx'], name="ADX", line=dict(color="#3f51b5"),
                                    hovertext=[f"Date: {date.strftime('%m-%d-%Y')}<br>ADX: {adx:.2f}" for date, adx in zip(df['date'], df['adx'])], hoverinfo='text'), row=row, col=1)
            fig.add_hline(y=25, line_dash="dash", line_color="#0288d1", row=row, col=1)
        if "RVOL" in show_indicators:
            fig.add_trace(go.Scatter(x=df['date'], y=df['rvol'], name="RVOL", line=dict(color="#795548"), yaxis="y3",
                                    hovertext=[f"Date: {date.strftime('%m-%d-%Y')}<br>RVOL: {rvol:.2f}" for date, rvol in zip(df['date'], df['rvol'])], hoverinfo='text'), row=row, col=1)
            fig.update_layout(yaxis3=dict(overlaying='y', side='right'))

    def add_volume_trace(fig, df, row):
        fig.add_trace(go.Bar(x=df['date'], y=df['volume'], name="Volume", marker_color="#607d8b",
                            hovertext=[f"Volume: {x:,.0f}" for x in df['volume']], hoverinfo='text+x'), row=row, col=1)
        if 'vwap' in df.columns:
            fig.add_trace(go.Scatter(x=df['date'], y=df['vwap'], name="VWAP", line=dict(color="#0288d1"),
                                    hovertext=[f"VWAP: ${x:.2f}" for x in df['vwap']], hoverinfo='text+x'), row=row, col=1)

    def add_win_loss_trace(fig, df, row):
        if 'daily_return' not in df.columns:
            st.warning("Cannot plot Win/Loss Distribution: 'daily_return' column is missing.")
            return
        valid_returns = df['daily_return'][df['daily_return'].notna() & ~df['daily_return'].isin([np.inf, -np.inf])]
        if not valid_returns.empty:
            bins = np.histogram_bin_edges(valid_returns * 100, bins=20)
            hist_data = np.histogram(valid_returns * 100, bins=bins)
            fig.add_trace(go.Bar(
                x=bins[:-1],
                y=hist_data[0],
                name="Win/Loss Distribution",
                marker_color="#607d8b",
                hovertext=[f"Return: {x:.2f}% Count: {y}" for x, y in zip(bins[:-1], hist_data[0])],
                hoverinfo='text'
            ), row=row, col=1)
        else:
            st.warning("Cannot plot Win/Loss Distribution: No valid daily returns available.")

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

    fig.update_layout(height=200 * len(subplot_order), showlegend=True, template="plotly_white",
                      title_text=f"{st.session_state.symbol} Candlestick Analysis (Date Range: {st.session_state.start_date.strftime('%m-%d-%Y')} to {st.session_state.end_date.strftime('%m-%d-%Y')})",
                      hovermode="x unified", font=dict(family="Arial", size=12, color="#000000"))
    fig.update_xaxes(rangeslider_visible=True, tickformat="%m-%d-%Y", row=len(subplot_order), col=1)

    def on_click(trace, points, state):
        if points.point_inds:
            idx = points[0].point_inds[0]
            row = st.session_state.aapl_df.iloc[idx]
            st.session_state.trade_details = {
                'Date': row['date'].strftime('%m-%d-%Y'),
                'Close': float(row['close']),
                'Stop-Loss': float(row['stop_loss']),
                'Take-Profit': float(row['take_profit']),
                'Buy Signal': 'Yes' if row['buy_signal'] else 'No'
            }

    for trace in fig.data:
        if trace.name == "Candlestick":
            trace.on_click(on_click)

    st.plotly_chart(fig, use_container_width=True)

# Profit/Loss Analysis Section
st.header("Profit/Loss Analysis")
st.write(f"**Date Range:** {st.session_state.start_date.strftime('%m-%d-%Y')} to {st.session_state.end_date.strftime('%m-%d-%Y')}")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-box'><b>Average Return</b><br>{st.session_state.aapl_metrics['Average Return']:.2f}%</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-box'><b>Volatility</b><br>{st.session_state.aapl_metrics['Volatility']:.2f}%</div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-box'><b>Win Ratio</b><br>{st.session_state.aapl_metrics['Win Ratio']:.2f}%</div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-box'><b>Max Drawdown</b><br>{st.session_state.aapl_metrics['Max Drawdown']:.2f}%</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='metric-box'><b>Significant Events</b><br>"
    "Largest single-period loss was {largest_loss:.2f}% on {largest_loss_date}, indicating a significant market correction.<br>"
    "Largest single-period gain was {largest_gain:.2f}% on {largest_gain_date}.</div>".format(
        largest_loss=st.session_state.aapl_metrics['Largest Loss'],
        largest_loss_date=st.session_state.aapl_metrics['Largest Loss Date'],
        largest_gain=st.session_state.aapl_metrics['Largest Gain'],
        largest_gain_date=st.session_state.aapl_metrics['Largest Gain Date']
    ),
    unsafe_allow_html=True
)

# Price Movement Alerts Section
st.header("Price Movement Alerts")
with st.expander("View Alerts"):
    if 'alerts' in st.session_state and st.session_state.alerts:
        # Parse alerts into a DataFrame with date and percentage change
        alerts_df = pd.DataFrame({
            'Date': [alert.split(': ')[0] for alert in st.session_state.alerts],
            'Change (%)': [float(alert.split(': ')[1].replace('% change', '')) for alert in st.session_state.alerts]
        })
        # Add filters for min and max percentage change
        min_change = st.slider("Minimum % Change", -100.0, 100.0, -100.0, 0.1)
        max_change = st.slider("Maximum % Change", -100.0, 100.0, 100.0, 0.1)
        filtered_alerts = alerts_df[(alerts_df['Change (%)'] >= min_change) & (alerts_df['Change (%)'] <= max_change)]
        
        if not filtered_alerts.empty:
            # Split into two columns for two-row display
            col1, col2 = st.columns(2)
            with col1:
                st.table(filtered_alerts.iloc[:len(filtered_alerts)//2])
            with col2:
                st.table(filtered_alerts.iloc[len(filtered_alerts)//2:])
        else:
            st.write("No alerts match the selected percentage change range.")
    else:
        st.write("No significant price movements (>2%) detected.")

# Backtesting Results
st.header("Backtesting Results")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-box'><b>Win Rate</b><br>{st.session_state.backtest_results['Win Rate']:.2f}%</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-box'><b>Profit Factor</b><br>{st.session_state.backtest_results['Profit Factor']:.2f}</div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-box'><b>Total Return</b><br>{st.session_state.backtest_results['Total Return']:.2f}%</div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-box'><b>Trades</b><br>{st.session_state.backtest_results['Trades']}</div>", unsafe_allow_html=True)

# Price Prediction Section
st.header("Price Prediction (Next 5 Trading Days)")
prediction_df = st.session_state.price_prediction
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=prediction_df['date'], y=prediction_df['predicted_close'], mode='lines+markers', name="USD", line=dict(color="#0288d1"),
                              hovertext=[f"Date: {d.strftime('%m-%d-%Y')}<br>Predicted Close: ${p:.2f}" for d, p in zip(prediction_df['date'], prediction_df['predicted_close'])], hoverinfo='text+x'))
fig_pred.update_layout(title=f"{st.session_state.symbol} Price Prediction", height=400, template="plotly_white",
                       hovermode="x unified", font=dict(family="Arial", size=12, color="#000000"), xaxis_tickformat="%m-%d-%Y")
st.plotly_chart(fig_pred, use_container_width=True)

# Decision Dashboard
st.header("Decision Dashboard")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-box'><b>Recommendation</b><br>{st.session_state.score['Recommendation']}</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-box'><b>Total Score</b><br>{st.session_state.score['Total']:.1f}/100</div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-box'><b>Breakout Timeframe</b><br>{st.session_state.breakout_timeframe}</div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-box'><b>CAGR</b><br>{st.session_state.aapl_metrics['CAGR']:.2f}%</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-box'><b>Sharpe Ratio</b><br>{st.session_state.aapl_metrics['Sharpe Ratio']:.2f}</div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-box'><b>RSI</b><br>{st.session_state.aapl_df['rsi'].iloc[-1]:.2f} ({st.session_state.signals['RSI']})</div>", unsafe_allow_html=True)

# Display trade details
if st.session_state.trade_details and all(key in st.session_state.trade_details for key in ['Date', 'Close', 'Stop-Loss', 'Take-Profit', 'Buy Signal']):
    st.header("Selected Trade Details")
    details = st.session_state.trade_details
    try:
        rr_ratio = (details['Take-Profit'] - details['Close']) / (details['Close'] - details['Stop-Loss']) if (details['Close'] - details['Stop-Loss']) > 0 else 'N/A'
        st.markdown(
            "<div class='trade-details'>"
            "<b>Date:</b> {date}<br>"
            "<b>Close:</b> ${close:.2f}<br>"
            "<b>Stop-Loss:</b> ${stop_loss:.2f}<br>"
            "<b>Take-Profit:</b> ${take_profit:.2f}<br>"
            "<b>Buy Signal:</b> {buy_signal}<br>"
            "<b>Risk-Reward Ratio:</b> {rr_ratio}"
            "</div>".format(
                date=details['Date'],
                close=details['Close'],
                stop_loss=details['Stop-Loss'],
                take_profit=details['Take-Profit'],
                buy_signal=details['Buy Signal'],
                rr_ratio=f"{rr_ratio:.2f}" if isinstance(rr_ratio, float) else rr_ratio
            ),
            unsafe_allow_html=True
        )
    except Exception as e:
        st.warning(f"Error displaying trade details: {str(e)}. Please select a candlestick to view trade details.")
else:
    st.info("Click a candlestick on the chart to view trade details.")

# Latest Trade Setup
latest_buy = st.session_state.aapl_df[st.session_state.aapl_df['buy_signal'] == True].iloc[-1] if not st.session_state.aapl_df[st.session_state.aapl_df['buy_signal'] == True].empty else None
if latest_buy is not None:
    st.header("Latest Trade Setup")
    st.markdown(
        "<div class='trade-details'>"
        "<b>Date:</b> {date}<br>"
        "<b>Entry:</b> ${entry:.2f}<br>"
        "<b>Stop-Loss:</b> ${stop_loss:.2f}<br>"
        "<b>Take-Profit:</b> ${take_profit:.2f}"
        "</div>".format(
            date=latest_buy['date'].strftime('%m-%d-%Y'),
            entry=latest_buy['close'],
            stop_loss=latest_buy['stop_loss'],
            take_profit=latest_buy['take_profit']
        ),
        unsafe_allow_html=True
    )

# Display candlestick chart
st.plotly_chart(fig, use_container_width=True)

# Benchmark comparison
fig_bench = None
if st.session_state.get('secondary_file') and not st.session_state.pl_df.empty:
    st.header("Benchmark Comparison")
    try:
        pl_cum_return = (1 + st.session_state.pl_df['Profit/Loss (Percentage)'] / 100).cumprod() - 1
        fig_bench = go.Figure()
        fig_bench.add_trace(go.Scatter(
            x=st.session_state.aapl_df['date'],
            y=st.session_state.aapl_df['cumulative_return'],
            name=st.session_state['symbol'],
            line=dict(color="#0288d1"),
            hovertext=[f"{st.session_state.symbol} Return: {x:.2%}" for x in st.session_state.aapl_df['cumulative_return']],
            hoverinfo='text+x'
        ))
        fig_bench.add_trace(go.Scatter(
            x=st.session_state.pl_df['End Date'],
            y=pl_cum_return,  # Corrected from df_cum_return to pl_cum_return
            name="Benchmark",
            line=dict(color="#ff9800"),
            hovertext=[f"Benchmark Return: {x:.4f}" for x in pl_cum_return],
            hoverinfo='text+x'
        ))
        fig_bench.update_layout(
            title=f"{st.session_state.symbol} vs. Benchmark Cumulative Returns (Date Range: {st.session_state.start_date.strftime('%m-%d-%Y')} to {st.session_state.end_date.strftime('%m-%d-%Y')})",
            height=400,
            template="plotly_white",
            hovermode="x unified",
            font=dict(family="Arial", size=12, color="#000000"),
            xaxis_tickformat="%m-%d-%Y"
        )
        st.plotly_chart(fig_bench, use_container_width=True)
    except Exception as e:
        st.warning(f"Error plotting benchmark comparison: {str(e)}. Skipping benchmark chart.")

# Seasonality heatmap
st.header("Seasonality Analysis")
if not pd.api.types.is_datetime64_any_dtype(st.session_state.aapl_df['date']):
    st.session_state.aapl_df['date'] = pd.to_datetime(st.session_state.aapl_df['date'], errors='coerce', format='%m-%d-%Y')
st.session_state.aapl_df['month'] = st.session_state.aapl_df['date'].dt.month
st.session_state.aapl_df['year'] = st.session_state.aapl_df['date'].dt.year
monthly_returns = st.session_state.aapl_df.groupby(['year', 'month'])['daily_return'].mean().unstack() * 100
month_names = {i: calendar.month_name[i] for i in range(1, 13)}
fig_heatmap = go.Figure(data=go.Heatmap(
    z=monthly_returns.values,
    x=[month_names[col] for col in monthly_returns.columns],
    y=monthly_returns.index,
    colorscale='RdYlGn',
    hovertext=[[f"Year: {year}<br>Month: {month_names[month]}<br>Return: {x:.2f}%" for month, x in enumerate(row, 1)] for year, row in monthly_returns.iterrows()],
    hoverinfo='text'
))
fig_heatmap.update_layout(
    title=f"Monthly Average Returns Heatmap (Date Range: {st.session_state.start_date.strftime('%m-%d-%Y')} to {st.session_state.end_date.strftime('%m-%d-%Y')})",
    height=400, template="plotly_white",
    font=dict(family="Arial", size=12, color="#000000"),
    xaxis_title="Month",
    yaxis_title="Year",
    xaxis=dict(tickmode='array', tickvals=list(range(12)), ticktext=list(month_names.values()))
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# Export data as CSV and Excel
st.header("Export Data and Reports")
if not st.session_state.aapl_df.empty:
    # Filter out NaT values and get valid min/max dates
    valid_dates = st.session_state.aapl_df['date'].dropna()
    if not valid_dates.empty:
        min_date = valid_dates.min().strftime('%m-%d-%Y')
        max_date = valid_dates.max().strftime('%m-%d-%Y')
    else:
        min_date = '01-01-2020'  # Fallback if no valid dates
        max_date = '06-24-2025'  # Current date as fallback
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

# Export HTML report
if not st.session_state.aapl_df.empty:
    valid_dates = st.session_state.aapl_df['date'].dropna()
    if not valid_dates.empty:
        min_date = valid_dates.min().strftime('%m-%d-%Y')
        max_date = valid_dates.max().strftime('%m-%d-%Y')
    else:
        min_date = '01-01-2020'
        max_date = '06-25-2025'
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

    # Generate alerts table for HTML report
    alerts_html = "<div class='alert-box'>No significant price movements (>2%) detected.</div>"
    if st.session_state.alerts and st.session_state.alerts[0] != "No significant price movements (>2%) detected.":
        alerts_data = [
            {"Date": alert.split(': ')[0], "Change (%)": float(alert.split(': ')[1].replace('% change', ''))}
            for alert in st.session_state.alerts
        ]
        alerts_table = "<table style='width:100%; border-collapse: collapse; margin: 10px 0;'><tr><th style='border: 1px solid #ddd; padding: 8px; background-color: #f0f0f0;'>Date</th><th style='border: 1px solid #ddd; padding: 8px; background-color: #f0f0f0;'>Change (%)</th></tr>"
        for alert in alerts_data:
            alerts_table += f"<tr><td style='border: 1px solid #ddd; padding: 8px;'>{alert['Date']}</td><td style='border: 1px solid #ddd; padding: 8px;'>{alert['Change (%)']:.2f}</td></tr>"
        alerts_table += "</table>"
        alerts_html = f"<div class='alert-box'>{alerts_table}</div>"

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
            table {{ width: 100%; border-collapse: collapse; }}
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
            {alerts_html}
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
                <p><b>Stop-Loss:</b> ${stop:.2f}</p>
                <p><b>Take-Profit:</b> ${take:.2f}</p>
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
        stop=stop_loss_value,
        take=take_profit_value,
        alerts_html=alerts_html,
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
    # Convert price_prediction DataFrame to a JSON-serializable format
    price_prediction_dict = st.session_state.price_prediction.to_dict(orient='records')
    for pred in price_prediction_dict:
        pred['date'] = pred['date'].isoformat()  # Convert Timestamp to ISO string
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
    
    #### 13. xls. Export Functionality
    - **change the date fromat to MM-DD-YYY in both stock data and bench mark files and select the date in the date field with in min and max history data available in market data xls.
    - ** Upload your stock data CSV generated by program 1_1_5_AI_READ_PRICE.py(From Technical Analysis tab by applying all indicators from full data frame and for bench mark us DATA-DRIVEN program generated profit and loss file.
    """
    st.write(help_text.format(symbol=st.session_state.symbol))
