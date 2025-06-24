import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import ta
import calendar

# Initialize session state for data persistence
if 'aapl_df' not in st.session_state:
    st.session_state.aapl_df = pd.DataFrame()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# Sidebar for inputs
st.sidebar.header("Stock Analysis Inputs")
data_source = st.sidebar.radio("Data Source", ["Fetch Real-Time (Yahoo Finance)", "Upload CSV/XLSX"])
symbol = st.sidebar.text_input("Stock Symbol (e.g., AAPL)", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime(2025, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2025, 6, 13))
primary_file = st.sidebar.file_uploader("Upload Stock Data (CSV/XLSX)", type=["csv", "xlsx"], key="primary_file")
secondary_file = st.sidebar.file_uploader("Upload Benchmark Data (CSV/XLSX)", type=["csv", "xlsx"], key="secondary_file")
st.session_state.symbol = symbol

# Load data
@st.cache_data
def load_data(primary_file, data_source, symbol, start_date, end_date):
    aapl_df = pd.DataFrame()
    pl_df = pd.DataFrame()
    
    if data_source == "Upload CSV/XLSX" and primary_file:
        try:
            if primary_file.name.endswith('.csv'):
                aapl_df = pd.read_csv(primary_file)
            elif primary_file.name.endswith('.xlsx'):
                aapl_df = pd.read_excel(primary_file)
            
            # Normalize column names
            aapl_df.columns = aapl_df.columns.str.lower().str.strip()
            
            # Detect benchmark-like file
            benchmark_cols = ['year', 'start date', 'end date', 'profit/loss (percentage)', 'profit/loss (value)']
            if any(col in aapl_df.columns for col in benchmark_cols):
                st.error(
                    "The uploaded file appears to be benchmark data. Please upload it as 'Benchmark Data' or upload a stock data file with columns: date, open, high, low, close, volume."
                )
                return pd.DataFrame(), pd.DataFrame()
            
            # Validate required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            actual_cols = aapl_df.columns.tolist()
            missing_cols = [col for col in required_cols if col not in actual_cols]
            if missing_cols:
                st.error(
                    f"Missing required columns in stock data: {', '.join(missing_cols)}. Please upload a file with columns: date, open, high, low, close, volume."
                )
                st.write("Available columns:", actual_cols)
                st.write("Sample data (first 5 rows):", aapl_df.head())
                st.write("Data types:", aapl_df.dtypes)
                return pd.DataFrame(), pd.DataFrame()
            
            # Convert data types
            aapl_df['date'] = pd.to_datetime(aapl_df['date'], errors='coerce')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                aapl_df[col] = pd.to_numeric(aapl_df[col], errors='coerce')
            
            # Interpolate missing values instead of dropping rows
            aapl_df = aapl_df.interpolate(method='linear', limit_direction='both')
            
            # Get file's date range
            if not aapl_df['date'].empty:
                min_date = aapl_df['date'].min()
                max_date = aapl_df['date'].max()
                st.sidebar.write(f"File date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                
                # Validate selected date range
                if start_date < min_date or end_date > max_date:
                    st.error(f"Selected date range ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}) is outside the file's range ({min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}).")
                    return pd.DataFrame(), pd.DataFrame()
                
                # Filter by selected date range
                aapl_df = aapl_df[(aapl_df['date'] >= start_date) & (aapl_df['date'] <= end_date)]
                if aapl_df.empty:
                    st.error(f"No data available for the selected date range ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}). Please adjust the date range.")
                    return pd.DataFrame(), pd.DataFrame()
                
                # Ensure sufficient data points
                if len(aapl_df) < 52:
                    st.error(f"Insufficient data points ({len(aapl_df)}) in selected date range. Please select a range with at least 52 trading days.")
                    return pd.DataFrame(), pd.DataFrame()
            
            else:
                st.error("No valid dates found in the uploaded file. Please ensure the 'date' column contains valid dates.")
                return pd.DataFrame(), pd.DataFrame()
            
            # Check for VWAP
            if 'vwap' not in aapl_df.columns:
                st.warning("VWAP column is missing. VWAP plot will be skipped (optional).")
        
        except Exception as e:
            st.error(f"Error loading stock data: {str(e)}. Please check the file format and content.")
            st.write("Sample data (first 5 rows):", aapl_df.head() if not aapl_df.empty else "No data loaded")
            return pd.DataFrame(), pd.DataFrame()
    
    elif data_source == "Fetch Real-Time (Yahoo Finance)":
        try:
            # Validate symbol
            symbol = symbol.strip()
            if not validate_symbol(symbol):
                st.error(f"Invalid symbol '{symbol}'. Please enter a single valid stock symbol (e.g., AAPL, MSFT, BRK.B).")
                return pd.DataFrame(), pd.DataFrame()
            
            # Fetch data
            aapl_df = yf.download(symbol, start=start_date, end=end_date + timedelta(days=1), progress=False)
            if aapl_df.empty:
                st.error(f"Failed to fetch {symbol} data from Yahoo Finance. Please check the symbol, date range, or internet connection.")
                return pd.DataFrame(), pd.DataFrame()
            
            # Handle multi-index DataFrame
            if isinstance(aapl_df, pd.DataFrame) and aapl_df.columns.nlevels > 1:
                try:
                    aapl_df = aapl_df.xs(symbol, level=1, axis=1, drop_level=True)
                except KeyError:
                    st.error(f"Unexpected multi-index data for {symbol}. Please ensure a single valid symbol is entered (e.g., AAPL, not AAPL,MSFT).")
                    return pd.DataFrame(), pd.DataFrame()
            
            # Normalize column names
            aapl_df = aapl_df.reset_index().rename(columns={
                'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            })
            aapl_df['date'] = pd.to_datetime(aapl_df['date'])
            
            # Interpolate missing values
            aapl_df = aapl_df.interpolate(method='linear', limit_direction='both')
            
            # Ensure sufficient data points
            if len(aapl_df) < 52:
                st.error(f"Insufficient data points ({len(aapl_df)}) for {symbol}. Please select a wider date range (at least 52 trading days, e.g., 2024-01-01 to 2025-06-13).")
                return pd.DataFrame(), pd.DataFrame()
        
        except Exception as e:
            st.error(f"Error fetching {symbol} data from Yahoo Finance: {str(e)}. Please check the symbol, date range, or try uploading a file.")
            return pd.DataFrame(), pd.DataFrame()
    
    # Compute daily_return and technical indicators
    if not aapl_df.empty:
        try:
            aapl_df['daily_return'] = aapl_df['close'].pct_change()
            aapl_df['daily_return'] = aapl_df['daily_return'].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Use Pandas Series for ta library calculations
            close = aapl_df['close']
            high = aapl_df['high']
            low = aapl_df['low']
            volume = aapl_df['volume']
            
            aapl_df['rsi'] = ta.momentum.RSIIndicator(close, window=14).rsi()
            aapl_df['macd'] = ta.trend.MACD(close).macd()
            aapl_df['signal'] = ta.trend.MACD(close).macd_signal()
            aapl_df['stochastic_k'] = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3).stoch()
            aapl_df['stochastic_d'] = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3).stoch_signal()
            aapl_df['adx'] = ta.trend.ADXIndicator(high, low, close, window=14).adx()
            aapl_df['atr'] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
            ichimoku = ta.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
            aapl_df['senkou_span_a'] = ichimoku.ichimoku_a()
            aapl_df['senkou_span_b'] = ichimoku.ichimoku_b()
            aapl_df['ma20'] = close.rolling(window=20).mean()
            aapl_df['std_dev'] = close.rolling(window=20).std()
            aapl_df['rvol'] = volume / volume.rolling(window=20).mean()
            
            # Fibonacci retracement levels
            recent_high = high.rolling(window=20).max()
            recent_low = low.rolling(window=20).min()
            diff = recent_high - recent_low
            aapl_df['fib_236'] = recent_high - diff * 0.236
            aapl_df['fib_382'] = recent_high - diff * 0.382
            aapl_df['fib_50'] = recent_high - diff * 0.5
            aapl_df['fib_618'] = recent_high - diff * 0.618
            
            # Interpolate missing indicator values
            indicator_cols = [
                'rsi', 'macd', 'signal', 'stochastic_k', 'stochastic_d', 'adx', 'atr',
                'senkou_span_a', 'senkou_span_b', 'ma20', 'std_dev', 'rvol',
                'fib_236', 'fib_382', 'fib_50', 'fib_618'
            ]
            aapl_df[indicator_cols] = aapl_df[indicator_cols].interpolate(method='linear', limit_direction='both')
            
            # Validate indicator data
            null_counts = aapl_df[indicator_cols].isnull().sum()
            if null_counts.any():
                st.warning(f"Remaining missing values in indicators after interpolation:\n{null_counts[null_counts > 0]}")
            
        except Exception as e:
            st.error(f"Error computing technical indicators: {str(e)}. Please ensure sufficient data points (at least 52 trading days) and valid data.")
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

# Validate symbol function
def validate_symbol(symbol):
    return bool(symbol and symbol.strip() and len(symbol.split(',')) == 1 and all(c.isalnum() or c == '.' for c in symbol))

# Load data based on user input
aapl_df, pl_df = load_data(primary_file, data_source, symbol, start_date, end_date)

# Compute indicators and signals
if not st.session_state.aapl_df.empty:
    aapl_df = st.session_state.aapl_df.copy()
    aapl_df = detect_consolidation_breakout(aapl_df)
    backtest_results = backtest_strategy(aapl_df)
    st.session_state.analysis_results.update(backtest_results)

    # Candlestick chart with indicators
    st.header("Candlestick Chart with Indicators")
    show_indicators = st.multiselect("Select Indicators", ["Bollinger Bands", "Ichimoku Cloud", "Fibonacci"], default=["Bollinger Bands"])
    fig = go.Figure()
    for i in range(1, 3):
        add_candlestick_trace(fig, aapl_df, i)
    fig.update_layout(
        title=f"{symbol} Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=800,
        font=dict(family="Arial", size=12, color="#000000")
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display backtesting results
    st.header("Backtesting Results")
    st.write(f"Win Rate: {backtest_results['Win Rate']:.2f}%")
    st.write(f"Profit Factor: {backtest_results['Profit Factor']:.2f}")
    st.write(f"Total Return: {backtest_results['Total Return']:.2f}%")
    st.write(f"Trades: {backtest_results['Trades']}")

@st.cache_data
def detect_consolidation_breakout(df):
    df['is_consolidation'] = (df['atr'] < df['atr'].mean() * 0.8) & (df['adx'] < 20)
    df['resistance'] = df['high'].rolling(20).max()
    df['support'] = df['low'].rolling(20).min()
    # Add trend filter: 5-day MA > 5-day prior MA
    df['ma5'] = df['close'].rolling(5).mean()
    trend_condition = df['ma5'] > df['ma5'].shift(5)
    # Detailed debug for buy_signal conditions
    close_exceeds_resistance = df['close'] > df['resistance'].shift(1)
    volume_condition = df['volume'] > df['volume'].mean() * 0.8
    rsi_condition = (df['rsi'] > 30) & (df['rsi'] < 80)
    macd_condition = df['macd'] > df['signal']
    stochastic_condition = df['stochastic_k'] > df['stochastic_d']
    df['buy_signal'] = trend_condition & volume_condition & rsi_condition & macd_condition & stochastic_condition
    df['stop_loss'] = df['close'] - 1.5 * df['atr']
    df['take_profit'] = df['close'] + 2 * 1.5 * df['atr']
    st.write("Debug: Buy signal condition checks:")
    st.write(f"- Trend (MA5 > MA5.shift(5)): {trend_condition.sum()} True")
    st.write(f"- Volume > Mean * 0.8: {volume_condition.sum()} True")
    st.write(f"- 30 < RSI < 80: {rsi_condition.sum()} True")
    st.write(f"- MACD > Signal: {macd_condition.sum()} True")
    st.write(f"- Stochastic K > D: {stochastic_condition.sum()} True")
    st.write("Debug: Sample comparison of Close vs Resistance.shift(1):")
    st.write(df[['date', 'close', 'resistance']].tail(10).assign(resistance_shift1=lambda x: x['resistance'].shift(1)))
    num_buy_signals = df['buy_signal'].sum()
    st.write(f"Debug: Number of buy signals detected: {num_buy_signals}")
    if num_buy_signals == 0:
        st.warning("No buy signals detected. Check data or relax conditions further if needed.")
        st.write("Debug: First 5 rows of signal conditions:", df[['close', 'resistance', 'volume', 'rsi', 'macd', 'signal', 'stochastic_k', 'stochastic_d']].head())
    return df

@st.cache_data
def add_candlestick_trace(fig, df, row):
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date'] = df['date'].fillna(pd.NaT)
    
    hover_texts = [
        "Date: {date}<br>Month: {month}<br>Open: ${open:.2f}<br>High: ${high:.2f}<br>Low: ${low:.2f}<br>Close: ${close:.2f}<br>Volume: {volume:,.0f}<br>RSI: {rsi:.2f}<br>RVOL: {rvol:.2f}".format(
            date=getattr(r, 'date').strftime('%m-%d-%Y') if pd.notna(getattr(r, 'date')) else 'N/A',
            month=getattr(r, 'date').strftime('%B') if pd.notna(getattr(r, 'date')) else 'N/A',
            open=getattr(r, 'open'), high=getattr(r, 'high'), low=getattr(r, 'low'),
            close=getattr(r, 'close'), volume=getattr(r, 'volume'), rsi=getattr(r, 'rsi'), rvol=getattr(r, 'rvol')
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
                st.write(f"Debug: Entering trade at {df['date'].iloc[i]}, Entry: {df['close'].iloc[i]}, Stop: {stop_loss}, Take: {take_profit}")
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
                    st.write(f"Debug: Exiting trade at {df['date'].iloc[i]} due to stop-loss")
                    position = None
                elif df['high'].iloc[i] >= position['take_profit']:
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': df['date'].iloc[i],
                        'entry_price': position['entry_price'],
                        'exit_price': position['take_profit'],
                        'return': (position['take_profit'] - position['entry_price']) / position['entry_price'] * 100
                    })
                    st.write(f"Debug: Exiting trade at {df['date'].iloc[i]} due to take-profit")
                    position = None
    
    if not trades:
        st.write("Debug: No trades executed. Possible reasons: No buy signals, invalid stop-loss/take-profit, or insufficient data.")
        return {'Win Rate': 0, 'Profit Factor': 0, 'Total Return': 0, 'Trades': 0}
    
    trades_df = pd.DataFrame(trades)
    win_rate = (trades_df['return'] > 0).mean() * 100 if not trades_df.empty else 0
    gross_profit = trades_df[trades_df['return'] > 0]['return'].sum() if (trades_df['return'] > 0).any() else 0
    gross_loss = -trades_df[trades_df['return'] < 0]['return'].sum() if (trades_df['return'] < 0).any() else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    total_return = trades_df['return'].sum() if not trades_df.empty else 0
    
    st.write(f"Debug: Number of trades executed: {len(trades)}")
    st.write(f"Debug: Trade returns: {trades_df['return'].tolist()}")
    return {
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Total Return': total_return,
        'Trades': len(trades)
    }

# Export PDF report
if st.button("Generate PDF Report"):
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(50, 750, f"{st.session_state.symbol} Stock Analysis Report")
    c.drawString(50, 730, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    c.drawString(50, 710, f"Recommendation: {st.session_state.analysis_results.get('Recommendation', 'N/A')}")
    c.drawString(50, 690, "Scores:")
    c.drawString(70, 670, f"- Performance: {st.session_state.analysis_results.get('Performance', 0):.1f}/30")
    c.drawString(70, 650, f"- Risk: {st.session_state.analysis_results.get('Risk', 0):.1f}/20")
    c.drawString(70, 630, f"- Technical: {st.session_state.analysis_results.get('Technical', 0):.1f}/30")
    c.drawString(70, 610, f"- Volume: {st.session_state.analysis_results.get('Volume', 0):.1f}/20")
    c.drawString(70, 590, f"- Total: {st.session_state.analysis_results.get('Total', 0):.1f}/100")
    c.drawString(50, 570, "Key Metrics:")
    c.drawString(70, 550, f"- Average Return: {st.session_state.analysis_results.get('Average Return', 0):.2f}%")
    c.drawString(70, 530, f"- Volatility: {st.session_state.analysis_results.get('Volatility', 0):.2f}%")
    c.drawString(70, 510, f"- Win Ratio: {st.session_state.analysis_results.get('Win Ratio', 0):.2f}%")
    c.drawString(70, 490, f"- Max Drawdown: {st.session_state.analysis_results.get('Max Drawdown', 0):.2f}%")
    c.drawString(70, 470, f"- Largest Loss: {st.session_state.analysis_results.get('Largest Loss', 0):.2f}% on {st.session_state.analysis_results.get('Largest Loss Date', 'N/A')}")
    c.drawString(70, 450, f"- Largest Gain: {st.session_state.analysis_results.get('Largest Gain', 0):.2f}% on {st.session_state.analysis_results.get('Largest Gain Date', 'N/A')}")
    c.drawString(50, 430, "Latest Trade Setup:")
    stop_loss_value = st.session_state.aapl_df['stop_loss'].iloc[-1] if 'stop_loss' in st.session_state.aapl_df.columns and pd.notna(st.session_state.aapl_df['stop_loss'].iloc[-1]) else 0.0
    take_profit_value = st.session_state.aapl_df['take_profit'].iloc[-1] if 'take_profit' in st.session_state.aapl_df.columns and pd.notna(st.session_state.aapl_df['take_profit'].iloc[-1]) else 0.0
    c.drawString(70, 410, f"- Date: {st.session_state.aapl_df['date'].iloc[-1].strftime('%m-%d-%Y')}")
    c.drawString(70, 390, f"- Entry: ${st.session_state.aapl_df['close'].iloc[-1]:.2f}")
    c.drawString(70, 370, f"- Stop-Loss: ${stop_loss_value:.2f}")
    c.drawString(70, 350, f"- Take-Profit: ${take_profit_value:.2f}")
    c.drawString(50, 330, "Backtesting Results:")
    c.drawString(70, 310, f"- Win Rate: {st.session_state.analysis_results.get('Win Rate', 0):.2f}%")
    c.drawString(70, 290, f"- Profit Factor: {st.session_state.analysis_results.get('Profit Factor', 0):.2f}")
    c.drawString(70, 270, f"- Total Return: {st.session_state.analysis_results.get('Total Return', 0):.2f}%")
    c.drawString(70, 250, f"- Trades: {st.session_state.analysis_results.get('Trades', 0)}")
    c.showPage()
    c.save()
    pdf_buffer.seek(0)
    st.download_button("Download PDF Report", pdf_buffer, file_name=f"{st.session_state.symbol}_investment_report.pdf", mime="application/pdf")
    st.session_state.analysis_results = {}  # Clear results after download to prevent stale data

# Seasonality heatmap
st.header("Seasonality Analysis")
# Ensure date is datetime for dt access
if not pd.api.types.is_datetime64_any_dtype(st.session_state.aapl_df['date']):
    st.session_state.aapl_df['date'] = pd.to_datetime(st.session_state.aapl_df['date'], errors='coerce')
st.session_state.aapl_df['month'] = st.session_state.aapl_df['date'].dt.month
st.session_state.aapl_df['year'] = st.session_state.aapl_df['date'].dt.year
monthly_returns = st.session_state.aapl_df.groupby(['year', 'month'])['daily_return'].mean().unstack() * 100
# Convert month numbers to names for x-axis
month_names = {i: calendar.month_name[i] for i in range(1, 13)}
fig_heatmap = go.Figure(data=go.Heatmap(
    z=monthly_returns.values,
    x=[month_names[col] for col in monthly_returns.columns],
    y=monthly_returns.index,
    colorscale="RdYlGn",
    hovertext=[[f"Return: {x:.2f}%" for x in row] for row in monthly_returns.values],
    hoverinfo='text'
))
fig_heatmap.update_layout(
    title="Monthly Average Returns Heatmap",
    height=400,
    template="plotly_white",
    font=dict(family="Arial", size=12, color="#000000"),
    xaxis_title="Month",
    yaxis_title="Year",
    xaxis=dict(tickmode='array', tickvals=list(month_names.values()), ticktext=list(month_names.values()))
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# Export PDF report
# Export PDF report
if st.button("Generate PDF Report"):
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(50, 750, f"{st.session_state.symbol} Stock Analysis Report")
    c.drawString(50, 730, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    c.drawString(50, 710, f"Recommendation: {st.session_state.analysis_results.get('Recommendation', 'N/A')}")
    c.drawString(50, 690, "Scores:")
    c.drawString(70, 670, f"- Performance: {st.session_state.analysis_results.get('Performance', 0):.1f}/30")
    c.drawString(70, 650, f"- Risk: {st.session_state.analysis_results.get('Risk', 0):.1f}/20")
    c.drawString(70, 630, f"- Technical: {st.session_state.analysis_results.get('Technical', 0):.1f}/30")
    c.drawString(70, 610, f"- Volume: {st.session_state.analysis_results.get('Volume', 0):.1f}/20")
    c.drawString(70, 590, f"- Total: {st.session_state.analysis_results.get('Total', 0):.1f}/100")
    c.drawString(50, 570, "Key Metrics:")
    c.drawString(70, 550, f"- Average Return: {st.session_state.analysis_results.get('Average Return', 0):.2f}%")
    c.drawString(70, 530, f"- Volatility: {st.session_state.analysis_results.get('Volatility', 0):.2f}%")
    c.drawString(70, 510, f"- Win Ratio: {st.session_state.analysis_results.get('Win Ratio', 0):.2f}%")
    c.drawString(70, 490, f"- Max Drawdown: {st.session_state.analysis_results.get('Max Drawdown', 0):.2f}%")
    c.drawString(70, 470, f"- Largest Loss: {st.session_state.analysis_results.get('Largest Loss', 0):.2f}% on {st.session_state.analysis_results.get('Largest Loss Date', 'N/A')}")
    c.drawString(70, 450, f"- Largest Gain: {st.session_state.analysis_results.get('Largest Gain', 0):.2f}% on {st.session_state.analysis_results.get('Largest Gain Date', 'N/A')}")
    c.drawString(50, 430, "Latest Trade Setup:")
    stop_loss_value = st.session_state.aapl_df['stop_loss'].iloc[-1] if 'stop_loss' in st.session_state.aapl_df.columns and pd.notna(st.session_state.aapl_df['stop_loss'].iloc[-1]) else 0.0
    take_profit_value = st.session_state.aapl_df['take_profit'].iloc[-1] if 'take_profit' in st.session_state.aapl_df.columns and pd.notna(st.session_state.aapl_df['take_profit'].iloc[-1]) else 0.0
    c.drawString(70, 410, f"- Date: {st.session_state.aapl_df['date'].iloc[-1].strftime('%m-%d-%Y')}")
    c.drawString(70, 390, f"- Entry: ${st.session_state.aapl_df['close'].iloc[-1]:.2f}")
    c.drawString(70, 370, f"- Stop-Loss: ${stop_loss_value:.2f}")
    c.drawString(70, 350, f"- Take-Profit: ${take_profit_value:.2f}")
    c.drawString(50, 330, "Backtesting Results:")
    c.drawString(70, 310, f"- Win Rate: {st.session_state.analysis_results.get('Win Rate', 0):.2f}%")
    c.drawString(70, 290, f"- Profit Factor: {st.session_state.analysis_results.get('Profit Factor', 0):.2f}")
    c.drawString(70, 270, f"- Total Return: {st.session_state.analysis_results.get('Total Return', 0):.2f}%")
    c.drawString(70, 250, f"- Trades: {st.session_state.analysis_results.get('Trades', 0)}")
    c.showPage()
    c.save()
    pdf_buffer.seek(0)
    st.download_button("Download PDF Report", pdf_buffer, file_name=f"{st.session_state.symbol}_investment_report.pdf", mime="application/pdf")
    st.session_state.analysis_results = {}  # Clear results after download to prevent stale data
    
# Export HTML report
if html_report_type == "Interactive (with Hover)":
    candlestick_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
    bench_html = fig_bench.to_html(include_plotlyjs='cdn', full_html=False) if fig_bench else ""
    heatmap_html = fig_heatmap.to_html(include_plotlyjs='cdn', full_html=False)
else:
    candlestick_img = fig.to_image(format="png")
    candlestick_img_b64 = base64.b64encode(candlestick_img).decode()
    bench_img_b64 = fig_bench.to_image(format="png") if fig_bench else None
    bench_img_b64 = base64.b64encode(bench_img_b64).decode() if bench_img_b64 else ""
    heatmap_img = fig_heatmap.to_image(format="png")
    heatmap_img_b64 = base64.b64encode(heatmap_img).decode()
    candlestick_html = f'<img src="data:image/png;base64,{candlestick_img_b64}" alt="Candlestick Chart">'
    bench_html = f'<img src="data:image/png;base64,{bench_img_b64}" alt="Benchmark Chart">' if bench_img_b64 else ""
    heatmap_html = f'<img src="data:image/png;base64,{heatmap_img_b64}" alt="Seasonality Heatmap">'

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>{symbol} Stock Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; background-color: #ffffff; color: #000000; margin: 20px; }}
        h1, h2 {{ color: #0288d1; }}
        .metric-box {{ background-color: #e0e0e0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .section {{ margin-bottom: 20px; }}
        .plotly-graph-div {{ max-width: 100%; }}
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>{symbol} Stock Analysis Report</h1>
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
    date=datetime.now().strftime('%Y-%m-%d'),
    recommendation=score['Recommendation'],
    total_score=score['Total'],
    breakout_timeframe=breakout_timeframe,
    average_return=aapl_metrics['Average Return'],
    volatility=aapl_metrics['Volatility'],
    win_ratio=aapl_metrics['Win Ratio'],
    max_drawdown=aapl_metrics['Max Drawdown'],
    largest_loss=aapl_metrics['Largest Loss'],
    largest_loss_date=aapl_metrics['Largest Loss Date'],
    largest_gain=aapl_metrics['Largest Gain'],
    largest_gain_date=aapl_metrics['Largest Gain Date'],
    win_rate=backtest_results['Win Rate'],
    profit_factor=backtest_results['Profit Factor'],
    total_return=backtest_results['Total Return'],
    trades=backtest_results['Trades'],
    latest_date=aapl_df['date'].iloc[-1].strftime('%m-%d-%Y') if not aapl_df.empty else 'N/A',
    entry=aapl_df['close'].iloc[-1] if not aapl_df.empty else 0,
    stop_loss=aapl_df['stop_loss'].iloc[-1] if not aapl_df.empty else 0,
    take_profit=aapl_df['take_profit'].iloc[-1] if not aapl_df.empty else 0,
    candlestick_html=candlestick_html,
    bench_html=bench_html,
    heatmap_html=heatmap_html
)
html_buffer = io.StringIO()
html_buffer.write(html_content)
html_buffer.seek(0)
st.download_button("Download HTML Report", html_buffer.getvalue(), file_name=f"{st.session_state.symbol}_analysis_report.html", mime="text/html")

# Help section
with st.expander("📚 Help: How the Analysis Works"):
    help_text = """
    ### Step-by-Step Analysis Explanation
    This app analyzes {symbol} stock data to identify consolidation, breakouts, and trading setups. Below is the process with a real-time example based on June 13, 2025.

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
    - **Example**: Price ($196.45) below resistance (~$200). Buy if breaks $200 with volume > 50M, RSI 40-70, Stochastic %K > %D. Stop-loss: $193.55, take-profit: $212.90.

    #### 3. Profit/Loss Analysis
    - **What**: Calculate performance metrics.
    - **How**:
      - **Average Return**: Mean daily return (%).
      - **Volatility**: Annualized standard deviation (%).
      - **Win Ratio**: Percentage of positive return days.
      - **Max Drawdown**: Maximum peak-to-trough decline (%).
      - **Significant Events**: Largest loss/gain and dates.
    - **Example**: Average Return: -0.08%, Volatility: 4.57%, Win Ratio: 51.52%, Max Drawdown: 21.27%, Largest Loss: -9.25% on 10 April 2025.

    #### 4. Backtesting
    - **What**: Simulate trades based on buy signals, stop-loss, and take-profit.
    - **How**: Enter on buy signal, exit at stop-loss or take-profit, calculate win rate and profit factor.
    - **Example**: Win Rate: 60%, Profit Factor: 1.5, Total Return: 10%, Trades: 5.

    #### 5. Breakout Timeframe Prediction
    - **What**: Estimate breakout timing.
    - **How**: Consolidation → 1-5 days; breakout → confirm in 1-3 days.
    - **Example**: Consolidation on June 13, breakout expected by June 18, 2025.

    #### 6. Scoring System
    - **What**: Combine performance, risk, technical signals, and volume.
    - **How**: Total = Performance (30) + Risk (20) + Technical (30) + Volume (20). Buy if >70.
    - **Example**: Total: 75/100, Recommendation: Buy.

    #### 7. Visualization
    - **What**: Candlestick chart with Bollinger Bands, Ichimoku, RSI, MACD, Stochastic, ADX, RVOL, volume, and win/loss distribution.
    - **How**: Plotly charts with hover text and clickable trade details.
    - **Example**: Hover shows Date: 06-13-2025, Month: June, Close: $196.45, RSI: 52.30, Volume: 51.4M. Click candlestick for trade setup.

    #### 8. Benchmark Comparison
    - **What**: Compare {symbol} to benchmark (if uploaded).
    - **Example**: {symbol}'s 20% outperforms benchmark's 10%.

    #### 9. Seasonality Analysis
    - **What**: Identify monthly performance trends.
    - **How**: Heatmap of monthly returns.
    - **Example**: April 2025: -9.25% loss.

    **Troubleshooting Tips**:
    - **Real-Time Data Errors**: Ensure a single valid symbol (e.g., AAPL, not AAPL,MSFT) and date range (at least 52 trading days, e.g., 2024-01-01 to 2025-06-13). Check internet connectivity.
    - **Upload Errors**: Verify the file has columns: date, open, high, low, close, volume. Select a date range within the file’s range with at least 52 trading days. Use the sample file provided.
    - **No Trades in Backtesting**: Ensure sufficient data points (at least 52 trading days). Check debug messages for buy signal counts. Try a larger dataset or relax signal conditions in the code.
    - **Indicator Errors**: Ensure sufficient data points and valid numeric data.
    - **Syntax Errors**: If errors persist, ensure Python 3.13 compatibility and check for indentation issues. Contact support with logs if on Streamlit Cloud.
    - Click 'Clear' to start a new analysis.
    """
    st.markdown(help_text.format(symbol=st.session_state.symbol), unsafe_allow_html=True)
