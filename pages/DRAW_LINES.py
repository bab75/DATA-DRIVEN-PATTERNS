import streamlit as st
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
import uuid

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
    h1, h2, h3 { color: #0288d1; font-family: 'Arial', sans-serif; }
    .stExpander { background-color: #f5f5f5; border-radius: 5px; }
    .metric-box { background-color: #e0e0e0; padding: 10px; border-radius: 5px; color: #000000; }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📊 AAPL Stock Analysis: Consolidation & Breakout")

# Sidebar for file upload and settings
st.sidebar.header("Upload Data")
primary_file = st.sidebar.file_uploader("Upload AAPL Data (CSV or XLSX)", type=["csv", "xlsx"])
secondary_file = st.sidebar.file_uploader("Upload Benchmark Data (CSV or XLSX, Optional)", type=["csv", "xlsx"])

st.sidebar.header("Chart Settings")
date_range = st.sidebar.date_input("Select Date Range", value=(pd.to_datetime("2025-01-01"), pd.to_datetime("2025-06-13")))
show_indicators = st.sidebar.multiselect("Select Indicators", ["Bollinger Bands", "Ichimoku Cloud", "RSI", "MACD", "Stochastic", "ADX"], default=["Bollinger Bands", "RSI"])

# Load and validate data
@st.cache_data
def load_data(primary_file, secondary_file):
    aapl_df = pd.DataFrame()
    pl_df = pd.DataFrame()
    
    if primary_file:
        try:
            if primary_file.name.endswith('.csv'):
                aapl_df = pd.read_csv(primary_file)
            elif primary_file.name.endswith('.xlsx'):
                aapl_df = pd.read_excel(primary_file)
            
            # Normalize column names (lowercase, strip whitespace)
            aapl_df.columns = aapl_df.columns.str.lower().str.strip()
            
            # Validate required columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'signal', 'stochastic_k', 'stochastic_d', 'adx', 'atr', 'senkou_span_a', 'senkou_span_b']
            actual_cols = aapl_df.columns.tolist()
            missing_cols = [col for col in required_cols if col not in actual_cols]
            if missing_cols:
                st.error(f"Missing required columns in AAPL data: {', '.join(missing_cols)}")
                st.write("Available columns:", actual_cols)
                st.write("Sample data (first 5 rows):", aapl_df.head())
                st.write("Data types:", aapl_df.dtypes)
                return pd.DataFrame(), pd.DataFrame()
            
            # Convert data types
            aapl_df['date'] = pd.to_datetime(aapl_df['date'], errors='coerce')
            for col in ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'signal', 'stochastic_k', 'stochastic_d', 'adx', 'atr', 'senkou_span_a', 'senkou_span_b']:
                aapl_df[col] = pd.to_numeric(aapl_df[col], errors='coerce')
            
            # Drop rows with null values in critical columns
            critical_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            initial_len = len(aapl_df)
            aapl_df = aapl_df.dropna(subset=critical_cols)
            if aapl_df.empty:
                st.error(f"AAPL data is empty after removing {initial_len} rows with null values in critical columns: {', '.join(critical_cols)}")
                st.write("Sample data (first 5 rows):", aapl_df.head())
                st.write("Data types:", aapl_df.dtypes)
                return pd.DataFrame(), pd.DataFrame()
            
            # Ensure no null dates
            if aapl_df['date'].isnull().any():
                st.error("Invalid or missing dates in AAPL data. Please check the 'date' column format (e.g., YYYY-MM-DD).")
                st.write("Sample data (first 5 rows):", aapl_df.head())
                return pd.DataFrame(), pd.DataFrame()
            
            # Verify numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if not pd.api.types.is_numeric_dtype(aapl_df[col]):
                    st.error(f"Column '{col}' contains non-numeric data. Please ensure all values are numbers (e.g., 196.45, not '$196.45').")
                    st.write("Sample data (first 5 rows):", aapl_df[[col]].head())
                    return pd.DataFrame(), pd.DataFrame()
            
            # Check for VWAP and warn if missing
            if 'vwap' not in aapl_df.columns:
                st.warning("VWAP column is missing in AAPL data. VWAP plot will be skipped (this is optional).")
                
        except Exception as e:
            st.error(f"Error loading AAPL data: {str(e)}. Please check the file format and content.")
            st.write("Sample data (first 5 rows):", aapl_df.head() if not aapl_df.empty else "No data loaded")
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

if primary_file:
    aapl_df, pl_df = load_data(primary_file, secondary_file)
else:
    st.warning("Please upload the AAPL data file to proceed.")
    st.stop()

if aapl_df.empty:
    st.error("Failed to load valid AAPL data. Please check the file and try again.")
    st.stop()

# Filter by date range
aapl_df = aapl_df[(aapl_df['date'] >= pd.to_datetime(date_range[0])) & (aapl_df['date'] <= pd.to_datetime(date_range[1]))]
if aapl_df.empty:
    st.error("No data available for the selected date range. Please adjust the date range or upload a different file.")
    st.stop()

# Calculate returns and metrics
def calculate_metrics(df):
    df['daily_return'] = df['close'].pct_change()
    # Handle non-finite daily returns
    if df['daily_return'].isna().any() or np.isinf(df['daily_return']).any():
        st.warning("Non-finite values (NaN or inf) detected in daily returns. Replacing with 0 for calculations.")
        df['daily_return'] = df['daily_return'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
    
    # Average Return (mean daily return)
    average_return = df['daily_return'].mean() * 100  # in percentage
    
    # Volatility (annualized standard deviation of daily returns)
    volatility = df['daily_return'].std() * np.sqrt(252) * 100  # in percentage
    
    # Win Ratio (percentage of days with positive returns)
    win_ratio = (df['daily_return'] > 0).mean() * 100  # in percentage
    
    # Annualized Return (CAGR)
    annualized_return = ((1 + df['cumulative_return'].iloc[-1]) ** (252 / len(df))) - 1 if len(df) > 0 else 0
    
    # Sharpe and Sortino Ratios
    sharpe_ratio = (annualized_return - 0.03) / (volatility / 100) if volatility > 0 else 0
    downside_returns = df['daily_return'][df['daily_return'] < 0]
    sortino_ratio = (annualized_return - 0.03) / (downside_returns.std() * np.sqrt(252)) if len(downside_returns) > 0 else 0
    
    # Max Drawdown
    drawdowns = (df['portfolio_value'] / df['portfolio_value'].cummax()) - 1 if 'portfolio_value' in df.columns else df['close'] / df['close'].cummax() - 1
    max_drawdown = drawdowns.min() * 100  # in percentage
    
    # Largest Single-Period Loss and Gain
    largest_loss = df['daily_return'].min() * 100  # in percentage
    largest_loss_date = df.loc[df['daily_return'].idxmin(), 'date'].strftime('%d %B %Y') if not df['daily_return'].empty and not np.isnan(df['daily_return'].min()) else "N/A"
    largest_gain = df['daily_return'].max() * 100  # in percentage
    largest_gain_date = df.loc[df['daily_return'].idxmax(), 'date'].strftime('%d %B %Y') if not df['daily_return'].empty and not np.isnan(df['daily_return'].max()) else "N/A"
    
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

aapl_metrics = calculate_metrics(aapl_df)

# Detect consolidation and breakout
def detect_consolidation_breakout(df):
    df['is_consolidation'] = (df.get('consolidation', pd.Series(dtype=bool)) == True) | (df['atr'] < df['atr'].mean() * 0.8) & (df['adx'] < 20)
    df['resistance'] = df['high'].rolling(20).max()
    df['support'] = df['low'].rolling(20).min()
    df['buy_signal'] = (df['close'] > df['resistance'].shift(1)) & \
                       (df['volume'] > df['volume'].mean() * 1.2) & \
                       ((df['rsi'] > 40) & (df['rsi'] < 70)) & \
                       (df['macd'] > df['signal']) & \
                       (df['stochastic_k'] < 20) & (df['stochastic_k'] > df['stochastic_d'])
    df['stop_loss'] = df['close'] - 1.5 * df['atr']
    df['take_profit'] = df['close'] + 2 * 1.5 * df['atr']  # 1:2 risk-reward
    return df

aapl_df = detect_consolidation_breakout(aapl_df)

# Technical signals
def get_signals(df):
    signals = {
        'RSI': 'Buy' if df['rsi'].iloc[-1] < 40 else 'Sell' if df['rsi'].iloc[-1] > 70 else 'Neutral',
        'MACD': 'Buy' if df['macd'].iloc[-1] > df['signal'].iloc[-1] else 'Sell',
        'Stochastic': 'Buy' if (df['stochastic_k'].iloc[-1] < 20 and df['stochastic_k'].iloc[-1] > df['stochastic_d'].iloc[-1]) else 'Sell' if (df['stochastic_k'].iloc[-1] > 80) else 'Neutral',
        'Ichimoku': 'Buy' if (df['close'].iloc[-1] > df['senkou_span_a'].iloc[-1] and df['close'].iloc[-1] > df['senkou_span_b'].iloc[-1]) else 'Sell',
        'ADX': 'Strong Trend' if df['adx'].iloc[-1] > 25 else 'Weak Trend'
    }
    return signals

signals = get_signals(aapl_df)

# Breakout timeframe prediction
def get_breakout_timeframe(df):
    if df['is_consolidation'].iloc[-1]:
        return "Breakout expected within 1-5 days"
    elif df['buy_signal'].iloc[-1]:
        return "Breakout detected; confirm within 1-3 days"
    else:
        return "No breakout expected soon"

breakout_timeframe = get_breakout_timeframe(aapl_df)

# Scoring system
def calculate_score(metrics, signals):
    performance_score = min(metrics['CAGR'], 30)
    risk_score = min((metrics['Sharpe Ratio'] * 5 + metrics['Sortino Ratio'] * 5), 20)
    technical_score = sum([10 if s in ['Buy', 'Strong Trend'] else 0 for s in signals.values()])
    volume_score = 20 if aapl_df['volume'].iloc[-1] > aapl_df['volume'].mean() else 10
    total_score = performance_score + risk_score + technical_score + volume_score
    recommendation = 'Buy' if total_score > 70 else 'Hold' if total_score > 50 else 'Avoid'
    return {
        'Performance': performance_score,
        'Risk': risk_score,
        'Technical': technical_score,
        'Volume': volume_score,
        'Total': total_score,
        'Recommendation': recommendation
    }

score = calculate_score(aapl_metrics, signals)

# Profit/Loss Analysis Section
st.header("Profit/Loss Analysis")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-box'><b>Average Return</b><br>{aapl_metrics['Average Return']:.2f}%</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-box'><b>Volatility</b><br>{aapl_metrics['Volatility']:.2f}%</div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-box'><b>Win Ratio</b><br>{aapl_metrics['Win Ratio']:.2f}%</div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-box'><b>Max Drawdown</b><br>{aapl_metrics['Max Drawdown']:.2f}%</div>", unsafe_allow_html=True)
st.markdown(f"<div class='metric-box'><b>Significant Events</b><br>Largest single-period loss was {aapl_metrics['Largest Loss']:.2f}% on {aapl_metrics['Largest Loss Date']}, indicating a significant market correction.<br>Largest single-period gain was {aapl_metrics['Largest Gain']:.2f}% on {aapl_metrics['Largest Gain Date']}.</div>", unsafe_allow_html=True)

# Decision Dashboard
st.header("Decision Dashboard")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-box'><b>Recommendation</b><br>{score['Recommendation']}</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-box'><b>Total Score</b><br>{score['Total']:.1f}/100</div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-box'><b>Breakout Timeframe</b><br>{breakout_timeframe}</div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-box'><b>CAGR</b><br>{aapl_metrics['CAGR']:.2f}%</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-box'><b>Sharpe Ratio</b><br>{aapl_metrics['Sharpe Ratio']:.2f}</div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-box'><b>RSI</b><br>{aapl_df['rsi'].iloc[-1]:.2f} ({signals['RSI']})</div>", unsafe_allow_html=True)

# Plotly candlestick chart with consolidation and breakout
fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                    subplot_titles=("Candlestick & Breakout", "RSI", "MACD & Stochastic", "ADX & Volatility", "Volume", "Win/Loss Distribution"),
                    row_heights=[0.35, 0.1, 0.1, 0.1, 0.1, 0.15])

# Candlestick chart
try:
    hover_texts = [
        f"Date: {row['date'].strftime('%m-%d-%Y')}<br>Open: ${row['open']:.2f}<br>High: ${row['high']:.2f}<br>Low: ${row['low']:.2f}<br>Close: ${row['close']:.2f}<br>Volume: {row['volume']:,.0f}"
        for _, row in aapl_df.iterrows()
    ]
    fig.add_trace(go.Candlestick(
        x=aapl_df['date'],
        open=aapl_df['open'], high=aapl_df['high'], low=aapl_df['low'], close=aapl_df['close'],
        name="Candlestick",
        increasing_line_color='#4CAF50', decreasing_line_color='#f44336',
        hovertext=hover_texts,
        hoverinfo='text',
        customdata=aapl_df['volume']
    ), row=1, col=1)
    fig.update_traces(xhoverformat="%m-%d-%Y", row=1, col=1)
except Exception as e:
    st.error(f"Error plotting candlestick chart: {str(e)}.")
    st.write("Available columns:", aapl_df.columns.tolist())
    st.write("Sample data (first 5 rows):", aapl_df[['date', 'open', 'high', 'low', 'close', 'volume']].head())
    st.write("Data types:", aapl_df[['date', 'open', 'high', 'low', 'close', 'volume']].dtypes)
    st.stop()

if "Bollinger Bands" in show_indicators and 'std_dev' in aapl_df.columns and 'ma20' in aapl_df.columns:
    fig.add_trace(go.Scatter(x=aapl_df['date'], y=aapl_df['ma20'] + 2*aapl_df['std_dev'], name="Bollinger Upper", line=dict(color="#0288d1")), row=1, col=1)
    fig.add_trace(go.Scatter(x=aapl_df['date'], y=aapl_df['ma20'] - 2*aapl_df['std_dev'], name="Bollinger Lower", line=dict(color="#0288d1"), fill='tonexty', fillcolor='rgba(2,136,209,0.1)'), row=1, col=1)
if "Ichimoku Cloud" in show_indicators:
    fig.add_trace(go.Scatter(x=aapl_df['date'], y=aapl_df['senkou_span_a'], name="Senkou Span A", line=dict(color="#4CAF50"), fill='tonexty', fillcolor='rgba(76,175,80,0.2)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=aapl_df['date'], y=aapl_df['senkou_span_b'], name="Senkou Span B", line=dict(color="#f44336"), fill='tonexty', fillcolor='rgba(244,67,54,0.2)'), row=1, col=1)

# Add breakout signals
buy_signals = aapl_df[aapl_df['buy_signal'] == True]
for _, row in buy_signals.iterrows():
    fig.add_annotation(x=row['date'], y=row['high'], text="Buy", showarrow=True, arrowhead=2, ax=0, ay=-30, font=dict(color="#000000"), row=1, col=1)

# Add stop-loss and take-profit
if not buy_signals.empty:
    latest_buy = buy_signals.iloc[-1]
    fig.add_hline(y=latest_buy['stop_loss'], line_dash="dash", line_color="#f44336", annotation_text="Stop-Loss", annotation_font_color="#000000", row=1, col=1)
    fig.add_hline(y=latest_buy['take_profit'], line_dash="dash", line_color="#4CAF50", annotation_text="Take-Profit", annotation_font_color="#000000", row=1, col=1)

# RSI chart
if "RSI" in show_indicators:
    fig.add_trace(go.Scatter(x=aapl_df['date'], y=aapl_df['rsi'], name="RSI", line=dict(color="#9c27b0"),
                             hovertext=[f"RSI: {x:.2f}" for x in aapl_df['rsi']], hoverinfo='text+x'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#f44336", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#4CAF50", row=2, col=1)

# MACD & Stochastic chart
if "MACD" in show_indicators:
    fig.add_trace(go.Scatter(x=aapl_df['date'], y=aapl_df['macd'], name="MACD", line=dict(color="#0288d1"),
                             hovertext=[f"MACD: {x:.2f}" for x in aapl_df['macd']], hoverinfo='text+x'), row=3, col=1)
    fig.add_trace(go.Scatter(x=aapl_df['date'], y=aapl_df['signal'], name="Signal Line", line=dict(color="#ff9800"),
                             hovertext=[f"Signal: {x:.2f}" for x in aapl_df['signal']], hoverinfo='text+x'), row=3, col=1)
if "Stochastic" in show_indicators:
    fig.add_trace(go.Scatter(x=aapl_df['date'], y=aapl_df['stochastic_k'], name="Stochastic %K", line=dict(color="#e91e63"), yaxis="y2",
                             hovertext=[f"Stochastic %K: {x:.2f}" for x in aapl_df['stochastic_k']], hoverinfo='text+x'), row=3, col=1)
    fig.add_trace(go.Scatter(x=aapl_df['date'], y=aapl_df['stochastic_d'], name="Stochastic %D", line=dict(color="#ff5722"), yaxis="y2",
                             hovertext=[f"Stochastic %D: {x:.2f}" for x in aapl_df['stochastic_d']], hoverinfo='text+x'), row=3, col=1)
    fig.update_layout(yaxis2=dict(overlaying='y', side='right', range=[0, 100]))

# ADX & Volatility chart
if "ADX" in show_indicators:
    fig.add_trace(go.Scatter(x=aapl_df['date'], y=aapl_df['adx'], name="ADX", line=dict(color="#3f51b5"),
                             hovertext=[f"ADX: {x:.2f}" for x in aapl_df['adx']], hoverinfo='text+x'), row=4, col=1)
    fig.add_hline(y=25, line_dash="dash", line_color="#0288d1", row=4, col=1)
fig.add_trace(go.Scatter(x=aapl_df['date'], y=aapl_df['atr'], name="ATR", line=dict(color="#795548"),
                         hovertext=[f"ATR: ${x:.2f}" for x in aapl_df['atr']], hoverinfo='text+x'), row=4, col=1)

# Volume chart
fig.add_trace(go.Bar(x=aapl_df['date'], y=aapl_df['volume'], name="Volume", marker_color="#607d8b",
                     hovertext=[f"Volume: {x:,.0f}" for x in aapl_df['volume']], hoverinfo='text+x'), row=5, col=1)
if 'vwap' in aapl_df.columns:
    fig.add_trace(go.Scatter(x=aapl_df['date'], y=aapl_df['vwap'], name="VWAP", line=dict(color="#0288d1"),
                             hovertext=[f"VWAP: ${x:.2f}" for x in aapl_df['vwap']], hoverinfo='text+x'), row=5, col=1)

# Win/Loss Distribution chart
try:
    valid_returns = aapl_df['daily_return'][aapl_df['daily_return'].notna() & ~aapl_df['daily_return'].isin([np.inf, -np.inf])]
    if not valid_returns.empty:
        bins = np.histogram_bin_edges(valid_returns * 100, bins=20)
        hist_data = np.histogram(valid_returns * 100, bins=bins)
        fig.add_trace(go.Bar(x=bins[:-1], y=hist_data[0], name="Win/Loss Distribution", marker_color="#607d8b",
                             hovertext=[f"Return: {x:.2f}% Count: {y}" for x, y in zip(bins[:-1], hist_data[0])], hoverinfo='text'), row=6, col=1)
    else:
        st.warning("Cannot plot Win/Loss Distribution: No valid daily returns available (all NaN or inf).")
except Exception as e:
    st.warning(f"Error plotting Win/Loss Distribution: {str(e)}. Skipping this chart.")

fig.update_layout(height=1200, showlegend=True, template="plotly_white", title_text="AAPL Candlestick Analysis",
                  hovermode="x unified", font=dict(family="Arial", size=12, color="#000000"))
fig.update_xaxes(rangeslider_visible=True, tickformat="%m-%d-%Y", row=6, col=1)
st.plotly_chart(fig, use_container_width=True)

# Benchmark comparison
if not pl_df.empty:
    st.header("Benchmark Comparison")
    try:
        pl_cum_return = (1 + pl_df['Profit/Loss (Percentage)']).cumprod() - 1
        fig_bench = go.Figure()
        fig_bench.add_trace(go.Scatter(x=aapl_df['date'], y=aapl_df['cumulative_return'], name="AAPL", line=dict(color="#0288d1"),
                                       hovertext=[f"AAPL Return: {x:.2%}" for x in aapl_df['cumulative_return']], hoverinfo='text+x'))
        fig_bench.add_trace(go.Scatter(x=pl_df['End Date'], y=pl_cum_return, name="Benchmark", line=dict(color="#ff9800"),
                                       hovertext=[f"Benchmark Return: {x:.2%}" for x in pl_cum_return], hoverinfo='text+x'))
        fig_bench.update_layout(title="AAPL vs. Benchmark Cumulative Returns", height=400, template="plotly_white",
                                hovermode="x unified", font=dict(family="Arial", size=12, color="#000000"), xaxis_tickformat="%m-%d-%Y")
        st.plotly_chart(fig_bench, use_container_width=True)
    except Exception as e:
        st.warning(f"Error plotting benchmark comparison: {str(e)}. Skipping benchmark chart.")

# Seasonality heatmap
st.header("Seasonality Analysis")
aapl_df['month'] = aapl_df['date'].dt.month
aapl_df['year'] = aapl_df['date'].dt.year
monthly_returns = aapl_df.groupby(['year', 'month'])['daily_return'].mean().unstack() * 100
fig_heatmap = go.Figure(data=go.Heatmap(z=monthly_returns.values, x=monthly_returns.columns, y=monthly_returns.index,
                                        colorscale="RdYlGn", hovertext=[[f"Return: {x:.2f}%" for x in row] for row in monthly_returns.values], hoverinfo='text'))
fig_heatmap.update_layout(title="Monthly Average Returns Heatmap", height=400, template="plotly_white",
                          font=dict(family="Arial", size=12, color="#000000"), xaxis_title="Month", yaxis_title="Year")
st.plotly_chart(fig_heatmap, use_container_width=True)

# Download report
st.header("Export Report")
buffer = io.BytesIO()
c = canvas.Canvas(buffer, pagesize=letter)
c.setFont("Helvetica", 12)
c.drawString(50, 750, "Stock Investment Analysis Report")
c.drawString(50, 730, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
c.drawString(50, 710, f"Recommendation: {score['Recommendation']}")
c.drawString(50, 690, "Scores:")
c.drawString(70, 670, f"- Performance: {score['Performance']:.1f}/30")
c.drawString(70, 650, f"- Risk: {score['Risk']:.1f}/20")
c.drawString(70, 630, f"- Technical: {score['Technical']:.1f}/30")
c.drawString(70, 610, f"- Volume: {score['Volume']:.1f}/20")
c.drawString(70, 590, f"- Total: {score['Total']:.1f}/100")
c.drawString(50, 570, "Key Metrics:")
c.drawString(70, 550, f"- Average Return: {aapl_metrics['Average Return']:.2f}%")
c.drawString(70, 530, f"- Volatility: {aapl_metrics['Volatility']:.2f}%")
c.drawString(70, 510, f"- Win Ratio: {aapl_metrics['Win Ratio']:.2f}%")
c.drawString(70, 490, f"- Max Drawdown: {aapl_metrics['Max Drawdown']:.2f}%")
c.drawString(70, 470, f"- Largest Loss: {aapl_metrics['Largest Loss']:.2f}% on {aapl_metrics['Largest Loss Date']}")
c.drawString(70, 450, f"- Largest Gain: {aapl_metrics['Largest Gain']:.2f}% on {aapl_metrics['Largest Gain Date']}")
c.drawString(50, 430, "Latest Trade Setup:")
c.drawString(70, 410, f"- Entry: ${aapl_df['close'].iloc[-1]:.2f}")
c.drawString(70, 390, f"- Stop-Loss: ${aapl_df['stop_loss'].iloc[-1]:.2f}")
c.drawString(70, 370, f"- Take-Profit: ${aapl_df['take_profit'].iloc[-1]:.2f}")
c.showPage()
c.save()
buffer.seek(0)
st.download_button("Download PDF Report", buffer, file_name="investment_report.pdf", mime="application/pdf")

# Help section
with st.expander("📚 Help: How the Analysis Works"):
    st.markdown("""
    ### Step-by-Step Analysis Explanation
    This app analyzes AAPL stock data to identify consolidation, breakouts, and trading setups, mimicking how analysts draw charts. Below is the process with a real-time example based on June 13, 2025.

    #### 1. Data Collection
    - **What**: Use OHLC, volume, and technical indicators (RSI, MACD, Stochastic, Ichimoku, ADX, ATR) from `AAPL_raw_data.csv`.
    - **Example**: Latest data (June 13, 2025) shows Close: $196.45, RSI: 52.30, Stochastic %K: 7.12, %D: 8.00, ADX: 31.79, Volume: 51.4M, ATR: $4.30.

    #### 2. Candlestick Analysis & Breakout Detection
    - **What**: Visualize price movements to identify consolidation (tight range) and breakouts (price above resistance).
    - **How**:
      - **Candlesticks**: Plot OHLC as green (bullish) or red (bearish) candles.
      - **Consolidation**: Low ATR (< mean * 0.8), ADX < 20, or `consolidation` = True. Narrow Bollinger Bands or price in Ichimoku Cloud.
      - **Breakout**: Close > 20-day high (resistance), volume > 1.2 * average, RSI 40-70, MACD > signal, Stochastic %K < 20 and > %D.
      - **Stop-Loss/Take-Profit**: Stop-loss = close - 1.5 * ATR; take-profit = close + 3 * ATR (1:2 risk-reward).
    - **Example**: Price ($196.45) below resistance (~$200). If breaks $200 with volume > 60M, RSI 40-70, Stochastic %K (7.12) < 20 and > %D (8.00), a buy signal triggers. Stop-loss: $193.55, take-profit: $212.90.

    #### 3. Profit/Loss Analysis
    - **What**: Calculate performance metrics to assess risk and return.
    - **How**:
      - **Average Return**: Mean daily return (%).
      - **Volatility**: Annualized standard deviation of daily returns (%).
      - **Win Ratio**: Percentage of days with positive returns.
      - **Max Drawdown**: Maximum peak-to-trough decline (%).
      - **Significant Events**: Largest loss/gain and dates.
    - **Example**: Average Return: -0.08%, Volatility: 4.57%, Win Ratio: 51.52%, Max Drawdown: 21.27%, Largest Loss: -9.25% on 10 April 2025, Largest Gain: 8.50% on 15 March 2025.

    #### 4. Breakout Timeframe Prediction
    - **What**: Estimate when a breakout may occur.
    - **How**: If in consolidation, expect breakout within 1-5 days; if breakout detected, confirm within 1-3 days.
    - **Example**: If consolidation on June 13, breakout expected by June 18, 2025.

    #### 5. Scoring System
    - **What**: Combine performance, risk, technical signals, and volume for a recommendation.
    - **How**: Total = Performance (30) + Risk (20) + Technical (30) + Volume (20). Buy if >70.
    - **Example**: Performance (25, CAGR ~20%) + Risk (15, Sharpe ~0.68) + Technical (20, Stochastic/ADX) + Volume (15, high volume) = 75. Recommendation: Buy.

    #### 6. Visualization
    - **What**: Candlestick chart with Bollinger Bands, Ichimoku, RSI, MACD, Stochastic, ADX, ATR, volume, and win/loss distribution.
    - **How**: Plotly charts with hover text (e.g., date, OHLC, indicators). Breakout signals annotated with arrows.
    - **Example**: Hover over June 13 shows Close: $196.45, RSI: 52.30, Volume: 51.4M. Buy signal at $200.

    #### 7. Benchmark Comparison
    - **What**: Compare AAPL to a benchmark (if uploaded).
    - **Example**: If benchmark returns 10% in 2025, AAPL’s 20% outperforms, supporting a buy.

    #### 8. Seasonality Analysis
    - **What**: Identify monthly performance trends.
    - **How**: Heatmap of average monthly returns.
    - **Example**: April 2025 shows -9.25% loss, indicating a correction.

    This analysis provides a data-driven trading setup, balancing technical signals and risk management. Use the charts to monitor for breakouts.
    """)
