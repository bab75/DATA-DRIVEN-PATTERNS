import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
from ta.trend import ADXIndicator
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Streamlit app configuration
st.set_page_config(page_title="Stock Technical Analysis", layout="wide", page_icon="📈")
# Enhanced CSS with modern design
try:
    st.markdown("""
        <style>
        body {background: linear-gradient(135deg, #f0f4f8, #d9e2ec); font-family: 'Segoe UI', sans-serif;}
        .stButton>button {background: linear-gradient(45deg, #4a90e2, #63b3ed); color: white; border: none; border-radius: 10px; padding: 10px 20px; transition: all 0.3s;}
        .stButton>button:hover {background: linear-gradient(45deg, #357abd, #4a90e2); transform: scale(1.05);}
        .stSelectbox, .stTextInput, .stNumberInput {background: #ffffff; border: 1px solid #d1d9e6; border-radius: 10px; padding: 5px;}
        .report-container {background: #ffffff; padding: 20px; border-radius: 15px; box-shadow: 0 6px 12px rgba(0,0,0,0.1); margin-bottom: 20px;}
        h1 {color: #2c3e50; font-size: 2.5em; text-align: center; text-transform: uppercase; letter-spacing: 2px;}
        h2, h3 {color: #34495e; font-weight: 500;}
        .mode-banner {background: #e6f3fa; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #3498db;}
        </style>
    """, unsafe_allow_html=True)
except TypeError:
    st.markdown("<!-- Custom CSS not applied due to Streamlit version incompatibility -->")

# Initialize session state
if 'real_time_data' not in st.session_state:
    st.session_state.real_time_data = None
if 'fundamental_data' not in st.session_state:
    st.session_state.fundamental_data = {'EPS': None, 'P/E': None, 'PEG': None, 'P/B': None, 'ROE': None, 'Revenue': None, 'Debt/Equity': None}
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'combine_report' not in st.session_state:
    st.session_state.combine_report = False

# Clear Analysis Button
def clear_analysis():
    st.session_state.real_time_data = None
    st.session_state.fundamental_data = {'EPS': None, 'P/E': None, 'PEG': None, 'P/B': None, 'ROE': None, 'Revenue': None, 'Debt/Equity': None}
    st.session_state.csv_data = None
    st.session_state.combine_report = False
    st.success("Analysis cleared. Start a new analysis.")

# Sidebar for inputs with reorganized navigation
st.sidebar.header("📊 Stock Analysis Controls")

# Step 1: Export File and Process Button
st.sidebar.subheader("📤 Upload Technical Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"], help="Upload a file with technical indicators.")
process_file_button = st.sidebar.button("📥 Process File")

if process_file_button and uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Volatility', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50', 'SMA_200', 'EMA_200', 'BB_Width', 'BB_Position', 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_A', 'Ichimoku_Senkou_B', 'Ichimoku_Chikou', 'PSAR', 'PSAR_Bull', 'PSAR_Bear', 'Stoch_K', 'Williams_R', 'CCI', 'Momentum', 'ROC', 'ATR', 'Keltner_Upper', 'Keltner_Lower', 'OBV', 'VWAP', 'Volume_SMA', 'MFI', 'Pivot', 'R1', 'S1', 'R2', 'S2', 'Fib_236', 'Fib_382', 'Fib_618']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            st.error(f"❌ Missing columns: {', '.join(missing)}. Ensure all required columns are present.")
        else:
            df['Date'] = pd.to_datetime(df['Date'])
            st.session_state.csv_data = df
            st.success("✅ File processed successfully! The 'Combine Report' checkbox is now enabled.")
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}. Ensure the file is a valid CSV/XLSX with required columns.")

# Step 2: Symbol Field and Fetch Real-Time Data Button
st.sidebar.subheader("📡 Real-Time Data")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., KRRO)", value="KRRO", help="Enter a valid stock ticker symbol.")
submit_button = st.sidebar.button("🔄 Fetch Real-Time Data")

if submit_button:
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d", interval="1d")
        if data.empty:
            st.error("❌ No data found for the ticker. Please try a different ticker (e.g., AAPL) or upload a CSV/XLSX.")
        else:
            latest = data.iloc[-1]
            st.session_state.real_time_data = {
                'Date': data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                'Open': latest['Open'],
                'High': latest['High'],
                'Low': latest['Low'],
                'Close': latest['Close'],
                'Volume': latest['Volume']
            }
            info = stock.info
            st.session_state.fundamental_data.update({
                'EPS': info.get('trailingEps', st.session_state.fundamental_data['EPS']),
                'P/E': info.get('trailingPE', st.session_state.fundamental_data['P/E']),
                'PEG': info.get('pegRatio', st.session_state.fundamental_data['PEG']),
                'P/B': info.get('priceToBook', st.session_state.fundamental_data['P/B']),
                'ROE': info.get('returnOnEquity', st.session_state.fundamental_data['ROE']),
                'Revenue': info.get('totalRevenue', st.session_state.fundamental_data['Revenue']),
                'Debt/Equity': info.get('debtToEquity', st.session_state.fundamental_data['Debt/Equity'])
            })
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}. Please try a different ticker (e.g., AAPL) or upload a CSV/XLSX.")

# Step 3: Combine Checkbox and Combine Report Button
st.sidebar.subheader("📈 Report Options")
combine_checkbox = st.sidebar.checkbox("Combine Report", value=st.session_state.combine_report, disabled=st.session_state.csv_data is None)
if combine_checkbox != st.session_state.combine_report:
    st.session_state.combine_report = combine_checkbox
combine_button = st.sidebar.button("📊 Combine Process")

# Manual fundamental inputs
with st.sidebar.expander("📋 Manual Fundamental Data (Optional)", expanded=False):
    with st.form(key="fundamental_form"):
        eps = st.number_input("EPS", value=float(st.session_state.fundamental_data['EPS'] or 0.0), step=0.01, format="%.2f", placeholder="e.g., -9.42")
        pe = st.number_input("P/E Ratio", value=float(st.session_state.fundamental_data['P/E'] or 0.0), step=0.01, format="%.2f")
        peg = st.number_input("PEG Ratio", value=float(st.session_state.fundamental_data['PEG'] or 0.0), step=0.01, format="%.2f")
        pb = st.number_input("P/B Ratio", value=float(st.session_state.fundamental_data['P/B'] or 0.0), step=0.01, format="%.2f")
        roe = st.number_input("ROE (%)", value=float(st.session_state.fundamental_data['ROE'] or 0.0), step=0.01, format="%.2f")
        revenue = st.number_input("Revenue (M)", value=float(st.session_state.fundamental_data['Revenue'] or 0.0), step=0.1, format="%.1f")
        debt_equity = st.number_input("Debt/Equity", value=float(st.session_state.fundamental_data['Debt/Equity'] or 0.0), step=0.01, format="%.2f")
        submit_fundamentals = st.form_submit_button("💾 Update Fundamentals")

if submit_fundamentals:
    st.session_state.fundamental_data = {
        'EPS': eps if eps != 0.0 else None,
        'P/E': pe if pe != 0.0 else None,
        'PEG': peg if peg != 0.0 else None,
        'P/B': pb if pb != 0.0 else None,
        'ROE': roe if roe != 0.0 else None,
        'Revenue': revenue if revenue != 0.0 else None,
        'Debt/Equity': debt_equity if debt_equity != 0.0 else None
    }

# Clear Analysis Button
st.sidebar.subheader("🗑️ Reset")
clear_button = st.sidebar.button("Clear Analysis")
if clear_button:
    clear_analysis()

# Function to combine data
def combine_dataframes(csv_df, real_time_data):
    if csv_df is None or real_time_data is None:
        return csv_df if csv_df is not None else pd.DataFrame([real_time_data]) if real_time_data is not None else None
    real_time_df = pd.DataFrame([real_time_data])
    real_time_df['Date'] = pd.to_datetime(real_time_df['Date'])
    combined_df = pd.concat([csv_df, real_time_df], ignore_index=True)
    combined_df = combined_df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')
    # Update with real-time values where applicable
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in real_time_df.columns:
            combined_df[col].iloc[-1] = real_time_df[col].iloc[0]
    return combined_df

# Function to generate PDF report
def generate_pdf_report(report_content, stock_name, report_type):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    custom_style = ParagraphStyle(name='Custom', parent=styles['Normal'], fontSize=12, leading=14)
    heading_style = ParagraphStyle(name='Heading', parent=styles['Heading1'], fontSize=16, leading=18, spaceAfter=12)
    elements = []

    elements.append(Paragraph(f"Stock Analysis Report: {stock_name} ({datetime.now().strftime('%Y-%m-%d')})", heading_style))
    elements.append(Spacer(1, 12))

    for line in report_content.split('\n'):
        if line.startswith('### '):
            elements.append(Paragraph(line[4:], heading_style))
        elif line.startswith('- **'):
            text = line.replace('- **', '').replace('**', '')
            elements.append(Paragraph(f"• {text}", custom_style))
        elif line.startswith('  - '):
            text = line.replace('  - ', '')
            elements.append(Paragraph(f"  • {text}", custom_style))
        else:
            elements.append(Paragraph(line, custom_style))
        elements.append(Spacer(1, 6))

    if any(v is not None for v in st.session_state.fundamental_data.values()):
        elements.append(Paragraph("Fundamentals", heading_style))
        data = [['Metric', 'Value']] + [[k, f"{v:.2f}" if v is not None else "N/A"] for k, v in st.session_state.fundamental_data.items()]
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# Function to analyze stock data
def analyze_stock_data(df=None, real_time_data=None, fundamental_data=None):
    stock_name = ticker.upper()
    is_real_time_only = real_time_data is not None and df is None
    data_source = df if df is not None else pd.DataFrame([real_time_data]) if real_time_data is not None else None

    if data_source is None:
        return None, None, None, stock_name, None, is_real_time_only, 50

    latest = data_source.iloc[-1]
    prev = data_source.iloc[-2] if len(data_source) > 1 else latest

    price = latest['Close']
    volatility = latest.get('Volatility', data_source['Volatility'].mean() if not is_real_time_only and len(data_source) > 1 else 0)
    rsi = latest.get('RSI', 50) if not is_real_time_only and len(data_source) > 1 else 50
    macd = latest.get('MACD', 0) if not is_real_time_only and len(data_source) > 1 else 0
    macd_signal = latest.get('MACD_Signal', 0) if not is_real_time_only and len(data_source) > 1 else 0
    macd_hist = latest.get('MACD_Histogram', 0) if not is_real_time_only and len(data_source) > 1 else 0
    bb_upper = latest.get('BB_Upper', price * 1.05) if not is_real_time_only and len(data_source) > 1 else price * 1.05
    bb_middle = latest.get('BB_Middle', price) if not is_real_time_only and len(data_source) > 1 else price
    bb_lower = latest.get('BB_Lower', price * 0.95) if not is_real_time_only and len(data_source) > 1 else price * 0.95
    bb_width = latest.get('BB_Width', data_source['BB_Width'].mean() if not is_real_time_only and len(data_source) > 1 else 0)
    bb_position = latest.get('BB_Position', data_source['BB_Position'].mean() if not is_real_time_only and len(data_source) > 1 else 0)
    sma_20 = latest.get('SMA_20', price) if not is_real_time_only and len(data_source) > 1 else price
    ema_20 = latest.get('EMA_20', price) if not is_real_time_only and len(data_source) > 1 else price
    sma_50 = latest.get('SMA_50', price) if not is_real_time_only and len(data_source) > 1 else price
    ema_50 = latest.get('EMA_50', price) if not is_real_time_only and len(data_source) > 1 else price
    sma_200 = latest.get('SMA_200', price) if not is_real_time_only and len(data_source) > 1 else price
    ema_200 = latest.get('EMA_200', price) if not is_real_time_only and len(data_source) > 1 else price
    ichimoku_tenkan = latest.get('Ichimoku_Tenkan', price) if not is_real_time_only and len(data_source) > 1 else price
    ichimoku_kijun = latest.get('Ichimoku_Kijun', price) if not is_real_time_only and len(data_source) > 1 else price
    ichimoku_senkou_a = latest.get('Ichimoku_Senkou_A', price) if not is_real_time_only and len(data_source) > 1 else price
    ichimoku_senkou_b = latest.get('Ichimoku_Senkou_B', price) if not is_real_time_only and len(data_source) > 1 else price
    ichimoku_chikou = latest.get('Ichimoku_Chikou', price) if not is_real_time_only and len(data_source) > 1 else price
    psar = latest.get('PSAR', price) if not is_real_time_only and len(data_source) > 1 else price
    psar_bull = latest.get('PSAR_Bull', price) if not is_real_time_only and len(data_source) > 1 else price
    psar_bear = latest.get('PSAR_Bear', price) if not is_real_time_only and len(data_source) > 1 else price
    stoch_k = latest.get('Stoch_K', 50) if not is_real_time_only and len(data_source) > 1 else 50
    williams_r = latest.get('Williams_R', -50) if not is_real_time_only and len(data_source) > 1 else -50
    cci = latest.get('CCI', 0) if not is_real_time_only and len(data_source) > 1 else 0
    momentum = latest.get('Momentum', 0) if not is_real_time_only and len(data_source) > 1 else 0
    roc = latest.get('ROC', 0) if not is_real_time_only and len(data_source) > 1 else 0
    atr = latest.get('ATR', data_source['ATR'].mean() if not is_real_time_only and len(data_source) > 1 else 0)
    keltner_upper = latest.get('Keltner_Upper', price * 1.05) if not is_real_time_only and len(data_source) > 1 else price * 1.05
    keltner_lower = latest.get('Keltner_Lower', price * 0.95) if not is_real_time_only and len(data_source) > 1 else price * 0.95
    obv = latest.get('OBV', latest['Volume']) if not is_real_time_only and len(data_source) > 1 else latest['Volume']
    vwap = latest.get('VWAP', price) if not is_real_time_only and len(data_source) > 1 else price
    volume_sma = latest.get('Volume_SMA', latest['Volume']) if not is_real_time_only and len(data_source) > 1 else latest['Volume']
    mfi = latest.get('MFI', 50) if not is_real_time_only and len(data_source) > 1 else 50
    pivot = latest.get('Pivot', price) if not is_real_time_only and len(data_source) > 1 else price
    r1 = latest.get('R1', price * 1.02) if not is_real_time_only and len(data_source) > 1 else price * 1.02
    s1 = latest.get('S1', price * 0.98) if not is_real_time_only and len(data_source) > 1 else price * 0.98
    r2 = latest.get('R2', price * 1.04) if not is_real_time_only and len(data_source) > 1 else price * 1.04
    s2 = latest.get('S2', price * 0.96) if not is_real_time_only and len(data_source) > 1 else price * 0.96
    fib_618 = latest.get('Fib_618', price * 1.01) if not is_real_time_only and len(data_source) > 1 else price * 1.01

    adx_value = 50
    if not is_real_time_only and all(col in data_source.columns for col in ['High', 'Low', 'Close']) and len(data_source) >= 14:
        adx_indicator = ADXIndicator(data_source['High'], data_source['Low'], data_source['Close'], window=14)
        adx_value = adx_indicator.adx().iloc[-1]

    trend_pattern = "Neutral"
    if not is_real_time_only and len(data_source) > 10:
        highs = data_source['High'].rolling(window=10).max()
        lows = data_source['Low'].rolling(window=10).min()
        if highs.iloc[-1] > highs.iloc[-2] and lows.iloc[-1] > lows.iloc[-2]:
            trend_pattern = "Higher Highs & Lows (Bullish)"
        elif highs.iloc[-1] < highs.iloc[-2] and lows.iloc[-1] < lows.iloc[-2]:
            trend_pattern = "Lower Highs & Lows (Bearish)"

    trend = "Bearish" if price < sma_20 and price < sma_50 else "Bullish" if price > sma_20 and price > sma_50 else "Neutral"

    quick_scan = f"""
### Quick Scan: {stock_name} ({latest['Date'] if isinstance(latest['Date'], str) else latest['Date'].strftime('%Y-%m-%d')})
- **Price**: ${price:.2f} ({trend} trend)
- **Support/Resistance**: Support at ${s1:.2f}, Resistance at ${r1:.2f}
- **RSI**: {rsi:.2f} ({'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'})
- **Volatility**: {volatility:.2f}%
- **Recommendation**: {'Buy near support (${s1:.2f}) for bounce to ${r1:.2f}' if rsi < 30 else f'Wait for breakout above ${sma_20:.2f}'}
"""

    moderate_detail = f"""
### Moderate Detail: {stock_name} ({latest['Date'] if isinstance(latest['Date'], str) else latest['Date'].strftime('%Y-%m-%d')})
- **Price Trend**: ${price:.2f}, {trend} (SMA20: ${sma_20:.2f}, SMA50: ${sma_50:.2f})
- **Momentum**:
  - RSI: {rsi:.2f} ({'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'})
  - MACD: {macd:.2f} (Signal: {macd_signal:.2f}, {'Bearish' if macd < macd_signal else 'Bullish'})
- **Volatility**: {volatility:.2f}% (ATR: ${atr:.2f})
- **Bollinger Bands**: Width: {bb_width:.2f}%, Position: {bb_position:.2f}% (Lower: ${bb_lower:.2f}, Upper: ${bb_upper:.2f})
- **ADX**: {adx_value:.2f} ({'Strong Trend' if adx_value > 25 else 'Weak Trend'})
- **Key Levels**: Support: ${s1:.2f}, Resistance: ${r1:.2f}, Fib 61.8%: ${fib_618:.2f}
- **Fundamentals**:
  {''.join([f'  - {k}: {v:.2f}\n' for k, v in fundamental_data.items() if v is not None]) if fundamental_data and any(v is not None for v in fundamental_data.values()) else '  - Not provided\n'}
- **Recommendation**:
  - Traders: {'Buy near ${s1:.2f} for bounce to ${r1:.2f}' if rsi < 30 or price < bb_middle else f'Wait for breakout above ${r1:.2f}'}
  - Investors: Confirm trend reversal above ${sma_20:.2f}.
"""

    in_depth = f"""
### In-Depth Analysis: {stock_name} ({latest['Date'] if isinstance(latest['Date'], str) else latest['Date'].strftime('%Y-%m-%d')})
#### Key Takeaways
- **Price**: ${price:.2f}, {trend} trend
- **Historical Pattern**: {trend_pattern}
- **ADX**: {adx_value:.2f} ({'Strong Trend' if adx_value > 25 else 'Weak Trend'})
- **Volatility**: {volatility:.2f}% (ATR: ${atr:.2f})

#### Price Trends
- **Close**: ${price:.2f}, {'below' if price < sma_20 else 'above'} SMA20 (${sma_20:.2f}), SMA50 (${sma_50:.2f}), SMA200 (${sma_200:.2f})
- **Trend**: {trend} (EMA20: ${ema_20:.2f}, EMA50: ${ema_50:.2f}, EMA200: ${ema_200:.2f})

#### Momentum Indicators
- **RSI**: {rsi:.2f} ({'Oversold (<30)' if rsi < 30 else 'Overbought (>70)' if rsi > 70 else 'Neutral'})
- **MACD**: {macd:.2f}, Signal: {macd_signal:.2f}, Histogram: {macd_hist:.2f} ({'Bearish' if macd < macd_signal else 'Bullish'})
- **Stochastic %K**: {stoch_k:.2f} ({'Oversold' if stoch_k < 20 else 'Overbought' if stoch_k > 80 else 'Neutral'})
- **Williams %R**: {williams_r:.2f} ({'Oversold' if williams_r < -80 else 'Overbought' if williams_r > -20 else 'Neutral'})
- **CCI**: {cci:.2f}, Momentum: {momentum:.2f}, ROC: {roc:.2f}
- **MFI**: {mfi:.2f} ({'Overbought' if mfi > 80 else 'Oversold' if mfi < 20 else 'Neutral'})

#### Volatility & Bands
- **Bollinger Bands**: Width: {bb_width:.2f}%, Position: {bb_position:.2f}% (Lower: ${bb_lower:.2f}, Middle: ${bb_middle:.2f}, Upper: ${bb_upper:.2f})
- **Keltner Channels**: Upper: ${keltner_upper:.2f}, Lower: ${keltner_lower:.2f}

#### Volume
- **OBV**: {obv:,.0f} ({'Declining' if obv < prev.get('OBV', obv) else 'Rising'})
- **Volume**: {latest['Volume']:,.0f} (SMA: {volume_sma:,.0f})
- **VWAP**: ${vwap:.2f}

#### Ichimoku Cloud
- **Tenkan**: ${ichimoku_tenkan:.2f}, Kijun: ${ichimoku_kijun:.2f}
- **Senkou A**: ${ichimoku_senkou_a:.2f}, Senkou B: ${ichimoku_senkou_b:.2f}
- **Chikou**: ${ichimoku_chikou:.2f}

#### Parabolic SAR
- **PSAR**: ${psar:.2f} (Bull: ${psar_bull:.2f}, Bear: ${psar_bear:.2f})

#### Key Levels
- **Support/Resistance**: S1: ${s1:.2f}, R1: ${r1:.2f}, S2: ${s2:.2f}, R2: ${r2:.2f}
- **Fibonacci**: 23.6%: ${latest.get('Fib_236', price):.2f}, 38.2%: ${latest.get('Fib_382', price):.2f}, 61.8%: ${fib_618:.2f}

#### Fundamentals
{''.join([f'- {k}: {v:.2f}\n' for k, v in fundamental_data.items() if v is not None]) if fundamental_data and any(v is not None for v in fundamental_data.values()) else '- Not provided\n'}

#### Recommendation
- **Conservative Investors**: Wait for price to break above SMA20 (${sma_20:.2f}).
- **Traders**: {'Buy near support (${s1:.2f}) for bounce to ${r1:.2f}' if rsi < 30 or stoch_k < 20 else f'Wait for breakout above ${r1:.2f}'}.
- **Risk**: {'High' if latest['Volume'] < data_source['Volume'].mean() else 'Moderate'} due to {'low volume' if latest['Volume'] < data_source['Volume'].mean() else 'market volatility'}.
"""

    return quick_scan, moderate_detail, in_depth, stock_name, data_source, is_real_time_only, adx_value

# Main app
st.title("📈 Stock Technical Analysis Dashboard")
st.markdown("Analyze stocks with real-time data or uploaded CSV/XLSX files containing technical indicators.")

# Mode indicator
mode = "XLSX/CSV Only" if st.session_state.csv_data is not None and st.session_state.real_time_data is None else "Real-Time Only" if st.session_state.real_time_data is not None and st.session_state.csv_data is None else "Combined" if st.session_state.combine_report and st.session_state.csv_data is not None and st.session_state.real_time_data is not None else "No Data"
st.markdown(f"<div class='mode-banner'><b>Active Mode: {mode}</b><br>{'Historical data and indicators from uploaded CSV/XLSX.' if mode == 'XLSX/CSV Only' else 'Real-time price and fundamentals from yfinance.' if mode == 'Real-Time Only' else 'Combines CSV/XLSX historical data with real-time price/fundamentals.' if mode == 'Combined' else 'Please fetch data or upload a file to begin.'}</div>", unsafe_allow_html=True)

# Display real-time and fundamental data in expandable section
with st.expander("📊 Stock Data Overview", expanded=True):
    if st.session_state.real_time_data:
        current_data = {
            "Metric": ["Date", "Open", "High", "Low", "Close", "Volume"],
            "Value": [
                st.session_state.real_time_data['Date'],
                f"${st.session_state.real_time_data['Open']:.2f}",
                f"${st.session_state.real_time_data['High']:.2f}",
                f"${st.session_state.real_time_data['Low']:.2f}",
                f"${st.session_state.real_time_data['Close']:.2f}",
                f"{st.session_state.real_time_data['Volume']:,.0f}"
            ]
        }
        st.table(pd.DataFrame(current_data))

    if any(v is not None for v in st.session_state.fundamental_data.values()):
        fundamental_data = {
            "Metric": ["EPS", "P/B", "ROE", "Revenue", "Debt/Equity"],
            "Value": [
                f"{st.session_state.fundamental_data['EPS']:.2f}" if st.session_state.fundamental_data['EPS'] is not None else "N/A",
                f"{st.session_state.fundamental_data['P/B']:.2f}" if st.session_state.fundamental_data['P/B'] is not None else "N/A",
                f"{st.session_state.fundamental_data['ROE']:.2f}" if st.session_state.fundamental_data['ROE'] is not None else "N/A",
                f"${st.session_state.fundamental_data['Revenue']:.2f}M" if st.session_state.fundamental_data['Revenue'] is not None else "N/A",
                f"{st.session_state.fundamental_data['Debt/Equity']:.2f}" if st.session_state.fundamental_data['Debt/Equity'] is not None else "N/A"
            ]
        }
        st.table(pd.DataFrame(fundamental_data))

# Analyze data based on the last action
data_source = st.session_state.csv_data if process_file_button and st.session_state.csv_data is not None else st.session_state.real_time_data if submit_button and st.session_state.real_time_data is not None else combine_dataframes(st.session_state.csv_data, st.session_state.real_time_data) if combine_button and st.session_state.combine_report and st.session_state.csv_data is not None and st.session_state.real_time_data is not None else None

if data_source is not None:
    quick_scan, moderate_detail, in_depth, stock_name, df, is_real_time_only, adx_value = analyze_stock_data(
        st.session_state.csv_data if process_file_button and st.session_state.csv_data is not None else None,
        st.session_state.real_time_data if submit_button and st.session_state.real_time_data is not None else None,
        st.session_state.fundamental_data
    )

    if is_real_time_only and not all(col in df.columns for col in ['RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram']):
        st.warning("⚠️ Real-time data lacks historical indicators. Upload a CSV/XLSX for full analysis.")

    # Report tabs
    tabs = st.tabs(["Quick Scan", "Moderate Detail", "In-Depth Analysis", "Visual Summary", "Interactive Dashboard"])

    with tabs[0]:
        st.markdown("<div class='report-container'>", unsafe_allow_html=True)
        st.markdown(quick_scan)
        st.markdown("</div>", unsafe_allow_html=True)
        buffer = io.StringIO()
        buffer.write(quick_scan)
        st.download_button(
            label="📥 Download Markdown Report",
            data=buffer.getvalue(),
            file_name=f"{stock_name}_Quick_Scan_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )
        pdf_buffer = generate_pdf_report(quick_scan, stock_name, "Quick Scan")
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name=f"{stock_name}_Quick_Scan_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )

    with tabs[1]:
        st.markdown("<div class='report-container'>", unsafe_allow_html=True)
        st.markdown(moderate_detail)
        st.markdown("</div>", unsafe_allow_html=True)
        buffer = io.StringIO()
        buffer.write(moderate_detail)
        st.download_button(
            label="📥 Download Markdown Report",
            data=buffer.getvalue(),
            file_name=f"{stock_name}_Moderate_Detail_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )
        pdf_buffer = generate_pdf_report(moderate_detail, stock_name, "Moderate Detail")
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name=f"{stock_name}_Moderate_Detail_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )

    with tabs[2]:
        st.markdown("<div class='report-container'>", unsafe_allow_html=True)
        st.markdown(in_depth)
        st.markdown("</div>", unsafe_allow_html=True)
        buffer = io.StringIO()
        buffer.write(in_depth)
        st.download_button(
            label="📥 Download Markdown Report",
            data=buffer.getvalue(),
            file_name=f"{stock_name}_In_Depth_Analysis_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )
        pdf_buffer = generate_pdf_report(in_depth, stock_name, "In-Depth Analysis")
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name=f"{stock_name}_In_Depth_Analysis_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )

    with tabs[3]:
        st.markdown("<div class='report-container'>", unsafe_allow_html=True)
        date_str = (df['Date'].iloc[-1] if isinstance(df['Date'].iloc[-1], str) else df['Date'].iloc[-1].strftime('%Y-%m-%d') if pd.notna(df['Date'].iloc[-1]) else df.get('Date', 'N/A'))
        st.markdown(f"### 📊 Visual Summary: {stock_name} ({date_str})")
        st.write(f"**Price**: ${df['Close'].iloc[-1]:.2f}")
        st.write(f"**Trend**: {'Bearish' if df['Close'].iloc[-1] < df.get('SMA_20', df['Close']).iloc[-1] else 'Bullish'}")
        
        if not is_real_time_only and len(df) > 1:
            fig = px.line(df, x='Date', y=['Close', 'SMA_20', 'SMA_50', 'SMA_200'], title='Price Trend', hover_data=['Open', 'High', 'Low'])
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            fig_rsi = px.line(df, x='Date', y='RSI', title='RSI Trend')
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(hovermode='x unified')
            st.plotly_chart(fig_rsi, use_container_width=True)
        else:
            st.info("⚠️ Historical charts unavailable with real-time data only. Upload a CSV/XLSX.")

        rsi_value = df.get('RSI', 50) if not is_real_time_only else 50
        if isinstance(rsi_value, pd.Series):
            rsi_value = rsi_value.iloc[-1]
        st.markdown(f"- **Support**: ${df.get('S1', df['Close'] * 0.98).iloc[-1]:.2f}, **Resistance**: ${df.get('R1', df['Close'] * 1.02).iloc[-1]:.2f}")
        st.markdown(f"- **RSI**: {rsi_value:.2f} ({'Oversold' if rsi_value < 30 else 'Overbought' if rsi_value > 70 else 'Neutral'})")
        st.markdown(f"- **Recommendation**: {'Buy near support' if rsi_value < 30 else 'Wait for breakout'}")
        visual_summary = f"### Visual Summary: {stock_name} ({date_str})\n- **Price**: ${df['Close'].iloc[-1]:.2f}\n- **Trend**: {'Bearish' if df['Close'].iloc[-1] < df.get('SMA_20', df['Close']).iloc[-1] else 'Bullish'}\n- **Support**: ${df.get('S1', df['Close'] * 0.98).iloc[-1]:.2f}, **Resistance**: ${df.get('R1', df['Close'] * 1.02).iloc[-1]:.2f}\n- **RSI**: {rsi_value:.2f} ({'Oversold' if rsi_value < 30 else 'Overbought' if rsi_value > 70 else 'Neutral'})\n- **Recommendation**: {'Buy near support' if rsi_value < 30 else 'Wait for breakout'}"
        buffer = io.StringIO()
        buffer.write(visual_summary)
        st.download_button(
            label="📥 Download Markdown Report",
            data=buffer.getvalue(),
            file_name=f"{stock_name}_Visual_Summary_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )
        pdf_buffer = generate_pdf_report(visual_summary, stock_name, "Visual Summary")
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name=f"{stock_name}_Visual_Summary_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )

    with tabs[4]:
        st.markdown("<div class='report-container'>", unsafe_allow_html=True)
        st.markdown(f"### 📉 Interactive Dashboard: {stock_name}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📏 Technical Indicators")
            st.write(f"- **Price**: ${df['Close'].iloc[-1]:.2f}")
            rsi_value = df.get('RSI', 50) if not is_real_time_only else 50
            if isinstance(rsi_value, pd.Series):
                rsi_value = rsi_value.iloc[-1]
            st.write(f"- **RSI**: {rsi_value:.2f}")
            macd_value = df.get('MACD', 0) if not is_real_time_only else 0
            macd_signal_value = df.get('MACD_Signal', 0) if not is_real_time_only else 0
            if isinstance(macd_value, pd.Series):
                macd_value = macd_value.iloc[-1]
            if isinstance(macd_signal_value, pd.Series):
                macd_signal_value = macd_signal_value.iloc[-1]
            st.write(f"- **MACD**: {macd_value:.2f} (Signal: {macd_signal_value:.2f})")
            stoch_k_value = df.get('Stoch_K', 50) if not is_real_time_only else 50
            if isinstance(stoch_k_value, pd.Series):
                stoch_k_value = stoch_k_value.iloc[-1]
            st.write(f"- **Stochastic %K**: {stoch_k_value:.2f}")
            st.write(f"- **ADX**: {adx_value:.2f} ({'Strong Trend' if adx_value > 25 else 'Weak Trend'})")
            st.write(f"- **Support**: ${df.get('S1', df['Close'] * 0.98).iloc[-1]:.2f}")
            st.write(f"- **Resistance**: ${df.get('R1', df['Close'] * 1.02).iloc[-1]:.2f}")
        with col2:
            st.markdown("#### 📋 Fundamental Metrics")
            for k, v in st.session_state.fundamental_data.items():
                if v is not None:
                    st.write(f"- **{k}**: {v:.2f}")
        
        if not is_real_time_only and len(df) > 1:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Candlestick'))
            if 'BB_Upper' in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name='BB Upper', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name='BB Lower', line=dict(color='green')))
            fig.update_layout(title='Candlestick with Bollinger Bands', xaxis_title='Date', yaxis_title='Price', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⚠️ Candlestick chart unavailable with real-time data only. Upload a CSV/XLSX.")
        interactive_dashboard = f"### Interactive Dashboard: {stock_name}\n- **Price**: ${df['Close'].iloc[-1]:.2f}\n- **RSI**: {rsi_value:.2f}\n- **MACD**: {macd_value:.2f} (Signal: {macd_signal_value:.2f})\n- **Stochastic %K**: {stoch_k_value:.2f}\n- **ADX**: {adx_value:.2f}\n- **Support**: ${df.get('S1', df['Close'] * 0.98).iloc[-1]:.2f}\n- **Resistance**: ${df.get('R1', df['Close'] * 1.02).iloc[-1]:.2f}"
        buffer = io.StringIO()
        buffer.write(interactive_dashboard)
        st.download_button(
            label="📥 Download Markdown Report",
            data=buffer.getvalue(),
            file_name=f"{stock_name}_Interactive_Dashboard_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )
        pdf_buffer = generate_pdf_report(interactive_dashboard, stock_name, "Interactive Dashboard")
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer,
            file_name=f"{stock_name}_Interactive_Dashboard_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )

    if data_source is not None and isinstance(data_source, pd.DataFrame):
        export_df = data_source.copy()
        if st.session_state.fundamental_data and any(v is not None for v in st.session_state.fundamental_data.values()):
            for k, v in st.session_state.fundamental_data.items():
                export_df[k] = v
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Export Data as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"{stock_name}_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
else:
    st.info("⚠️ Please fetch real-time data or upload a CSV/XLSX to begin analysis.")
