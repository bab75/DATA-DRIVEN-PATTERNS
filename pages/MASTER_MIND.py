import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
from ta.trend import ADXIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Streamlit app configuration
st.set_page_config(page_title="Stock Technical Analysis", layout="wide", page_icon="📈")
try:
    st.markdown("""
        <style>
        body {background: linear-gradient(135deg, #f0f4f8, #d9e2ec); font-family: 'Segoe UI', sans-serif;}
        .stButton>button {background: linear-gradient(45deg, #4a90e2, #63b3ed); color: white; border: none; border-radius: 10px; padding: 10px 20px; transition: all 0.3s;}
        .stButton>button:hover {background: linear-gradient(45deg, #357abd, #4a90e2); transform: scale(1.05);}
        .stSelectbox, .stTextInput, .stNumberInput {background: #ffffff; border: 1px solid #d1d9e6; border-radius: 10px; padding: 5px;}
        .report-container {background: #ffffff; padding: 20px; border-radius: 15px; box-shadow: 0 8px 16px rgba(0,0,0,0.1); margin: 10px 0; text-align: center;}
        .data-card {background: #e8f4f8; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin: 10px 0; text-align: left;}
        h1 {color: #2c3e50; font-size: 2.5em; text-align: center; text-transform: uppercase; letter-spacing: 2px; background: #3498db; padding: 10px; border-radius: 10px; color: white;}
        h2, h3 {color: #34495e; font-weight: 500; text-align: center;}
        .mode-banner {background: #e6f3fa; padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #3498db;}
        .tab-content {padding: 20px; background: #f9f9f9; border-radius: 10px;}
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
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'report_content' not in st.session_state:
    st.session_state.report_content = {}

# Clear Analysis Button
def clear_analysis():
    st.session_state.real_time_data = None
    st.session_state.fundamental_data = {'EPS': None, 'P/E': None, 'PEG': None, 'P/B': None, 'ROE': None, 'Revenue': None, 'Debt/Equity': None}
    st.session_state.csv_data = None
    st.session_state.combine_report = False
    st.session_state.analysis_data = None
    st.session_state.report_content = {}
    st.success("Analysis cleared. Start a new analysis.")

# Sidebar for inputs
st.sidebar.header("📊 Stock Analysis Controls")
st.sidebar.subheader("📤 Upload Technical Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"])
process_file_button = st.sidebar.button("📥 Process File")

if process_file_button and uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            st.error(f"❌ Missing minimal columns: {', '.join(missing)}")
        else:
            df['Date'] = pd.to_datetime(df['Date'])
            st.session_state.csv_data = df
            st.success("✅ File processed successfully!")
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")

st.sidebar.subheader("📡 Real-Time Data")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., KRRO)", value="KRRO")
submit_button = st.sidebar.button("🔄 Fetch Real-Time Data")

if submit_button:
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d", interval="1d")
        if data.empty:
            st.error("❌ No data found for the ticker.")
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
                'EPS': info.get('trailingEps'),
                'P/E': info.get('trailingPE'),
                'PEG': info.get('pegRatio'),
                'P/B': info.get('priceToBook'),
                'ROE': info.get('returnOnEquity'),
                'Revenue': info.get('totalRevenue'),
                'Debt/Equity': info.get('debtToEquity')
            })
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")

st.sidebar.subheader("📈 Report Options")
combine_checkbox = st.sidebar.checkbox("Combine Report", value=st.session_state.combine_report)
if combine_checkbox != st.session_state.combine_report:
    st.session_state.combine_report = combine_checkbox
combine_button = st.sidebar.button("📊 Combine Process", disabled=not combine_checkbox)

with st.sidebar.expander("📋 Manual Fundamental Data (Optional)"):
    with st.form(key="fundamental_form"):
        eps = st.number_input("EPS", value=float(st.session_state.fundamental_data['EPS'] or 0.0), step=0.01, format="%.2f")
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

st.sidebar.subheader("🗑️ Reset")
clear_button = st.sidebar.button("Clear Analysis")
if clear_button:
    clear_analysis()

# Function to combine data
def combine_dataframes(csv_df, real_time_data):
    if csv_df is None or real_time_data is None:
        return csv_df if csv_df is not None else pd.DataFrame([real_time_data]) if real_time_data is not None else None
    try:
        real_time_df = pd.DataFrame([real_time_data])
        real_time_df['Date'] = pd.to_datetime(real_time_df['Date'])
        # Use only available columns
        common_cols = [col for col in csv_df.columns if col in real_time_df.columns]
        if not common_cols:
            st.error("❌ No common columns found between CSV and real-time data.")
            return None
        combined_df = pd.concat([csv_df[common_cols], real_time_df[common_cols]], ignore_index=True)
        combined_df = combined_df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in real_time_df.columns:
                combined_df[col].iloc[-1] = real_time_df[col].iloc[0]
        # Add missing technical columns with default values only if not present
        required_columns = ['Volatility', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50', 'SMA_200', 'EMA_200', 'BB_Width', 'BB_Position', 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_A', 'Ichimoku_Senkou_B', 'Ichimoku_Chikou', 'PSAR', 'PSAR_Bull', 'PSAR_Bear', 'Stoch_K', 'Williams_R', 'CCI', 'Momentum', 'ROC', 'ATR', 'Keltner_Upper', 'Keltner_Lower', 'OBV', 'VWAP', 'Volume_SMA', 'MFI', 'Pivot', 'R1', 'S1', 'R2', 'S2', 'Fib_236', 'Fib_382', 'Fib_618']
        for col in required_columns:
            if col not in combined_df.columns:
                combined_df[col] = 0
        st.write("Debug: Combined DataFrame:", combined_df.head())  # Debug output
        st.write("Debug: Combined Columns:", combined_df.columns.tolist())  # Debug column list
        return combined_df if not combined_df.empty else None
    except Exception as e:
        st.error(f"❌ Combine error: {str(e)}")
        return None

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
# Function to analyze stock data
def analyze_stock_data(df=None, real_time_data=None, fundamental_data=None):
    stock_name = ticker.upper()
    is_real_time_only = real_time_data is not None and df is None
    data_source = df if df is not None else pd.DataFrame([real_time_data]) if real_time_data is not None else None

    if data_source is None or not isinstance(data_source, pd.DataFrame) or data_source.empty:
        return "", "", "", stock_name, None, is_real_time_only, 50

    latest = data_source.iloc[-1] if not data_source.empty else None
    if latest is None:
        return "", "", "", stock_name, data_source, is_real_time_only, 50

    prev = data_source.iloc[-2] if len(data_source) > 1 else latest

    price = latest['Close']
    # Validate and calculate technical indicators dynamically
    sma_20 = price
    sma_50 = price
    sma_200 = price
    rsi = 50
    macd = macd_signal = macd_hist = 0
    bb_middle = bb_upper = bb_lower = price
    bb_width = bb_position = 0
    adx_value = 50
    atr = 0

    if len(data_source) > 1 and all(col in data_source.columns for col in ['High', 'Low', 'Close']):
        st.write("Debug: Close column:", data_source['Close'].head())  # Debug check
        if data_source['Close'].dtype in [int, float] and not data_source['Close'].isna().all():
            try:
                # SMA
                if len(data_source) >= 20:
                    sma_20 = SMAIndicator(close=data_source['Close'], window=20).sma().iloc[-1]
                if len(data_source) >= 50:
                    sma_50 = SMAIndicator(close=data_source['Close'], window=50).sma().iloc[-1]
                if len(data_source) >= 200:
                    sma_200 = SMAIndicator(close=data_source['Close'], window=200).sma().iloc[-1]
                # RSI
                if len(data_source) >= 14:
                    rsi = RSIIndicator(close=data_source['Close'], window=14).rsi().iloc[-1]
                # MACD
                if len(data_source) >= 26:
                    macd = MACD(close=data_source['Close']).macd().iloc[-1]
                    macd_signal = MACD(close=data_source['Close']).macd_signal().iloc[-1]
                    macd_hist = MACD(close=data_source['Close']).macd_diff().iloc[-1]
                # Bollinger Bands
                if len(data_source) >= 20:
                    bb = BollingerBands(close=data_source['Close'], window=20, window_dev=2)
                    bb_middle = bb.bollinger_mavg().iloc[-1]
                    bb_upper = bb.bollinger_hband().iloc[-1]
                    bb_lower = bb.bollinger_lband().iloc[-1]
                    bb_width = ((bb_upper - bb_lower) / bb_middle * 100) if bb_middle != 0 else 0
                    bb_position = ((price - bb_lower) / (bb_upper - bb_lower) * 100) if (bb_upper - bb_lower) != 0 else 0
                # ADX
                if len(data_source) >= 14:
                    adx_value = ADXIndicator(high=data_source['High'], low=data_source['Low'], close=data_source['Close'], window=14).adx().iloc[-1]
                # ATR (simplified)
                if len(data_source) >= 14:
                    atr = data_source['High'].rolling(window=14).max().diff().abs().rolling(window=14).mean().iloc[-1]
            except AttributeError as e:
                st.error(f"❌ Error calculating indicators: {str(e)}. Using default values.")
        else:
            st.error("❌ 'Close' column is not numeric or contains only NaN values.")

    volatility = atr * 100 / price if atr > 0 else 0  # Simplified volatility estimate
    trend = "Bearish" if price < sma_20 and price < sma_50 else "Bullish" if price > sma_20 and price > sma_50 else "Neutral"

    quick_scan = f"""
### Quick Scan: {stock_name} ({latest['Date'] if isinstance(latest['Date'], str) else latest['Date'].strftime('%Y-%m-%d')})
- **Price**: ${price:.2f} ({trend} trend)
- **Support/Resistance**: Support at ${price * 0.98:.2f}, Resistance at ${price * 1.02:.2f}
- **RSI**: {rsi:.2f} ({'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'})
- **Volatility**: {volatility:.2f}%
- **Recommendation**: {'Buy near support (${price * 0.98:.2f}) for bounce to ${price * 1.02:.2f}' if rsi < 30 else f'Wait for breakout above ${sma_20:.2f}'}
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
- **Key Levels**: Support: ${price * 0.98:.2f}, Resistance: ${price * 1.02:.2f}
- **Fundamentals**:
  {''.join([f'  - {k}: {v:.2f}\n' for k, v in fundamental_data.items() if v is not None]) if fundamental_data and any(v is not None for v in fundamental_data.values()) else '  - Not provided\n'}
- **Recommendation**:
  - Traders: {'Buy near ${price * 0.98:.2f} for bounce to ${price * 1.02:.2f}' if rsi < 30 else f'Wait for breakout above ${price * 1.02:.2f}'}
  - Investors: Confirm trend reversal above ${sma_20:.2f}.
"""

    in_depth = f"""
### In-Depth Analysis: {stock_name} ({latest['Date'] if isinstance(latest['Date'], str) else latest['Date'].strftime('%Y-%m-%d')})
#### Key Takeaways
- **Price**: ${price:.2f}, {trend} trend
- **ADX**: {adx_value:.2f} ({'Strong Trend' if adx_value > 25 else 'Weak Trend'})
- **Volatility**: {volatility:.2f}% (ATR: ${atr:.2f})

#### Price Trends
- **Close**: ${price:.2f}, {'below' if price < sma_20 else 'above'} SMA20 (${sma_20:.2f}), SMA50 (${sma_50:.2f}), SMA200 (${sma_200:.2f})
- **Trend**: {trend}

#### Momentum Indicators
- **RSI**: {rsi:.2f} ({'Oversold (<30)' if rsi < 30 else 'Overbought (>70)' if rsi > 70 else 'Neutral'})
- **MACD**: {macd:.2f}, Signal: {macd_signal:.2f}, Histogram: {macd_hist:.2f} ({'Bearish' if macd < macd_signal else 'Bullish'})

#### Volatility & Bands
- **Bollinger Bands**: Width: {bb_width:.2f}%, Position: {bb_position:.2f}% (Lower: ${bb_lower:.2f}, Middle: ${bb_middle:.2f}, Upper: ${bb_upper:.2f})

#### Volume
- **Volume**: {latest['Volume']:,.0f}

#### Fundamentals
{''.join([f'- {k}: {v:.2f}\n' for k, v in fundamental_data.items() if v is not None]) if fundamental_data and any(v is not None for v in fundamental_data.values()) else '- Not provided\n'}

#### Recommendation
- **Conservative Investors**: Wait for price to break above SMA20 (${sma_20:.2f}).
- **Traders**: {'Buy near support (${price * 0.98:.2f}) for bounce to ${price * 1.02:.2f}' if rsi < 30 else f'Wait for breakout above ${price * 1.02:.2f}'}.
"""

    return quick_scan, moderate_detail, in_depth, stock_name, data_source, is_real_time_only, adx_value

# Function to generate consolidated recommendation
def generate_consolidated_recommendation(quick_scan, moderate_detail, in_depth, stock_name, date_str):
    rsi_value = '50'
    if 'RSI:' in in_depth:
        rsi_part = in_depth.split('RSI: ')[1]
        if '(' in rsi_part:
            rsi_value = rsi_part.split(' (')[0].strip()
        else:
            rsi_value = rsi_part.split('\n')[0].strip()

    price = quick_scan.split('Price: $')[1].split(' (')[0] if 'Price:' in quick_scan else 'N/A'
    trend = moderate_detail.split('Price Trend: ')[1].split(',')[0] if 'Price Trend:' in moderate_detail else 'Neutral'
    support = quick_scan.split('Support at $')[1].split(',')[0] if 'Support at $' in quick_scan else 'N/A'
    resistance = quick_scan.split('Resistance at $')[1].split(')')[0] if 'Resistance at $' in quick_scan else 'N/A'

    recommendation = f"""
### Consolidated Recommendation: {stock_name} ({date_str})
#### Summary of Analysis
- **Price**: ${price}
- **Trend**: {trend}
- **RSI**: {rsi_value} ({'Oversold' if float(rsi_value) < 30 else 'Overbought' if float(rsi_value) > 70 else 'Neutral'})
- **Support**: ${support}
- **Resistance**: ${resistance}

#### Recommendation
- **Buy**: Recommended if RSI is oversold (<30) and price is near support (${support}), with a target near resistance (${resistance}). Current RSI is {rsi_value}, suggesting {'a buy opportunity' if float(rsi_value) < 30 else 'to wait for better conditions'}.
- **Hold**: Advised if price is between support and resistance with a neutral trend ({trend}) and RSI is neutral (30-70). Current trend is {trend}, supporting a {'hold' if 30 <= float(rsi_value) <= 70 else 'reconsideration'}.
- **Sell**: Suggested if RSI is overbought (>70) or price breaks below support (${support}). Current RSI is {rsi_value}, indicating {'a potential sell' if float(rsi_value) > 70 else 'no immediate sell signal'}.
"""

    return recommendation

# Download callback function
def download_report(report_type, content, stock_name):
    st.session_state.report_content[report_type] = content
    buffer = io.StringIO()
    buffer.write(content.replace(f'### {report_type}:', '').strip())
    b64 = base64.b64encode(buffer.getvalue().encode()).decode()
    href = f'<a href="data:text/markdown;base64,{b64}" download="{stock_name}_{report_type}_{datetime.now().strftime("%Y%m%d")}.md">📥 Download Markdown</a>'
    st.markdown(href, unsafe_allow_html=True)
    pdf_buffer = generate_pdf_report(content.replace(f'### {report_type}:', '').strip(), stock_name, report_type)
    b64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode()
    href_pdf = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{stock_name}_{report_type}_{datetime.now().strftime("%Y%m%d")}.pdf">📥 Download PDF</a>'
    st.markdown(href_pdf, unsafe_allow_html=True)

# Main app
st.title("📈 Stock Technical Analysis Dashboard")
st.markdown("Analyze stocks with real-time data or uploaded CSV/XLSX files containing technical indicators.")

mode = "XLSX/CSV Only" if st.session_state.csv_data is not None and st.session_state.real_time_data is None else "Real-Time Only" if st.session_state.real_time_data is not None and st.session_state.csv_data is None else "Combined" if st.session_state.combine_report and st.session_state.csv_data is not None and st.session_state.real_time_data is not None else "No Data"
st.markdown(f"<div class='mode-banner'><b>Active Mode: {mode}</b><br>{'Historical data from CSV/XLSX.' if mode == 'XLSX/CSV Only' else 'Real-time data from yfinance.' if mode == 'Real-Time Only' else 'Combined data.' if mode == 'Combined' else 'Please provide data.'}</div>", unsafe_allow_html=True)

st.subheader("📊 Stock Data Overview")
if st.session_state.real_time_data:
    date_str = st.session_state.real_time_data['Date']
    st.markdown(f"### Stock Price as on {date_str}")
    st.markdown(f"<div class='data-card'><b>Price:</b> ${st.session_state.real_time_data['Close']:.2f}<br><b>Volume:</b> {st.session_state.real_time_data['Volume']:,.0f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='data-card'><b>Open:</b> ${st.session_state.real_time_data['Open']:.2f}<br><b>High:</b> ${st.session_state.real_time_data['High']:.2f}<br><b>Low:</b> ${st.session_state.real_time_data['Low']:.2f}</div>", unsafe_allow_html=True)

if any(v is not None for v in st.session_state.fundamental_data.values()):
    with st.expander("### Fundamental Analysis"):
        st.markdown("<div class='data-card'>", unsafe_allow_html=True)
        for k, v in st.session_state.fundamental_data.items():
            if v is not None:
                st.markdown(f"<b>{k}:</b> {v:.2f}<br>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Data source determination
data_source = st.session_state.csv_data if process_file_button and st.session_state.csv_data is not None else st.session_state.real_time_data if submit_button and st.session_state.real_time_data is not None else combine_dataframes(st.session_state.csv_data, st.session_state.real_time_data) if combine_button and st.session_state.combine_report and st.session_state.csv_data is not None and st.session_state.real_time_data is not None else None
st.write("Debug: data_source:", data_source)  # Debug output

# Analyze data
if data_source is not None and isinstance(data_source, pd.DataFrame) and not data_source.empty:
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_columns if col not in data_source.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns in data_source: {', '.join(missing_cols)}")
        st.session_state.analysis_data = None
    else:
        quick_scan, moderate_detail, in_depth, stock_name, df, is_real_time_only, adx_value = analyze_stock_data(data_source, None if process_file_button or submit_button else st.session_state.real_time_data, st.session_state.fundamental_data)
        st.session_state.analysis_data = {
            'quick_scan': quick_scan,
            'moderate_detail': moderate_detail,
            'in_depth': in_depth,
            'stock_name': stock_name,
            'df': df,
            'is_real_time_only': is_real_time_only,
            'adx_value': adx_value
        }
else:
    st.session_state.analysis_data = None

# Render tabs only if analysis data exists
if st.session_state.analysis_data is not None:
    quick_scan = st.session_state.analysis_data['quick_scan']
    moderate_detail = st.session_state.analysis_data['moderate_detail']
    in_depth = st.session_state.analysis_data['in_depth']
    stock_name = st.session_state.analysis_data['stock_name']
    df = st.session_state.analysis_data['df']
    is_real_time_only = st.session_state.analysis_data['is_real_time_only']
    adx_value = st.session_state.analysis_data['adx_value']

    st.write("Debug: df:", df)  # Debug output
    if isinstance(df, pd.DataFrame) and not df.empty and 'Date' in df.columns:
        tabs = st.tabs(["Quick Scan", "Moderate Detail", "In-Depth Analysis", "Visual Summary", "Interactive Dashboard", "Consolidated Recommendation"])

        with tabs[0]:
            st.markdown("<div class='report-container'><div class='tab-content'>", unsafe_allow_html=True)
            st.markdown(f"### Quick Scan: {stock_name}")  # Use Markdown header
            st.markdown(quick_scan.replace('### Quick Scan:', '').strip())  # Strip header
            col1, col2 = st.columns(2)
            with col1:
                download_report("Quick Scan", quick_scan, stock_name)
            with col2:
                pdf_buffer = generate_pdf_report(quick_scan.replace('### Quick Scan:', '').strip(), stock_name, "Quick Scan")
                b64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode()
                href_pdf = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{stock_name}_Quick_Scan_{datetime.now().strftime("%Y%m%d")}.pdf">📥 Download PDF</a>'
                st.markdown(href_pdf, unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

        with tabs[1]:
            st.markdown("<div class='report-container'><div class='tab-content'>", unsafe_allow_html=True)
            st.markdown(f"### Moderate Detail: {stock_name}")  # Use Markdown header
            st.markdown(moderate_detail.replace('### Moderate Detail:', '').strip())  # Strip header
            col1, col2 = st.columns(2)
            with col1:
                download_report("Moderate Detail", moderate_detail, stock_name)
            with col2:
                pdf_buffer = generate_pdf_report(moderate_detail.replace('### Moderate Detail:', '').strip(), stock_name, "Moderate Detail")
                b64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode()
                href_pdf = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{stock_name}_Moderate_Detail_{datetime.now().strftime("%Y%m%d")}.pdf">📥 Download PDF</a>'
                st.markdown(href_pdf, unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

        with tabs[2]:
            st.markdown("<div class='report-container'><div class='tab-content'>", unsafe_allow_html=True)
            st.markdown(f"### In-Depth Analysis: {stock_name}")  # Use Markdown header
            st.markdown(in_depth.replace('### In-Depth Analysis:', '').strip())  # Strip header
            col1, col2 = st.columns(2)
            with col1:
                download_report("In-Depth Analysis", in_depth, stock_name)
            with col2:
                pdf_buffer = generate_pdf_report(in_depth.replace('### In-Depth Analysis:', '').strip(), stock_name, "In-Depth Analysis")
                b64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode()
                href_pdf = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{stock_name}_In_Depth_Analysis_{datetime.now().strftime("%Y%m%d")}.pdf">📥 Download PDF</a>'
                st.markdown(href_pdf, unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

        with tabs[5]:
            st.markdown("<div class='report-container'><div class='tab-content'>", unsafe_allow_html=True)
            st.markdown(f"### Consolidated Recommendation: {stock_name}")  # Use Markdown header
            date_str = df['Date'].iloc[-1].strftime('%Y-%m-%d') if pd.notna(df['Date'].iloc[-1]) else datetime.now().strftime('%Y-%m-%d')
            recommendation = generate_consolidated_recommendation(quick_scan, moderate_detail, in_depth, stock_name, date_str)
            st.markdown(recommendation.replace('### Consolidated Recommendation:', '').strip())  # Strip header
            col1, col2 = st.columns(2)
            with col1:
                download_report("Consolidated Recommendation", recommendation, stock_name)
            with col2:
                pdf_buffer = generate_pdf_report(recommendation.replace('### Consolidated Recommendation:', '').strip(), stock_name, "Consolidated Recommendation")
                b64_pdf = base64.b64encode(pdf_buffer.getvalue()).decode()
                href_pdf = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{stock_name}_Consolidated_Recommendation_{datetime.now().strftime("%Y%m%d")}.pdf">📥 Download PDF</a>'
                st.markdown(href_pdf, unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

        if isinstance(data_source, pd.DataFrame) and not data_source.empty:
            export_df = data_source.copy()
            if st.session_state.fundamental_data and any(v is not None for v in st.session_state.fundamental_data.values()):
                for k, v in st.session_state.fundamental_data.items():
                    export_df[k] = v
            csv_buffer = io.StringIO()
            export_df.to_csv(csv_buffer, index=False)
            st.download_button("📥 Export Data as CSV", csv_buffer.getvalue(), f"{stock_name}_data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
else:
    st.info("⚠️ Please upload a CSV/XLSX or fetch real-time data to begin analysis.")
