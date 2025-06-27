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

# Clear Analysis Button
def clear_analysis():
    st.session_state.real_time_data = None
    st.session_state.fundamental_data = {'EPS': None, 'P/E': None, 'PEG': None, 'P/B': None, 'ROE': None, 'Revenue': None, 'Debt/Equity': None}
    st.session_state.csv_data = None
    st.session_state.combine_report = False
    st.session_state.analysis_data = None
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
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Volatility', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50', 'SMA_200', 'EMA_200', 'BB_Width', 'BB_Position', 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_A', 'Ichimoku_Senkou_B', 'Ichimoku_Chikou', 'PSAR', 'PSAR_Bull', 'PSAR_Bear', 'Stoch_K', 'Williams_R', 'CCI', 'Momentum', 'ROC', 'ATR', 'Keltner_Upper', 'Keltner_Lower', 'OBV', 'VWAP', 'Volume_SMA', 'MFI', 'Pivot', 'R1', 'S1', 'R2', 'S2', 'Fib_236', 'Fib_382', 'Fib_618']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            st.error(f"❌ Missing columns: {', '.join(missing)}")
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

# Function to handle data selection (no merging)
def combine_dataframes(csv_df, real_time_data):
    """
    Don't combine dataframes. Use historical CSV for analysis, real-time for current price only.
    """
    if csv_df is None and real_time_data is None:
        return None
    
    # If only real-time data exists, return basic dataframe for display
    if csv_df is None and real_time_data is not None:
        return pd.DataFrame([real_time_data])
    
    # If CSV exists, use it as the primary analysis data
    # Real-time data will be used separately for current price context
    return csv_df

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
    
    # Use historical data for technical analysis
    analysis_data = df  # This contains all technical indicators
    
    # Use real-time data only for current price context
    current_price = real_time_data['Close'] if real_time_data else None
    current_date = real_time_data['Date'] if real_time_data else datetime.now().strftime('%Y-%m-%d')
    
    # Perform analysis using historical data
    if analysis_data is not None and not analysis_data.empty:
        latest_historical = analysis_data.iloc[-1]
        prev_historical = analysis_data.iloc[-2] if len(analysis_data) > 1 else latest_historical
        price = current_price if current_price else latest_historical['Close']
        volatility = analysis_data['Volatility'].mean()
        rsi = latest_historical['RSI']
        macd = latest_historical['MACD']
        macd_signal = latest_historical['MACD_Signal']
        macd_hist = latest_historical['MACD_Histogram']
        bb_upper = latest_historical['BB_Upper']
        bb_middle = latest_historical['BB_Middle']
        bb_lower = latest_historical['BB_Lower']
        bb_width = analysis_data['BB_Width'].mean()
        bb_position = analysis_data['BB_Position'].mean()
        sma_20 = latest_historical['SMA_20']
        ema_20 = latest_historical['EMA_20']
        sma_50 = latest_historical['SMA_50']
        ema_50 = latest_historical['EMA_50']
        sma_200 = latest_historical['SMA_200']
        ema_200 = latest_historical['EMA_200']
        ichimoku_tenkan = latest_historical['Ichimoku_Tenkan']
        ichimoku_kijun = latest_historical['Ichimoku_Kijun']
        ichimoku_senkou_a = latest_historical['Ichimoku_Senkou_A']
        ichimoku_senkou_b = latest_historical['Ichimoku_Senkou_B']
        ichimoku_chikou = latest_historical['Ichimoku_Chikou']
        psar = latest_historical['PSAR']
        psar_bull = latest_historical['PSAR_Bull']
        psar_bear = latest_historical['PSAR_Bear']
        stoch_k = latest_historical['Stoch_K']
        williams_r = latest_historical['Williams_R']
        cci = latest_historical['CCI']
        momentum = latest_historical['Momentum']
        roc = latest_historical['ROC']
        atr = analysis_data['ATR'].mean()
        keltner_upper = latest_historical['Keltner_Upper']
        keltner_lower = latest_historical['Keltner_Lower']
        obv = latest_historical['OBV']
        vwap = latest_historical['VWAP']
        volume_sma = latest_historical['Volume_SMA']
        mfi = latest_historical['MFI']
        pivot = latest_historical['Pivot']
        r1 = latest_historical['R1']
        s1 = latest_historical['S1']
        r2 = latest_historical['R2']
        s2 = latest_historical['S2']
        fib_618 = latest_historical['Fib_618']
        
        adx_value = 50
        if all(col in analysis_data.columns for col in ['High', 'Low', 'Close']) and len(analysis_data) >= 14:
            adx_indicator = ADXIndicator(analysis_data['High'], analysis_data['Low'], analysis_data['Close'], window=14)
            adx_value = adx_indicator.adx().iloc[-1]

        trend_pattern = "Neutral"
        if len(analysis_data) > 10:
            highs = analysis_data['High'].rolling(window=10).max()
            lows = analysis_data['Low'].rolling(window=10).min()
            if highs.iloc[-1] > highs.iloc[-2] and lows.iloc[-1] > lows.iloc[-2]:
                trend_pattern = "Higher Highs & Lows (Bullish)"
            elif highs.iloc[-1] < highs.iloc[-2] and lows.iloc[-1] < lows.iloc[-2]:
                trend_pattern = "Lower Highs & Lows (Bearish)"

        trend = "Bearish" if price < sma_20 and price < sma_50 else "Bullish" if price > sma_20 and price > sma_50 else "Neutral"
    else:
        # If no historical data, limited analysis with real-time only
        price = current_price if current_price else 0
        volatility = 0
        rsi = 50
        macd = 0
        macd_signal = 0
        macd_hist = 0
        bb_upper = price * 1.05
        bb_middle = price
        bb_lower = price * 0.95
        bb_width = 0
        bb_position = 0
        sma_20 = price
        ema_20 = price
        sma_50 = price
        ema_50 = price
        sma_200 = price
        ema_200 = price
        ichimoku_tenkan = price
        ichimoku_kijun = price
        ichimoku_senkou_a = price
        ichimoku_senkou_b = price
        ichimoku_chikou = price
        psar = price
        psar_bull = price
        psar_bear = price
        stoch_k = 50
        williams_r = -50
        cci = 0
        momentum = 0
        roc = 0
        atr = 0
        keltner_upper = price * 1.05
        keltner_lower = price * 0.95
        obv = real_time_data['Volume'] if real_time_data else 0
        vwap = price
        volume_sma = real_time_data['Volume'] if real_time_data else 0
        mfi = 50
        pivot = price
        r1 = price * 1.02
        s1 = price * 0.98
        r2 = price * 1.04
        s2 = price * 0.96
        fib_618 = price * 1.01
        adx_value = 50
        trend_pattern = "Neutral"
        trend = "Neutral"
        latest_historical = None
        prev_historical = None

    quick_scan = f"""
### Quick Scan: {stock_name} ({current_date})
- **Price**: ${price:.2f} ({trend} trend)
- **Support/Resistance**: Support at ${s1:.2f}, Resistance at ${r1:.2f}
- **RSI**: {rsi:.2f} ({'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'})
- **Volatility**: {volatility:.2f}%
- **Recommendation**: {'Buy near support (${s1:.2f}) for bounce to ${r1:.2f}' if rsi < 30 else f'Wait for breakout above ${sma_20:.2f}'}
"""

    moderate_detail = f"""
### Moderate Detail: {stock_name} ({current_date})
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
### In-Depth Analysis: {stock_name} ({current_date})
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
- **OBV**: {obv:,.0f} ({'Declining' if obv < prev_historical['OBV'] if prev_historical is not None else obv else 'Stable'})
- **Volume**: {real_time_data['Volume'] if real_time_data else latest_historical['Volume']:,.0f} (SMA: {volume_sma:,.0f})
- **VWAP**: ${vwap:.2f}

#### Ichimoku Cloud
- **Tenkan**: ${ichimoku_tenkan:.2f}, Kijun: ${ichimoku_kijun:.2f}
- **Senkou A**: ${ichimoku_senkou_a:.2f}, Senkou B: ${ichimoku_senkou_b:.2f}
- **Chikou**: ${ichimoku_chikou:.2f}

#### Parabolic SAR
- **PSAR**: ${psar:.2f} (Bull: ${psar_bull:.2f}, Bear: ${psar_bear:.2f})

#### Key Levels
- **Support/Resistance**: S1: ${s1:.2f}, R1: ${r1:.2f}, S2: ${s2:.2f}, R2: ${r2:.2f}
- **Fibonacci**: 23.6%: ${latest_historical.get('Fib_236', price):.2f if latest_historical is not None else price:.2f}, 38.2%: ${latest_historical.get('Fib_382', price):.2f if latest_historical is not None else price:.2f}, 61.8%: ${fib_618:.2f}

#### Fundamentals
{''.join([f'- {k}: {v:.2f}\n' for k, v in fundamental_data.items() if v is not None]) if fundamental_data and any(v is not None for v in fundamental_data.values()) else '- Not provided\n'}

#### Recommendation
- **Conservative Investors**: Wait for price to break above SMA20 (${sma_20:.2f}).
- **Traders**: {'Buy near support (${s1:.2f}) for bounce to ${r1:.2f}' if rsi < 30 or stoch_k < 20 else f'Wait for breakout above ${r1:.2f}'}.
- **Risk**: {'High' if (real_time_data['Volume'] if real_time_data else latest_historical['Volume'] if latest_historical is not None else 0) < (analysis_data['Volume'].mean() if analysis_data is not None else 0) else 'Moderate'} due to {'low volume' if (real_time_data['Volume'] if real_time_data else latest_historical['Volume'] if latest_historical is not None else 0) < (analysis_data['Volume'].mean() if analysis_data is not None else 0) else 'market volatility'}.
"""

    return quick_scan, moderate_detail, in_depth, stock_name, analysis_data, real_time_data is not None and analysis_data is None, adx_value

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
    volume_trend = in_depth.split('OBV: ')[1].split(' (')[1].split(')')[0] if 'OBV:' in in_depth else 'N/A'
    adx = in_depth.split('ADX: ')[1].split(' (')[0] if 'ADX:' in in_depth else '50'

    recommendation = f"""
### Consolidated Recommendation: {stock_name} ({date_str})
#### Summary of Analysis
- **Price**: ${price}
- **Trend**: {trend}
- **RSI**: {rsi_value} ({'Oversold' if float(rsi_value) < 30 else 'Overbought' if float(rsi_value) > 70 else 'Neutral'})
- **Support**: ${support}
- **Resistance**: ${resistance}
- **Volume Trend**: {volume_trend}

#### Recommendation
- **Buy**: Recommended if RSI is oversold (<30) and price is near support (${support}), with a target near resistance (${resistance}). Current RSI is {rsi_value}, suggesting {'a buy opportunity' if float(rsi_value) < 30 else 'to wait for better conditions'}.
- **Hold**: Advised if price is between support and resistance with a neutral trend ({trend}) and RSI is neutral (30-70). Current trend is {trend}, supporting a {'hold' if 30 <= float(rsi_value) <= 70 else 'reconsideration'}.
- **Sell**: Suggested if RSI is overbought (>70) or price breaks below support (${support}) with declining volume. Current RSI is {rsi_value}, indicating {'a potential sell' if float(rsi_value) > 70 else 'no immediate sell signal'}.
- **Additional Notes**: Volume trend ({volume_trend}) and ADX ({adx} {'Strong Trend' if float(adx) > 25 else 'Weak Trend'}) should be monitored for confirmation.
"""

    return recommendation

# Main app
st.title("📈 Stock Technical Analysis Dashboard")
st.markdown("Analyze stocks with real-time data or uploaded CSV/XLSX files containing technical indicators.")

mode = "XLSX/CSV Only" if st.session_state.csv_data is not None and st.session_state.real_time_data is None else "Real-Time Only" if st.session_state.real_time_data is not None and st.session_state.csv_data is None else "Combined" if st.session_state.combine_report and st.session_state.csv_data is not None and st.session_state.real_time_data is not None else "No Data"
st.markdown(f"<div class='mode-banner'><b>Active Mode: {mode}</b><br>{'Historical data from CSV/XLSX.' if mode == 'XLSX/CSV Only' else 'Real-time data from yfinance.' if mode == 'Real-Time Only' else 'Historical data with real-time price context.' if mode == 'Combined' else 'Please provide data.'}</div>", unsafe_allow_html=True)

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

# Data source determination and analysis
if process_file_button and st.session_state.csv_data is not None:
    analysis_df = st.session_state.csv_data
    st.session_state.analysis_data = analyze_stock_data(
        df=analysis_df,
        real_time_data=st.session_state.real_time_data,
        fundamental_data=st.session_state.fundamental_data
    )
elif submit_button and st.session_state.real_time_data is not None and st.session_state.csv_data is None:
    analysis_df = pd.DataFrame([st.session_state.real_time_data])
    st.session_state.analysis_data = analyze_stock_data(
        df=None,
        real_time_data=st.session_state.real_time_data,
        fundamental_data=st.session_state.fundamental_data
    )
elif combine_button and st.session_state.combine_report and st.session_state.csv_data is not None:
    analysis_df = st.session_state.csv_data
    st.session_state.analysis_data = analyze_stock_data(
        df=analysis_df,
        real_time_data=st.session_state.real_time_data,
        fundamental_data=st.session_state.fundamental_data
    )
else:
    st.session_state.analysis_data = None

# Render tabs only if analysis data exists
if st.session_state.analysis_data is not None:
    quick_scan, moderate_detail, in_depth, stock_name, df, is_real_time_only, adx_value = st.session_state.analysis_data

    if isinstance(df, pd.DataFrame) and not df.empty and 'Date' in df.columns:
        tabs = st.tabs(["Quick Scan", "Moderate Detail", "In-Depth Analysis", "Visual Summary", "Interactive Dashboard", "Consolidated Recommendation"])

        with tabs[0]:
            st.markdown("<div class='report-container'><div class='tab-content'>", unsafe_allow_html=True)
            st.markdown(f"### Quick Scan: {stock_name}")
            st.markdown(quick_scan.replace('### Quick Scan:', '').strip())
            col1, col2 = st.columns(2)
            with col1:
                buffer = io.StringIO()
                buffer.write(quick_scan.replace('### Quick Scan:', '').strip())
                st.download_button("📥 funkcDownload Markdown", buffer.getvalue(), f"{stock_name}_Quick_Scan_{datetime.now().strftime('%Y%m%d')}.md", "text/markdown")
            with col2:
                pdf_buffer = generate_pdf_report(quick_scan.replace('### Quick Scan:', '').strip(), stock_name, "Quick Scan")
                st.download_button("📥 Download PDF", pdf_buffer, f"{stock_name}_Quick_Scan_{datetime.now().strftime('%Y%m%d')}.pdf", "application/pdf")
            st.markdown("</div></div>", unsafe_allow_html=True)

        with tabs[1]:
            st.markdown("<div class='report-container'><div class='tab-content'>", unsafe_allow_html=True)
            st.markdown(f"### Moderate Detail: {stock_name}")
            st.markdown(moderate_detail.replace('### Moderate Detail:', '').strip())
            col1, col2 = st.columns(2)
            with col1:
                buffer代替buffer = io.StringIO()
                buffer.write(moderate_detail.replace('### Moderate Detail:', '').strip())
                st.download_button("📥 Download Markdown", buffer.getvalue(), f"{stock_name}_Moderate_Detail_{datetime.now().strftime('%Y%m%d')}.md", "text/markdown")
            with col2:
                pdf_buffer = generate_pdf_report(moderate_detail.replace('### Moderate Detail:', '').strip(), stock_name, "Moderate Detail")
                st.download_button("📥 Download PDF", pdf_buffer, f"{stock_name}_Moderate_Detail_{datetime.now().strftime('%Y%m%d')}.pdf", "application/pdf")
            st.markdown("</div></div>", unsafe_allow_html=True)

        with tabs[2]:
            st.markdown("<div class='report-container'><div class='tab-content'>", unsafe_allow_html=True)
            st.markdown(f"### In-Depth Analysis: {stock_name}")
            st.markdown(in_depth.replace('### In-Depth Analysis:', '').strip())
            col1, col2 = st.columns(2)
            with col1:
                buffer = io.StringIO()
                buffer.write(in_depth.replace('### In-Depth Analysis:', '').strip())
                st.download_button("📥 Download Markdown", buffer.getvalue(), f"{stock_name}_In_Depth_Analysis_{datetime.now().strftime('%Y%m%d')}.md", "text/markdown")
            with col2:
                pdf_buffer = generate_pdf_report(in_depth.replace('### In-Depth Analysis:', '').strip(), stock_name, "In-Depth Analysis")
                st.download_button("📥 Download PDF", pdf_buffer, f"{stock_name}_In_Depth_Analysis_{datetime.now().strftime('%Y%m%d')}.pdf", "application/pdf")
            st.markdown("</div></div>", unsafe_allow_html=True)

        with tabs[3]:
            st.markdown("<div class='report-container'><div class='tab-content'>", unsafe_allow_html=True)
            date_str = df['Date'].iloc[-1].strftime('%Y-%m-%d') if pd.notna(df['Date'].iloc[-1]) else datetime.now().strftime('%Y-%m-%d')
            st.markdown(f"### Visual Summary: {stock_name} ({date_str})")
            st.write(f"**Price**: ${st.session_state.real_time_data['Close'] if st.session_state.real_time_data else df['Close'].iloc[-1]:.2f}")
            st.write(f"**Trend**: {'Bearish' if (st.session_state.real_time_data['Close'] if st.session_state.real_time_data else df['Close'].iloc[-1]) < df['SMA_20'].iloc[-1] else 'Bullish'}")
            if not is_real_time_only and len(df) > 1:
                fig = px.line(df, x='Date', y=['Close', 'SMA_20', 'SMA_50', 'SMA_200'], title='Price Trend')
                st.plotly_chart(fig)
                fig_rsi = px.line(df, x='Date', y='RSI', title='RSI Trend')
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                st.plotly_chart(fig_rsi)
            rsi_value = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
            st.markdown(f"- **Support**: ${df['S1'].iloc[-1]:.2f}, **Resistance**: ${df['R1'].iloc[-1]:.2f}")
            st.markdown(f"- **RSI**: {rsi_value:.2f} ({'Oversold' if rsi_value < 30 else 'Overbought' if rsi_value > 70 else 'Neutral'})")
            st.markdown(f"- **Recommendation**: {'Buy near support' if rsi_value < 30 else 'Wait for breakout'}")
            visual_summary = f"### Visual Summary: {stock_name} ({date_str})\n- **Price**: ${st.session_state.real_time_data['Close'] if st.session_state.real_time_data else df['Close'].iloc[-1]:.2f}\n- **Trend**: {'Bearish' if (st.session_state.real_time_data['Close'] if st.session_state.real_time_data else df['Close'].iloc[-1]) < df['SMA_20'].iloc[-1] else 'Bullish'}\n- **Support**: ${df['S1'].iloc[-1]:.2f}, **Resistance**: ${df['R1'].iloc[-1]:.2f}\n- **RSI**: {rsi_value:.2f} ({'Oversold' if rsi_value < 30 else 'Overbought' if rsi_value > 70 else 'Neutral'})\n- **Recommendation**: {'Buy near support' if rsi_value < 30 else 'Wait for breakout'}"
            col1, col2 = st.columns(2)
            with col1:
                buffer = io.StringIO()
                buffer.write(visual_summary.replace('### Visual Summary:', '').strip())
                st.download_button("📥 Download Markdown", buffer.getvalue(), f"{stock_name}_Visual_Summary_{datetime.now().strftime('%Y%m%d')}.md", "text/markdown")
            with col2:
                pdf_buffer = generate_pdf_report(visual_summary.replace('### Visual Summary:', '').strip(), stock_name, "Visual Summary")
                st.download_button("📥 Download PDF", pdf_buffer, f"{stock_name}_Visual_Summary_{datetime.now().strftime('%Y%m%d')}.pdf", "application/pdf")
            st.markdown("</div></div>", unsafe_allow_html=True)

        with tabs[4]:
            st.markdown("<div class='report-container'><div class='tab-content'>", unsafe_allow_html=True)
            st.markdown(f"### Interactive Dashboard: {stock_name}")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📏 Technical Indicators")
                st.write(f"- **Price**: ${st.session_state.real_time_data['Close'] if st.session_state.real_time_data else df['Close'].iloc[-1]:.2f}")
                rsi_value = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
                st.write(f"- **RSI**: {rsi_value:.2f}")
                macd_value = df['MACD'].iloc[-1] if 'MACD' in df.columns else 0
                macd_signal_value = df['MACD_Signal'].iloc[-1] if 'MACD_Signal' in df.columns else 0
                st.write(f"- **MACD**: {macd_value:.2f} (Signal: {macd_signal_value:.2f})")
                stoch_k_value = df['Stoch_K'].iloc[-1] if 'Stoch_K' in df.columns else 50
                st.write(f"- **Stochastic %K**: {stoch_k_value:.2f}")
                st.write(f"- **ADX**: {adx_value:.2f} ({'Strong Trend' if adx_value > 25 else 'Weak Trend'})")
                st.write(f"- **Support**: ${df['S1'].iloc[-1]:.2f}")
                st.write(f"- **Resistance**: ${df['R1'].iloc[-1]:.2f}")
            with col2:
                st.markdown("#### 📋 Fundamental Metrics")
                for k, v in st.session_state.fundamental_data.items():
                    if v is not None:
                        st.write(f"- **{k}**: {v:.2f}")
            if not is_real_time_only and len(df) > 1:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
                if 'BB_Upper' in df.columns:
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name='BB Upper', line=dict(color='red')))
                    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name='BB Lower', line=dict(color='green')))
                st.plotly_chart(fig)
            interactive_dashboard = f"### Interactive Dashboard: {stock_name}\n- **Price**: ${st.session_state.real_time_data['Close'] if st.session_state.real_time_data else df['Close'].iloc[-1]:.2f}\n- **RSI**: {rsi_value:.2f}\n- **MACD**: {macd_value:.2f} (Signal: {macd_signal_value:.2f})\n- **Stochastic %K**: {stoch_k_value:.2f}\n- **ADX**: {adx_value:.2f}\n- **Support**: ${df['S1'].iloc[-1]:.2f}\n- **Resistance**: ${df['R1'].iloc[-1]:.2f}"
            col1, col2 = st.columns(2)
            with col1:
                buffer = io.StringIO()
                buffer.write(interactive_dashboard.replace('### Interactive Dashboard:', '').strip())
                st.download_button("📥 Download Markdown", buffer.getvalue(), f"{stock_name}_Interactive_Dashboard_{datetime.now().strftime('%Y%m%d')}.md", "text/markdown")
            with col2:
                pdf_buffer = generate_pdf_report(interactive_dashboard.replace('### Interactive Dashboard:', '').strip(), stock_name, "Interactive Dashboard")
                st.download_button("📥 Download PDF", pdf_buffer, f"{stock_name}_Interactive_Dashboard_{datetime.now().strftime('%Y%m%d')}.pdf", "application/pdf")
            st.markdown("</div></div>", unsafe_allow_html=True)

        with tabs[5]:
            st.markdown("<div class='report-container'><div class='tab-content'>", unsafe_allow_html=True)
            st.markdown(f"### Consolidated Recommendation: {stock_name}")
            date_str = df['Date'].iloc[-1].strftime('%Y-%m-%d') if pd.notna(df['Date'].iloc[-1]) else datetime.now().strftime('%Y-%m-%d')
            recommendation = generate_consolidated_recommendation(quick_scan, moderate_detail, in_depth, stock_name, date_str)
            st.markdown(recommendation.replace('### Consolidated Recommendation:', '').strip())
            col1, col2 = st.columns(2)
            with col1:
                buffer = io.StringIO()
                buffer.write(recommendation.replace('### Consolidated Recommendation:', '').strip())
                st.download_button("📥 Download Markdown", buffer.getvalue(), f"{stock_name}_Consolidated_Recommendation_{datetime.now().strftime('%Y%m%d')}.md", "text/markdown")
            with col2:
                pdf_buffer = generate_pdf_report(recommendation.replace('### Consolidated Recommendation:', '').strip(), stock_name, "Consolidated Recommendation")
                st.download_button("📥 Download PDF", pdf_buffer, f"{stock_name}_Consolidated_Recommendation_{datetime.now().strftime('%Y%m%d')}.pdf", "application/pdf")
            st.markdown("</div></div>", unsafe_allow_html=True)

        if isinstance(df, pd.DataFrame) and not df.empty:
            export_df = df.copy()
            if st.session_state.fundamental_data and any(v is not None for v in st.session_state.fundamental_data.values()):
                for k, v in st.session_state.fundamental_data.items():
                    export_df[k] = v
            csv_buffer = io.StringIO()
            export_df.to_csv(csv_buffer, index=False)
            st.download_button("📥 Export Data as CSV", csv_buffer.getvalue(), f"{stock_name}_data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
else:
    st.info("⚠️ Please upload a CSV/XLSX or fetch real-time data to begin analysis.")
