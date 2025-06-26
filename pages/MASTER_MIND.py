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
# Simplified CSS with unsafe_allow_html
try:
    st.markdown("""
        <style>
        body {background-color: #f5f5f5;}
        .stButton>button {background-color: #1e3a8a; color: white; border-radius: 8px;}
        .stSelectbox, .stTextInput, .stNumberInput {background-color: #e5e7eb; border-radius: 8px;}
        .report-container {background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
        h1, h2, h3 {color: #374151; font-family: Arial, sans-serif;}
        .mode-banner {background-color: #e0f2fe; padding: 10px; border-radius: 5px; margin-bottom: 20px;}
        </style>
    """, unsafe_allow_html=True)
except TypeError:
    st.markdown("<!-- Custom CSS not applied due to Streamlit version incompatibility -->")

# Initialize session state
if 'real_time_data' not in st.session_state:
    st.session_state.real_time_data = None
if 'fundamental_data' not in st.session_state:
    st.session_state.fundamental_data = {
        'EPS': None, 'P/E': None, 'PEG': None, 'P/B': None,
        'ROE': None, 'Revenue': None, 'Debt/Equity': None
    }
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None

# Clear Analysis Button
def clear_analysis():
    st.session_state.real_time_data = None
    st.session_state.fundamental_data = {
        'EPS': None, 'P/E': None, 'PEG': None, 'P/B': None,
        'ROE': None, 'Revenue': None, 'Debt/Equity': None
    }
    st.session_state.csv_data = None
    st.success("Analysis cleared. Start a new analysis.")

# Sidebar for inputs
st.sidebar.header("Stock Data Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., KRRO)", value="KRRO")
submit_button = st.sidebar.button("Fetch Real-Time Data")
combine_data = st.sidebar.checkbox("Combine with Real-Time Data", value=False, disabled=st.session_state.csv_data is None)
clear_button = st.sidebar.button("Clear Analysis")
if clear_button:
    clear_analysis()

# Manual fundamental inputs
with st.sidebar.expander("Manual Fundamental Data (Optional)"):
    with st.form(key="fundamental_form"):
        eps = st.number_input("EPS", value=float(st.session_state.fundamental_data['EPS'] or 0.0), step=0.01, format="%.2f", placeholder="Enter EPS or leave blank")
        pe = st.number_input("P/E Ratio", value=float(st.session_state.fundamental_data['P/E'] or 0.0), step=0.01, format="%.2f", placeholder="Enter P/E or leave blank")
        peg = st.number_input("PEG Ratio", value=float(st.session_state.fundamental_data['PEG'] or 0.0), step=0.01, format="%.2f", placeholder="Enter PEG or leave blank")
        pb = st.number_input("P/B Ratio", value=float(st.session_state.fundamental_data['P/B'] or 0.0), step=0.01, format="%.2f", placeholder="Enter P/B or leave blank")
        roe = st.number_input("ROE (%)", value=float(st.session_state.fundamental_data['ROE'] or 0.0), step=0.01, format="%.2f", placeholder="Enter ROE or leave blank")
        revenue = st.number_input("Revenue (in millions)", value=float(st.session_state.fundamental_data['Revenue'] or 0.0), step=0.1, format="%.1f", placeholder="Enter Revenue or leave blank")
        debt_equity = st.number_input("Debt/Equity Ratio", value=float(st.session_state.fundamental_data['Debt/Equity'] or 0.0), step=0.01, format="%.2f", placeholder="Enter Debt/Equity or leave blank")
        submit_fundamentals = st.form_submit_button("Update Fundamentals")

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

# Fetch real-time data with yfinance
if submit_button:
    with st.spinner("Fetching real-time data..."):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d", interval="1d")
            if data.empty:
                st.error("No data found for the ticker. Please check the ticker or upload a CSV/XLSX.")
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
                st.session_state.fundamental_data = {
                    'EPS': info.get('trailingEps', st.session_state.fundamental_data['EPS']),
                    'P/E': info.get('trailingPE', st.session_state.fundamental_data['P/E']),
                    'PEG': info.get('pegRatio', st.session_state.fundamental_data['PEG']),
                    'P/B': info.get('priceToBook', st.session_state.fundamental_data['P/B']),
                    'ROE': info.get('returnOnEquity', st.session_state.fundamental_data['ROE']),
                    'Revenue': info.get('totalRevenue', st.session_state.fundamental_data['Revenue']),
                    'Debt/Equity': info.get('debtToEquity', st.session_state.fundamental_data['Debt/Equity'])
                }
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}. Please try another ticker (e.g., AAPL) or upload a CSV/XLSX.")

# CSV/XLSX upload
st.sidebar.subheader("Upload Technical Indicators CSV/XLSX")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"])
process_file_button = st.sidebar.button("Process File")

if process_file_button and uploaded_file:
    with st.spinner("Processing file..."):
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            required_columns = ['Date', 'Close', 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'Stoch_K', 'Williams_R', 'CCI', 'Momentum', 'ROC', 'OBV', 'Volume', 'Pivot', 'R1', 'S1', 'Fib_236', 'Fib_382', 'Fib_618', 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_A', 'Ichimoku_Senkou_B', 'PSAR']
            if not all(col in df.columns for col in required_columns):
                st.error("File must contain required columns: " + ", ".join(required_columns))
            else:
                df['Date'] = pd.to_datetime(df['Date'])
                st.session_state.csv_data = df
                st.success("File processed successfully!")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}. Ensure the file is a valid CSV/XLSX with required columns.")

# Function to generate PDF report using reportlab
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

# Function to combine CSV/XLSX and real-time data
def combine_dataframes(csv_df, real_time_data):
    if real_time_data is None:
        return csv_df
    real_time_df = pd.DataFrame([real_time_data])
    real_time_df['Date'] = pd.to_datetime(real_time_df['Date'])
    combined_df = pd.concat([csv_df, real_time_df], ignore_index=True)
    combined_df = combined_df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')
    for col in ['SMA_20', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'Stoch_K', 'Williams_R', 'CCI', 'Momentum', 'ROC', 'OBV', 'Pivot', 'R1', 'S1', 'Fib_236', 'Fib_382', 'Fib_618', 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_A', 'Ichimoku_Senkou_B', 'PSAR']:
        if col not in real_time_df.columns:
            combined_df[col] = combined_df[col].fillna(combined_df['Close'] if col in ['SMA_20', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50', 'BB_Middle', 'Pivot'] else 50 if col == 'RSI' else 0 if col in ['MACD', 'MACD_Signal', 'MACD_Histogram', 'CCI', 'Momentum', 'ROC'] else combined_df['Close'] * 1.05 if col == 'BB_Upper' else combined_df['Close'] * 0.95 if col == 'BB_Lower' else 50 if col == 'Stoch_K' else -50 if col == 'Williams_R' else combined_df['Close'] * 1.02 if col == 'R1' else combined_df['Close'] * 0.98 if col == 'S1' else combined_df['Close'] * 1.01 if col == 'Fib_618' else combined_df['Close'])
    return combined_df

# Function to analyze stock data
def analyze_stock_data(df=None, real_time_data=None, fundamental_data=None):
    stock_name = ticker.upper()
    is_real_time_only = df is None and real_time_data is not None
    data_source = combine_dataframes(df, real_time_data) if df is not None and combine_data else df if df is not None else pd.DataFrame([real_time_data]) if real_time_data else None
    if data_source is None:
        return None, None, None, stock_name, None, is_real_time_only, 50

    latest = data_source.iloc[-1]
    prev = data_source.iloc[-2] if len(data_source) > 1 else latest

    price = latest['Close']
    sma_20 = latest.get('SMA_20', price) if not is_real_time_only else price
    sma_50 = latest.get('SMA_50', price) if not is_real_time_only else price
    sma_200 = latest.get('SMA_200', price) if not is_real_time_only else price
    ema_20 = latest.get('EMA_20', price) if not is_real_time_only else price
    ema_50 = latest.get('EMA_50', price) if not is_real_time_only else price
    rsi = latest.get('RSI', 50) if not is_real_time_only else 50
    macd = latest.get('MACD', 0) if not is_real_time_only else 0
    macd_signal = latest.get('MACD_Signal', 0) if not is_real_time_only else 0
    macd_hist = latest.get('MACD_Histogram', 0) if not is_real_time_only else 0
    bb_upper = latest.get('BB_Upper', price * 1.05) if not is_real_time_only else price * 1.05
    bb_middle = latest.get('BB_Middle', price) if not is_real_time_only else price
    bb_lower = latest.get('BB_Lower', price * 0.95) if not is_real_time_only else price * 0.95
    stoch_k = latest.get('Stoch_K', 50) if not is_real_time_only else 50
    williams_r = latest.get('Williams_R', -50) if not is_real_time_only else -50
    cci = latest.get('CCI', 0) if not is_real_time_only else 0
    momentum = latest.get('Momentum', 0) if not is_real_time_only else 0
    roc = latest.get('ROC', 0) if not is_real_time_only else 0
    obv = latest.get('OBV', latest['Volume']) if not is_real_time_only else latest['Volume']
    volume = latest['Volume']
    pivot = latest.get('Pivot', price) if not is_real_time_only else price
    r1 = latest.get('R1', price * 1.02) if not is_real_time_only else price * 1.02
    s1 = latest.get('S1', price * 0.98) if not is_real_time_only else price * 0.98
    fib_618 = latest.get('Fib_618', price * 1.01) if not is_real_time_only else price * 1.01

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
### Quick Scan: {stock_name} ({latest['Date']})
- **Price**: ${price:.2f} ({trend} trend)
- **Support/Resistance**: Support at ${s1:.2f}, Resistance at ${r1:.2f}
- **RSI**: {rsi:.2f} ({'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'})
- **Recommendation**: {'Buy near support (${s1:.2f}) for bounce to ${r1:.2f}' if rsi < 30 else f'Wait for breakout above ${sma_20:.2f}'}
"""

    moderate_detail = f"""
### Moderate Detail: {stock_name} ({latest['Date']})
- **Price Trend**: ${price:.2f}, {trend} (SMA20: ${sma_20:.2f}, SMA50: ${sma_50:.2f})
- **Momentum**:
  - RSI: {rsi:.2f} ({'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'})
  - MACD: {macd:.2f} (Signal: {macd_signal:.2f}, {'Bearish' if macd < macd_signal else 'Bullish'})
- **Bollinger Bands**: Price near {'lower' if price < bb_middle else 'upper'} band (Lower: ${bb_lower:.2f}, Upper: ${bb_upper:.2f})
- **ADX**: {adx_value:.2f} ({'Strong Trend' if adx_value > 25 else 'Weak Trend'})
- **Key Levels**: Support: ${s1:.2f}, Resistance: ${r1:.2f}, Fib 61.8%: ${fib_618:.2f}
- **Fundamentals**:
  {''.join([f'  - {k}: {v:.2f}\n' for k, v in fundamental_data.items() if v is not None]) if fundamental_data and any(v is not None for v in fundamental_data.values()) else '  - Not provided\n'}
- **Recommendation**:
  - Traders: {'Buy near ${s1:.2f} for bounce to ${r1:.2f}' if rsi < 30 or price < bb_middle else f'Wait for breakout above ${r1:.2f}'}
  - Investors: Confirm trend reversal above ${sma_20:.2f}.
"""

    in_depth = f"""
### In-Depth Analysis: {stock_name} ({latest['Date']})
#### Key Takeaways
- **Price**: ${price:.2f}, {trend} trend
- **Historical Pattern**: {trend_pattern}
- **ADX**: {adx_value:.2f} ({'Strong Trend' if adx_value > 25 else 'Weak Trend'})

#### Price Trends
- **Close**: ${price:.2f}, {'below' if price < sma_20 else 'above'} SMA20 (${sma_20:.2f}), SMA50 (${sma_50:.2f}), SMA200 (${sma_200:.2f})
- **Trend**: {trend} (EMA20: ${ema_20:.2f}, EMA50: ${ema_50:.2f})

#### Momentum Indicators
- **RSI**: {rsi:.2f} ({'Oversold (<30)' if rsi < 30 else 'Overbought (>70)' if rsi > 70 else 'Neutral'})
- **MACD**: {macd:.2f}, Signal: {macd_signal:.2f}, Histogram: {macd_hist:.2f} ({'Bearish' if macd < macd_signal else 'Bullish'})
- **Stochastic %K**: {stoch_k:.2f} ({'Oversold' if stoch_k < 20 else 'Overbought' if stoch_k > 80 else 'Neutral'})
- **Williams %R**: {williams_r:.2f} ({'Oversold' if williams_r < -80 else 'Overbought' if williams_r > -20 else 'Neutral'})
- **CCI**: {cci:.2f}, Momentum: {momentum:.2f}, ROC: {roc:.2f}

#### Volatility & Bollinger Bands
- **Price Position**: {'Near lower band' if price < bb_middle else 'Near upper band'} (Lower: ${bb_lower:.2f}, Middle: ${bb_middle:.2f}, Upper: ${bb_upper:.2f})

#### Volume
- **OBV**: {obv:,.0f} ({'Declining' if obv < prev.get('OBV', obv) else 'Rising'})
- **Volume**: {volume:,.0f} (Recent trend: {'Low' if volume < data_source['Volume'].mean() else 'High'})

#### Fundamentals
{''.join([f'- {k}: {v:.2f}\n' for k, v in fundamental_data.items() if v is not None]) if fundamental_data and any(v is not None for v in fundamental_data.values()) else '- Not provided\n'}

#### Recommendation
- **Conservative Investors**: Wait for price to break above SMA20 (${sma_20:.2f}).
- **Traders**: {'Buy near support (${s1:.2f}) for bounce to ${r1:.2f}' if rsi < 30 or stoch_k < 20 else f'Wait for breakout above ${r1:.2f}'}.
- **Key Levels**: Support: ${s1:.2f}, Resistance: ${r1:.2f}, Fib 61.8%: ${fib_618:.2f}.
- **Risk**: {'High' if volume < data_source['Volume'].mean() else 'Moderate'} due to {'low volume' if volume < data_source['Volume'].mean() else 'market volatility'}.
"""

    return quick_scan, moderate_detail, in_depth, stock_name, data_source, is_real_time_only, adx_value

# Main app
st.title("📈 Stock Technical Analysis")
st.markdown("Fetch real-time data or upload a CSV/XLSX file with technical indicators to generate a stock analysis report.")

# Mode indicator and comparison table
mode = "Real-Time Only" if st.session_state.csv_data is None and st.session_state.real_time_data is not None else "XLSX/CSV Only" if st.session_state.csv_data is not None and not combine_data else "Combined" if st.session_state.csv_data is not None and combine_data else "No Data"
st.markdown(f"<div class='mode-banner'><b>Active Mode: {mode}</b><br>{'Real-time price and fundamentals from yfinance.' if mode == 'Real-Time Only' else 'Historical data and indicators from uploaded CSV/XLSX.' if mode == 'XLSX/CSV Only' else 'Combines CSV/XLSX historical data with real-time price/fundamentals.' if mode == 'Combined' else 'Please fetch data or upload a file to begin.'}</div>", unsafe_allow_html=True)

with st.expander("How Reports Differ by Mode"):
    st.markdown("""
    | **Mode** | **Data Source** | **Report Features** | **Limitations** |
    |----------|-----------------|---------------------|-----------------|
    | XLSX/CSV Only | Uploaded file | Full technical analysis (RSI, SMA, ADX, charts) | No real-time updates |
    | Real-Time Only | yfinance | Latest price, fundamentals; default indicators | No historical trends/charts |
    | Combined | XLSX/CSV + yfinance | Full analysis with real-time price updates | Requires matching ticker |
    """)

# Display real-time data
if st.session_state.real_time_data:
    rt = st.session_state.real_time_data
    st.markdown(f"### Current Stock Data: {ticker.upper()} (As of {rt['Date']})")
    st.write(f"- **Open**: ${rt['Open']:.2f}")
    st.write(f"- **High**: ${rt['High']:.2f}")
    st.write(f"- **Low**: ${rt['Low']:.2f}")
    st.write(f"- **Close**: ${rt['Close']:.2f}")
    st.write(f"- **Volume**: {rt['Volume']:,.0f}")

# Display fundamental data
if any(v is not None for v in st.session_state.fundamental_data.values()):
    st.markdown("### Fundamental Data")
    for k, v in st.session_state.fundamental_data.items():
        if v is not None:
            st.write(f"- **{k}**: {v:.2f}")

# Analyze data
data_source = combine_dataframes(st.session_state.csv_data, st.session_state.real_time_data) if st.session_state.csv_data is not None and combine_data else st.session_state.csv_data if st.session_state.csv_data is not None else st.session_state.real_time_data
if data_source is not None:
    quick_scan, moderate_detail, in_depth, stock_name, df, is_real_time_only, adx_value = analyze_stock_data(
        st.session_state.csv_data,
        st.session_state.real_time_data if combine_data or st.session_state.csv_data is None else None,
        st.session_state.fundamental_data
    )

    if is_real_time_only:
        st.warning("Real-time data lacks technical indicators (e.g., RSI, SMA). Upload a CSV/XLSX for full analysis in Visual Summary and Interactive Dashboard.")

    # Report selection
    st.markdown("<div class='report-container'>", unsafe_allow_html=True)
    report_type = st.selectbox("Select Report Type", ["Quick Scan", "Moderate Detail", "In-Depth Analysis", "Visual Summary", "Interactive Dashboard"])

    # Display report
    date_str = df['Date'].iloc[-1].strftime('%Y-%m-%d') if isinstance(df['Date'], pd.Series) else df['Date']
    if report_type == "Quick Scan":
        st.markdown(quick_scan)
    elif report_type == "Moderate Detail":
        st.markdown(moderate_detail)
    elif report_type == "In-Depth Analysis":
        st.markdown(in_depth)
    elif report_type == "Visual Summary":
        st.markdown(f"### Visual Summary: {stock_name} ({date_str})")
        st.write(f"**Price**: ${df['Close'].iloc[-1]:.2f}")
        st.write(f"**Trend**: {'Bearish' if df['Close'].iloc[-1] < df.get('SMA_20', df['Close']).iloc[-1] else 'Bullish'}")
        
        if not is_real_time_only and len(df) > 1:
            fig = px.line(df.tail(30), x='Date', y='Close', title='Price Trend (Last 30 Days)', hover_data=['Open', 'High', 'Low'])
            if 'SMA_20' in df.columns:
                fig.add_scatter(x=df['Date'], y=df['SMA_20'], name='SMA20', line=dict(color='orange'))
            if 'SMA_50' in df.columns:
                fig.add_scatter(x=df['Date'], y=df['SMA_50'], name='SMA50', line=dict(color='green'))
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Price trend chart unavailable with real-time data only. Upload a CSV/XLSX for historical data.")

        if not is_real_time_only and 'RSI' in df.columns and len(df) > 1:
            fig_rsi = px.line(df.tail(30), x='Date', y='RSI', title='RSI (Last 30 Days)')
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(hovermode='x unified')
            st.plotly_chart(fig_rsi, use_container_width=True)
        else:
            st.info("RSI chart unavailable with real-time data only. Upload a CSV/XLSX for RSI data.")

        rsi_value = df.get('RSI', 50) if not is_real_time_only else 50
        if isinstance(rsi_value, pd.Series):
            rsi_value = rsi_value.iloc[-1]
        st.markdown(f"- **Support**: ${df.get('S1', df['Close'] * 0.98).iloc[-1]:.2f}, **Resistance**: ${df.get('R1', df['Close'] * 1.02).iloc[-1]:.2f}")
        st.markdown(f"- **RSI**: {rsi_value:.2f} ({'Oversold' if rsi_value < 30 else 'Overbought' if rsi_value > 70 else 'Neutral'})")
        st.markdown(f"- **Recommendation**: {'Buy near support' if rsi_value < 30 else 'Wait for breakout'}")
    else:
        st.markdown(f"### Interactive Dashboard: {stock_name}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Technical Indicators")
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
            st.markdown("#### Fundamental Metrics")
            for k, v in st.session_state.fundamental_data.items():
                if v is not None:
                    st.write(f"- **{k}**: {v:.2f}")
        
        if not is_real_time_only and len(df) > 1:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df['Date'].tail(30), open=df['Open'].tail(30), high=df['High'].tail(30), low=df['Low'].tail(30), close=df['Close'].tail(30), name='Candlestick'))
            if 'BB_Upper' in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'].tail(30), y=df['BB_Upper'].tail(30), name='BB Upper', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=df['Date'].tail(30), y=df['BB_Lower'].tail(30), name='BB Lower', line=dict(color='green')))
            fig.update_layout(title='Candlestick with Bollinger Bands (Last 30 Days)', xaxis_title='Date', yaxis_title='Price', hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Candlestick chart unavailable with real-time data only. Upload a CSV/XLSX for full analysis.")

    st.markdown("</div>", unsafe_allow_html=True)

    # Download reports
    report_content = quick_scan if report_type == "Quick Scan" else moderate_detail if report_type == "Moderate Detail" else in_depth
    buffer = io.StringIO()
    buffer.write(report_content)
    st.download_button(
        label="Download Markdown Report",
        data=buffer.getvalue(),
        file_name=f"{stock_name}_{report_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
        mime="text/markdown"
    )
    pdf_buffer = generate_pdf_report(report_content, stock_name, report_type)
    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name=f"{stock_name}_{report_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf"
    )

    if data_source is not None:
        export_df = data_source.copy()
        if st.session_state.fundamental_data and any(v is not None for v in st.session_state.fundamental_data.values()):
            for k, v in st.session_state.fundamental_data.items():
                export_df[k] = v
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Export Data as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"{stock_name}_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
else:
    st.info("Please fetch real-time data or upload a CSV/XLSX to begin analysis.")
