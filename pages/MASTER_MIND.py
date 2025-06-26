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
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #1e3a8a; color: white; border-radius: 8px;}
    .stSelectbox, .stTextInput, .stNumberInput {background-color: #e5e7eb; border-radius: 8px;}
    .report-container {background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    h1 {color: #1e3a8a; font-family: 'Arial', sans-serif;}
    h2, h3 {color: #374151; font-family: 'Arial', sans-serif;}
    .sidebar .sidebar-content {background-color: #e5e7eb;}
    </style>
""", unsafe_html=True)

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

# Sidebar for inputs
st.sidebar.header("Stock Data Input")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., KRRO)", value="KRRO")
submit_button = st.sidebar.button("Fetch Real-Time Data")

# Manual fundamental inputs
st.sidebar.subheader("Manual Fundamental Data (Optional)")
with st.sidebar.form(key="fundamental_form"):
    eps = st.number_input("EPS", value=st.session_state.fundamental_data['EPS'], step=0.01, format="%.2f", placeholder="Enter EPS or leave blank")
    pe = st.number_input("P/E Ratio", value=st.session_state.fundamental_data['P/E'], step=0.01, format="%.2f", placeholder="Enter P/E or leave blank")
    peg = st.number_input("PEG Ratio", value=st.session_state.fundamental_data['PEG'], step=0.01, format="%.2f", placeholder="Enter PEG or leave blank")
    pb = st.number_input("P/B Ratio", value=st.session_state.fundamental_data['P/B'], step=0.01, format="%.2f", placeholder="Enter P/B or leave blank")
    roe = st.number_input("ROE (%)", value=st.session_state.fundamental_data['ROE'], step=0.01, format="%.2f", placeholder="Enter ROE or leave blank")
    revenue = st.number_input("Revenue (in millions)", value=st.session_state.fundamental_data['Revenue'], step=0.1, format="%.1f", placeholder="Enter Revenue or leave blank")
    debt_equity = st.number_input("Debt/Equity Ratio", value=st.session_state.fundamental_data['Debt/Equity'], step=0.01, format="%.2f", placeholder="Enter Debt/Equity or leave blank")
    submit_fundamentals = st.form_submit_button("Update Fundamentals")

if submit_fundamentals:
    st.session_state.fundamental_data = {
        'EPS': eps if eps != 0 else None,
        'P/E': pe if pe != 0 else None,
        'PEG': peg if peg != 0 else None,
        'P/B': pb if pb != 0 else None,
        'ROE': roe if roe != 0 else None,
        'Revenue': revenue if revenue != 0 else None,
        'Debt/Equity': debt_equity if debt_equity != 0 else None
    }

# Fetch real-time data with yfinance
if submit_button:
    try:
        stock = yf.Ticker(ticker)
        # Fetch latest daily data
        data = stock.history(period="1d", interval="1d")
        if data.empty:
            st.error("No data found for the ticker. Please check the ticker or upload a CSV.")
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
            # Fetch fundamental data
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
        st.error(f"Error fetching data: {str(e)}. Please upload a CSV or enter manual data.")

# CSV upload
st.sidebar.subheader("Upload Technical Indicators CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
process_csv_button = st.sidebar.button("Process CSV")

if process_csv_button and uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = ['Date', 'Close', 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'Stoch_K', 'Williams_R', 'CCI', 'Momentum', 'ROC', 'OBV', 'Volume', 'Pivot', 'R1', 'S1', 'Fib_236', 'Fib_382', 'Fib_618', 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_A', 'Ichimoku_Senkou_B', 'PSAR']
        if not all(col in df.columns for col in required_columns):
            st.error("CSV must contain required columns: " + ", ".join(required_columns))
        else:
            df['Date'] = pd.to_datetime(df['Date'])
            st.session_state.csv_data = df
            st.success("CSV processed successfully!")
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")

# Function to generate PDF report using reportlab
def generate_pdf_report(report_content, stock_name, report_type):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    custom_style = ParagraphStyle(name='Custom', parent=styles['Normal'], fontSize=12, leading=14)
    heading_style = ParagraphStyle(name='Heading', parent=styles['Heading1'], fontSize=16, leading=18, spaceAfter=12)
    elements = []

    # Add title
    elements.append(Paragraph(f"Stock Analysis Report: {stock_name} ({datetime.now().strftime('%Y-%m-%d')})", heading_style))
    elements.append(Spacer(1, 12))

    # Add report content
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

    # Add fundamentals table if available
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
    data_source = df if df is not None else pd.DataFrame([real_time_data]) if real_time_data else None
    if data_source is None:
        return None, None, None, stock_name, None

    latest = data_source.iloc[-1]
    prev = data_source.iloc[-2] if len(data_source) > 1 else latest

    # Extract indicators
    price = latest['Close']
    sma_20 = latest.get('SMA_20', price)
    sma_50 = latest.get('SMA_50', price)
    sma_200 = latest.get('SMA_200', price)
    ema_20 = latest.get('EMA_20', price)
    ema_50 = latest.get('EMA_50', price)
    rsi = latest.get('RSI', 50)
    macd = latest.get('MACD', 0)
    macd_signal = latest.get('MACD_Signal', 0)
    macd_hist = latest.get('MACD_Histogram', 0)
    bb_upper = latest.get('BB_Upper', price * 1.05)
    bb_middle = latest.get('BB_Middle', price)
    bb_lower = latest.get('BB_Lower', price * 0.95)
    stoch_k = latest.get('Stoch_K', 50)
    williams_r = latest.get('Williams_R', -50)
    cci = latest.get('CCI', 0)
    momentum = latest.get('Momentum', 0)
    roc = latest.get('ROC', 0)
    obv = latest.get('OBV', latest['Volume'])
    volume = latest['Volume']
    pivot = latest.get('Pivot', price)
    r1 = latest.get('R1', price * 1.02)
    s1 = latest.get('S1', price * 0.98)
    fib_618 = latest.get('Fib_618', price * 1.01)

    # Calculate ADX
    adx_value = 50  # Default
    if df is not None and all(col in df.columns for col in ['High', 'Low', 'Close']):
        adx_indicator = ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
        adx_value = adx_indicator.adx().iloc[-1]

    # Historical trend analysis
    trend_pattern = "Neutral"
    if df is not None and len(df) > 10:
        highs = df['High'].rolling(window=10).max()
        lows = df['Low'].rolling(window=10).min()
        if highs.iloc[-1] > highs.iloc[-2] and lows.iloc[-1] > lows.iloc[-2]:
            trend_pattern = "Higher Highs & Lows (Bullish)"
        elif highs.iloc[-1] < highs.iloc[-2] and lows.iloc[-1] < lows.iloc[-2]:
            trend_pattern = "Lower Highs & Lows (Bearish)"

    # Determine trend
    trend = "Bearish" if price < sma_20 and price < sma_50 else "Bullish" if price > sma_20 and price > sma_50 else "Neutral"

    # Generate reports
    quick_scan = f"""
### Quick Scan: {stock_name} ({latest['Date']})
- **Price**: ${price:.2f} ({trend} trend)
- **Support/Resistance**: Support at ${s1:.2f}, Resistance at ${r1:.2f}
- **RSI**: {rsi:.2f} ({'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'})
- **Recommendation**: {'Buy near support ($' + f'{s1:.2f}) for bounce to ${r1:.2f}' if rsi < 30 else 'Wait for breakout above $' + f'{sma_20:.2f}'}
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
  {'\n  - '.join([f'{k}: {v:.2f}' for k, v in fundamental_data.items() if v is not None]) if fundamental_data and any(v is not None for v in fundamental_data.values()) else 'Not provided'}
- **Recommendation**: 
  - Traders: {'Buy near ${s1:.2f} for bounce to ${r1:.2f}' if rsi < 30 or price < bb_middle else 'Wait for breakout above ${r1:.2f}'}
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
{'- ' + '\n- '.join([f'{k}: {v:.2f}' for k, v in fundamental_data.items() if v is not None]) if fundamental_data and any(v is not None for v in fundamental_data.values()) else '- Not provided'}

#### Recommendation
- **Conservative Investors**: Wait for price to break above SMA20 (${sma_20:.2f}).
- **Traders**: {'Buy near support (${s1:.2f}) for bounce to ${r1:.2f}' if rsi < 30 or stoch_k < 20 else 'Wait for breakout above ${r1:.2f}'}.
- **Key Levels**: Support: ${s1:.2f}, Resistance: ${r1:.2f}, Fib 61.8%: ${fib_618:.2f}.
- **Risk**: {'High' if volume < data_source['Volume'].mean() else 'Moderate'} due to {'low volume' if volume < data_source['Volume'].mean() else 'market volatility'}.
"""

    return quick_scan, moderate_detail, in_depth, stock_name, data_source

# Main app
st.title("📈 Stock Technical Analysis")
st.markdown("Fetch real-time data or upload a CSV with technical indicators to generate a stock analysis report.")

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
data_source = st.session_state.csv_data if st.session_state.csv_data is not None else st.session_state.real_time_data
if data_source is not None:
    quick_scan, moderate_detail, in_depth, stock_name, df = analyze_stock_data(
        st.session_state.csv_data,
        st.session_state.real_time_data,
        st.session_state.fundamental_data
    )

    # Report selection
    st.markdown("<div class='report-container'>", unsafe_html=True)
    report_type = st.selectbox("Select Report Type", ["Quick Scan", "Moderate Detail", "In-Depth Analysis", "Visual Summary", "Interactive Dashboard"])

    # Display report
    if report_type == "Quick Scan":
        st.markdown(quick_scan)
    elif report_type == "Moderate Detail":
        st.markdown(moderate_detail)
    elif report_type == "In-Depth Analysis":
        st.markdown(in_depth)
    elif report_type == "Visual Summary":
        st.markdown(f"### Visual Summary: {stock_name} ({df['Date'].iloc[-1].strftime('%Y-%m-%d')})")
        st.write(f"**Price**: ${df['Close'].iloc[-1]:.2f}")
        st.write(f"**Trend**: {'Bearish' if df['Close'].iloc[-1] < df.get('SMA_20', df['Close']).iloc[-1] else 'Bullish'}")
        
        # Price trend chart
        fig = px.line(df.tail(30) if df is not None else df, x='Date', y='Close', title='Price Trend (Last 30 Days)')
        if df is not None:
            fig.add_scatter(x=df['Date'], y=df.get('SMA_20', df['Close']), name='SMA20', line=dict(color='orange'))
            fig.add_scatter(x=df['Date'], y=df.get('SMA_50', df['Close']), name='SMA50', line=dict(color='green'))
        st.plotly_chart(fig, use_container_width=True)
        
        # RSI chart
        if 'RSI' in df.columns:
            fig_rsi = px.line(df.tail(30), x='Date', y='RSI', title='RSI (Last 30 Days)')
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        st.markdown(f"- **Support**: ${df.get('S1', df['Close'] * 0.98).iloc[-1]:.2f}, **Resistance**: ${df.get('R1', df['Close'] * 1.02).iloc[-1]:.2f}")
        st.markdown(f"- **RSI**: {df.get('RSI', 50).iloc[-1]:.2f} ({'Oversold' if df.get('RSI', 50).iloc[-1] < 30 else 'Overbought' if df.get('RSI', 50).iloc[-1] > 70 else 'Neutral'})")
        st.markdown(f"- **Recommendation**: {'Buy near support' if df.get('RSI', 50).iloc[-1] < 30 else 'Wait for breakout'}")
    else:
        # Interactive Dashboard
        st.markdown(f"### Interactive Dashboard: {stock_name}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Technical Indicators")
            st.write(f"- **Price**: ${df['Close'].iloc[-1]:.2f}")
            st.write(f"- **RSI**: {df.get('RSI', 50).iloc[-1]:.2f}")
            st.write(f"- **MACD**: {df.get('MACD', 0).iloc[-1]:.2f} (Signal: {df.get('MACD_Signal', 0).iloc[-1]:.2f})")
            st.write(f"- **Stochastic %K**: {df.get('Stoch_K', 50).iloc[-1]:.2f}")
            adx_indicator = ADXIndicator(df['High'], df['Low'], df['Close'], window=14) if df is not None and all(col in df.columns for col in ['High', 'Low', 'Close']) else None
            adx_value = adx_indicator.adx().iloc[-1] if adx_indicator else 50
            st.write(f"- **ADX**: {adx_value:.2f} ({'Strong Trend' if adx_value > 25 else 'Weak Trend'})")
            st.write(f"- **Support**: ${df.get('S1', df['Close'] * 0.98).iloc[-1]:.2f}")
            st.write(f"- **Resistance**: ${df.get('R1', df['Close'] * 1.02).iloc[-1]:.2f}")
        with col2:
            st.markdown("#### Fundamental Metrics")
            for k, v in st.session_state.fundamental_data.items():
                if v is not None:
                    st.write(f"- **{k}**: {v:.2f}")
        
        # Price chart with Bollinger Bands
        if df is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'].tail(30), y=df['Close'].tail(30), name='Close'))
            if 'BB_Upper' in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'].tail(30), y=df['BB_Upper'].tail(30), name='BB Upper', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=df['Date'].tail(30), y=df['BB_Lower'].tail(30), name='BB Lower', line=dict(color='green')))
            fig.update_layout(title='Price with Bollinger Bands (Last 30 Days)', xaxis_title='Date', yaxis_title='Price')
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_html=True)

    # Download reports
    report_content = quick_scan if report_type == "Quick Scan" else moderate_detail if report_type == "Moderate Detail" else in_depth
    # Markdown download
    buffer = io.StringIO()
    buffer.write(report_content)
    st.download_button(
        label="Download Markdown Report",
        data=buffer.getvalue(),
        file_name=f"{stock_name}_{report_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
        mime="text/markdown"
    )
    # PDF download
    pdf_buffer = generate_pdf_report(report_content, stock_name, report_type)
    st.download_button(
        label="Download PDF Report",
        data=pdf_buffer,
        file_name=f"{stock_name}_{report_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf"
    )

    # Export combined data as CSV
    if st.session_state.csv_data is not None or st.session_state.real_time_data:
        export_df = pd.DataFrame([st.session_state.real_time_data]) if st.session_state.real_time_data else st.session_state.csv_data
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
    st.info("Please fetch real-time data or upload a CSV to begin analysis.")
