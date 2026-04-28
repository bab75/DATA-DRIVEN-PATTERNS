import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io

st.set_page_config(page_title="Stock Technical Analysis", layout="wide", page_icon="📈")

st.markdown("""
<style>
body { font-family: 'Segoe UI', sans-serif; }
.stButton>button { background: linear-gradient(45deg,#4a90e2,#63b3ed); color:white; border:none; border-radius:8px; padding:8px 18px; }
.metric-card { background:#f0f7ff; padding:12px; border-radius:10px; border-left:4px solid #3498db; margin:5px 0; }
.green { color: #27ae60; font-weight:bold; }
.red { color: #e74c3c; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────
for k, v in [('csv_data', None), ('ticker', ''), ('rt_data', None),
              ('fund_data', None), ('analysis_done', False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ────────────────────────────────────────────────────
def calc_indicators(df):
    """Calculate all technical indicators from OHLCV data."""
    df = df.copy().sort_values('Date').reset_index(drop=True)
    c = df['Close']
    h = df['High']
    l = df['Low']
    v = df['Volume']

    # Moving averages
    for w in [20, 50, 200]:
        df[f'SMA_{w}'] = c.rolling(w).mean()
        df[f'EMA_{w}'] = c.ewm(span=w, adjust=False).mean()

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    df['BB_Mid'] = c.rolling(20).mean()
    std = c.rolling(20).std()
    df['BB_Upper'] = df['BB_Mid'] + 2 * std
    df['BB_Lower'] = df['BB_Mid'] - 2 * std
    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid'] * 100).round(2)

    # ATR
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()

    # Stochastic
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df['Stoch_K'] = 100 * (c - low14) / (high14 - low14 + 1e-9)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # Williams %R
    df['Williams_R'] = -100 * (high14 - c) / (high14 - low14 + 1e-9)

    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    # VWAP (rolling 20)
    df['VWAP'] = (c * v).rolling(20).sum() / v.rolling(20).sum()

    # Pivot points
    df['Pivot'] = (h + l + c) / 3
    df['R1'] = 2 * df['Pivot'] - l
    df['S1'] = 2 * df['Pivot'] - h
    df['R2'] = df['Pivot'] + (h - l)
    df['S2'] = df['Pivot'] - (h - l)

    # Volatility
    df['Volatility'] = c.pct_change().rolling(20).std() * 100

    # ADX (simple)
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    plus_di = 100 * plus_dm.rolling(14).mean() / (tr.rolling(14).mean() + 1e-9)
    minus_di = 100 * minus_dm.rolling(14).mean() / (tr.rolling(14).mean() + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df['ADX'] = dx.rolling(14).mean()

    return df.dropna(subset=['SMA_20', 'RSI', 'MACD'])


def get_signal(df):
    r = df.iloc[-1]
    signals = []
    score = 0

    # RSI
    if r['RSI'] < 30:   signals.append("🟢 RSI Oversold (Buy)");  score += 2
    elif r['RSI'] > 70: signals.append("🔴 RSI Overbought (Sell)"); score -= 2
    else:               signals.append("⚪ RSI Neutral")

    # MACD
    if r['MACD'] > r['MACD_Signal']: signals.append("🟢 MACD Bullish Cross"); score += 1
    else:                             signals.append("🔴 MACD Bearish Cross"); score -= 1

    # Price vs MA
    if r['Close'] > r['SMA_20'] > r['SMA_50']: signals.append("🟢 Price above SMA20 & SMA50"); score += 2
    elif r['Close'] < r['SMA_20'] < r['SMA_50']: signals.append("🔴 Price below SMA20 & SMA50"); score -= 2
    else: signals.append("⚪ Mixed MA signals")

    # Stochastic
    if r['Stoch_K'] < 20: signals.append("🟢 Stoch Oversold"); score += 1
    elif r['Stoch_K'] > 80: signals.append("🔴 Stoch Overbought"); score -= 1

    # ADX trend strength
    adx_str = f"Strong ({r['ADX']:.1f})" if r['ADX'] > 25 else f"Weak ({r['ADX']:.1f})"
    signals.append(f"📊 ADX Trend: {adx_str}")

    # Volume
    avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
    if r['Volume'] > avg_vol * 1.5: signals.append("🟢 Volume Spike (Confirmation)")
    elif r['Volume'] < avg_vol * 0.5: signals.append("⚠️ Low Volume (Weak signal)")

    if score >= 3:   recommendation = "🟢 STRONG BUY"
    elif score >= 1: recommendation = "🟡 BUY / WATCH"
    elif score <= -3: recommendation = "🔴 STRONG SELL"
    elif score <= -1: recommendation = "🟠 SELL / CAUTION"
    else:            recommendation = "⚪ NEUTRAL / HOLD"

    return recommendation, signals, score


def fetch_realtime(sym):
    try:
        tk = yf.Ticker(sym)
        hist = tk.history(period='1y', interval='1d')
        if hist.empty:
            return None, None, "No data found"
        hist = hist.reset_index()
        hist.columns = [c.replace(' ', '_') for c in hist.columns]
        hist = hist.rename(columns={'Datetime': 'Date', 'Stock_Splits': 'Stock_Splits'})
        if 'Date' not in hist.columns and 'Datetime' in hist.columns:
            hist = hist.rename(columns={'Datetime': 'Date'})
        hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
        info = {}
        try:
            raw = tk.info
            info = {
                'EPS': raw.get('trailingEps'),
                'P/E': raw.get('trailingPE'),
                'P/B': raw.get('priceToBook'),
                'ROE': raw.get('returnOnEquity'),
                'Revenue ($B)': round(raw.get('totalRevenue', 0) / 1e9, 2) if raw.get('totalRevenue') else None,
                'Debt/Equity': raw.get('debtToEquity'),
                'Market Cap ($B)': round(raw.get('marketCap', 0) / 1e9, 2) if raw.get('marketCap') else None,
                'Sector': raw.get('sector'),
                'Name': raw.get('shortName', sym),
            }
        except:
            pass
        return hist[['Date','Open','High','Low','Close','Volume']], info, None
    except Exception as e:
        return None, None, str(e)


# ── SIDEBAR ────────────────────────────────────────────────────
st.sidebar.header("📊 Controls")

# Ticker input - ALWAYS shown and used as stock name
ticker_input = st.sidebar.text_input("🔎 Stock Ticker", value=st.session_state.ticker or "AAPL").strip().upper()
if ticker_input != st.session_state.ticker:
    st.session_state.ticker = ticker_input
    st.session_state.csv_data = None
    st.session_state.rt_data = None
    st.session_state.fund_data = None
    st.session_state.analysis_done = False

st.sidebar.markdown("---")
st.sidebar.subheader("Option 1: Upload OHLCV File")
uploaded_file = st.sidebar.file_uploader("CSV or XLSX (needs Date,Open,High,Low,Close,Volume)", type=["csv","xlsx"])
process_btn = st.sidebar.button("📥 Process Uploaded File")

st.sidebar.markdown("---")
st.sidebar.subheader("Option 2: Fetch from Yahoo Finance")
fetch_btn = st.sidebar.button("🌐 Fetch Real-Time Data")

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear / Reset"):
    for k in ['csv_data','rt_data','fund_data','analysis_done']:
        st.session_state[k] = None if k != 'analysis_done' else False
    st.rerun()

# ── PROCESS UPLOAD ─────────────────────────────────────────────
if process_btn and uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file, engine='openpyxl')
        df.columns = [c.strip() for c in df.columns]
        # Normalize common column name variants
        rename_map = {c.lower(): c for c in ['Date','Open','High','Low','Close','Volume']}
        df = df.rename(columns={c: rename_map[c.lower()] for c in df.columns if c.lower() in rename_map})
        required = ['Date','Open','High','Low','Close','Volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.sidebar.error(f"Missing: {', '.join(missing)}")
        else:
            df['Date'] = pd.to_datetime(df['Date'])
            st.session_state.csv_data = df[required]
            st.session_state.rt_data = None
            st.session_state.fund_data = None
            st.session_state.analysis_done = True
            st.sidebar.success(f"✅ Loaded {len(df)} rows")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# ── FETCH REALTIME ─────────────────────────────────────────────
if fetch_btn and ticker_input:
    with st.spinner(f"Fetching {ticker_input}..."):
        hist, info, err = fetch_realtime(ticker_input)
    if err:
        st.sidebar.error(f"❌ {err}")
    else:
        st.session_state.rt_data = hist
        st.session_state.fund_data = info
        st.session_state.csv_data = None
        st.session_state.analysis_done = True
        st.sidebar.success(f"✅ {info.get('Name', ticker_input)} loaded")

# ── MAIN AREA ──────────────────────────────────────────────────
st.title(f"📈 Stock Technical Analysis — {st.session_state.ticker or 'Enter Ticker'}")

with st.expander("ℹ️ How to use", expanded=False):
    st.markdown("""
| Step | Action |
|------|--------|
| 1 | Enter stock ticker (e.g. AAPL, TSLA) in sidebar |
| 2 | **Option A**: Upload your own CSV/XLSX with OHLCV data |
| 2 | **Option B**: Click *Fetch Real-Time Data* to pull from Yahoo Finance |
| 3 | App auto-calculates 15+ indicators and generates Buy/Sell signals |
| 4 | Explore tabs: Charts, Signals, Indicators, Fundamentals |
    """)

if not st.session_state.analysis_done:
    st.info("👈 Enter a ticker and either upload a file or fetch real-time data to begin.")
    st.stop()

# Get the working dataframe
raw_df = st.session_state.csv_data if st.session_state.csv_data is not None else st.session_state.rt_data

if raw_df is None or raw_df.empty:
    st.error("No data available. Please upload or fetch data.")
    st.stop()

# Calculate indicators
with st.spinner("Calculating indicators..."):
    df = calc_indicators(raw_df)

if df.empty:
    st.error("Not enough data to calculate indicators (need at least 50 rows).")
    st.stop()

r = df.iloc[-1]
prev = df.iloc[-2]
stock_name = st.session_state.fund_data.get('Name', st.session_state.ticker) if st.session_state.fund_data else st.session_state.ticker

# ── HEADER METRICS ─────────────────────────────────────────────
st.subheader(f"📌 {stock_name} ({st.session_state.ticker}) — {r['Date'].strftime('%Y-%m-%d')}")
price_change = r['Close'] - prev['Close']
pct_change = price_change / prev['Close'] * 100
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Price", f"${r['Close']:.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
col2.metric("High", f"${r['High']:.2f}")
col3.metric("Low", f"${r['Low']:.2f}")
col4.metric("Volume", f"{int(r['Volume']):,}")
col5.metric("ATR", f"${r['ATR']:.2f}")

st.markdown("---")

# ── RECOMMENDATION ─────────────────────────────────────────────
recommendation, signals, score = get_signal(df)
st.markdown(f"## Overall Signal: {recommendation}")
sig_cols = st.columns(3)
for i, s in enumerate(signals):
    sig_cols[i % 3].markdown(f"- {s}")

st.markdown("---")

# ── TABS ───────────────────────────────────────────────────────
tabs = st.tabs(["📈 Price Chart", "📊 Indicators", "🕯️ Candlestick", "📋 Fundamentals", "📥 Export"])

# TAB 1: Price Chart
with tabs[0]:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.05,
                        subplot_titles=["Price & MAs", "RSI", "MACD"])
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', line=dict(color='#2c3e50', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name='SMA20', line=dict(color='#3498db', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name='SMA50', line=dict(color='#e67e22', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name='BB Upper', line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name='BB Lower', line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(200,200,200,0.1)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='red', row=2, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='green', row=2, col=1)
    colors_macd = ['green' if v >= 0 else 'red' for v in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='MACD Hist', marker_color=colors_macd), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], name='Signal', line=dict(color='orange')), row=3, col=1)
    fig.update_layout(height=700, template='plotly_white', showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# TAB 2: Indicators Table
with tabs[1]:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📐 Trend")
        st.write(f"**SMA20:** ${r['SMA_20']:.2f}")
        st.write(f"**SMA50:** ${r['SMA_50']:.2f}")
        st.write(f"**SMA200:** ${r['SMA_200']:.2f}")
        st.write(f"**EMA20:** ${r['EMA_20']:.2f}")
        st.write(f"**ADX:** {r['ADX']:.1f} ({'Strong' if r['ADX']>25 else 'Weak'})")
        st.write(f"**Volatility:** {r['Volatility']:.2f}%")
    with col2:
        st.markdown("#### 🔄 Momentum")
        st.write(f"**RSI:** {r['RSI']:.1f} ({'Oversold' if r['RSI']<30 else 'Overbought' if r['RSI']>70 else 'Neutral'})")
        st.write(f"**MACD:** {r['MACD']:.3f}")
        st.write(f"**MACD Signal:** {r['MACD_Signal']:.3f}")
        st.write(f"**Stoch %K:** {r['Stoch_K']:.1f}")
        st.write(f"**Stoch %D:** {r['Stoch_D']:.1f}")
        st.write(f"**Williams %R:** {r['Williams_R']:.1f}")
    with col3:
        st.markdown("#### 🎯 Key Levels")
        st.write(f"**Pivot:** ${r['Pivot']:.2f}")
        st.write(f"**R1:** ${r['R1']:.2f}")
        st.write(f"**R2:** ${r['R2']:.2f}")
        st.write(f"**S1:** ${r['S1']:.2f}")
        st.write(f"**S2:** ${r['S2']:.2f}")
        st.write(f"**VWAP:** ${r['VWAP']:.2f}")
        st.write(f"**ATR:** ${r['ATR']:.2f}")
        st.write(f"**BB Width:** {r['BB_Width']:.1f}%")

    # Volume chart
    st.markdown("#### 📦 Volume Analysis")
    vol_avg = df['Volume'].rolling(20).mean()
    vfig = go.Figure()
    vfig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume',
        marker_color=['green' if df['Close'].iloc[i] >= df['Close'].iloc[i-1] else 'red' for i in range(len(df))]))
    vfig.add_trace(go.Scatter(x=df['Date'], y=vol_avg, name='20-day Avg', line=dict(color='orange')))
    vfig.update_layout(height=300, template='plotly_white')
    st.plotly_chart(vfig, use_container_width=True)

    # OBV chart
    st.markdown("#### 📈 On-Balance Volume (OBV)")
    ofig = px.line(df, x='Date', y='OBV', title='OBV Trend')
    ofig.update_layout(height=250, template='plotly_white')
    st.plotly_chart(ofig, use_container_width=True)

# TAB 3: Candlestick
with tabs[2]:
    period = st.selectbox("Period", ["1M", "3M", "6M", "1Y", "All"], index=2)
    n = {'1M': 21, '3M': 63, '6M': 126, '1Y': 252, 'All': len(df)}[period]
    dfc = df.tail(n)
    cfig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.05)
    cfig.add_trace(go.Candlestick(x=dfc['Date'], open=dfc['Open'], high=dfc['High'], low=dfc['Low'], close=dfc['Close'], name='OHLC'), row=1, col=1)
    cfig.add_trace(go.Scatter(x=dfc['Date'], y=dfc['BB_Upper'], name='BB Upper', line=dict(color='gray', dash='dot', width=1)), row=1, col=1)
    cfig.add_trace(go.Scatter(x=dfc['Date'], y=dfc['BB_Lower'], name='BB Lower', line=dict(color='gray', dash='dot', width=1), fill='tonexty', fillcolor='rgba(200,200,200,0.1)'), row=1, col=1)
    cfig.add_trace(go.Scatter(x=dfc['Date'], y=dfc['SMA_20'], name='SMA20', line=dict(color='blue', width=1)), row=1, col=1)
    vol_colors = ['green' if dfc['Close'].iloc[i] >= dfc['Open'].iloc[i] else 'red' for i in range(len(dfc))]
    cfig.add_trace(go.Bar(x=dfc['Date'], y=dfc['Volume'], name='Volume', marker_color=vol_colors), row=2, col=1)
    cfig.update_layout(height=650, xaxis_rangeslider_visible=False, template='plotly_white')
    st.plotly_chart(cfig, use_container_width=True)

# TAB 4: Fundamentals
with tabs[3]:
    fund = st.session_state.fund_data
    if fund:
        st.subheader(f"📋 Fundamentals — {fund.get('Name', st.session_state.ticker)}")
        st.write(f"**Sector:** {fund.get('Sector', 'N/A')}")
        fcols = st.columns(3)
        items = [(k, v) for k, v in fund.items() if k not in ('Name', 'Sector') and v is not None]
        for i, (k, v) in enumerate(items):
            fcols[i % 3].metric(k, f"{v:.2f}" if isinstance(v, float) else str(v))
    else:
        st.info("Fetch real-time data via Yahoo Finance to see fundamental metrics, or they are not available for uploaded files.")

# TAB 5: Export
with tabs[4]:
    st.subheader("📥 Export Data")
    col1, col2 = st.columns(2)
    with col1:
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button("📊 Download Full Data (CSV)", csv_buf.getvalue(),
                           f"{st.session_state.ticker}_analysis_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    with col2:
        # Summary report
        summary = f"""STOCK ANALYSIS REPORT
Ticker: {st.session_state.ticker}
Date: {datetime.now().strftime('%Y-%m-%d')}
Price: ${r['Close']:.2f} ({pct_change:+.2f}%)

SIGNAL: {recommendation}

KEY INDICATORS:
RSI: {r['RSI']:.1f}
MACD: {r['MACD']:.3f} / Signal: {r['MACD_Signal']:.3f}
ADX: {r['ADX']:.1f}
SMA20: ${r['SMA_20']:.2f}
SMA50: ${r['SMA_50']:.2f}

SUPPORT/RESISTANCE:
R2: ${r['R2']:.2f}
R1: ${r['R1']:.2f}
Pivot: ${r['Pivot']:.2f}
S1: ${r['S1']:.2f}
S2: ${r['S2']:.2f}

SIGNALS:
{chr(10).join(signals)}
"""
        st.download_button("📄 Download Summary (TXT)", summary,
                           f"{st.session_state.ticker}_summary_{datetime.now().strftime('%Y%m%d')}.txt", "text/plain")
