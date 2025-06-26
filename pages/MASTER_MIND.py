import streamlit as st
import pandas as pd
import plotly.express as px
import uuid
from datetime import datetime
import io

# Streamlit app configuration
st.set_page_config(page_title="Stock Technical Analysis", layout="wide", page_icon="📈")
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #1e3a8a; color: white; border-radius: 8px;}
    .stSelectbox {background-color: #e5e7eb; border-radius: 8px;}
    .report-container {background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    h1 {color: #1e3a8a; font-family: 'Arial', sans-serif;}
    h2, h3 {color: #374151; font-family: 'Arial', sans-serif;}
    </style>
""", unsafe_allow_html=True)

# Function to analyze the stock data
def analyze_stock_data(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    stock_name = "KRRO"  # Assuming stock name; can be extracted from filename if provided

    # Extract key indicators
    price = latest['Close']
    sma_20, sma_50, sma_200 = latest['SMA_20'], latest['SMA_50'], latest['SMA_200']
    ema_20, ema_50, ema_200 = latest['EMA_20'], latest['EMA_50'], latest['EMA_200']
    rsi = latest['RSI']
    macd, macd_signal, macd_hist = latest['MACD'], latest['MACD_Signal'], latest['MACD_Histogram']
    bb_upper, bb_middle, bb_lower = latest['BB_Upper'], latest['BB_Middle'], latest['BB_Lower']
    stoch_k, williams_r = latest['Stoch_K'], latest['Williams_R']
    cci = latest['CCI']
    momentum = latest['Momentum']
    roc = latest['ROC']
    obv = latest['OBV']
    volume = latest['Volume']
    pivot, r1, s1 = latest['Pivot'], latest['R1'], latest['S1']
    fib_236, fib_382, fib_618 = latest['Fib_236'], latest['Fib_382'], latest['Fib_618']
    ichimoku_tenkan, ichimoku_kijun = latest['Ichimoku_Tenkan'], latest['Ichimoku_Kijun']
    ichimoku_senkou_a, ichimoku_senkou_b = latest['Ichimoku_Senkou_A'], latest['Ichimoku_Senkou_B']
    psar = latest['PSAR']

    # Determine trend
    trend = "Bearish" if price < sma_20 and price < sma_50 else "Bullish" if price > sma_20 and price > sma_50 else "Neutral"

    # Generate reports
    quick_scan = f"""
    ### Quick Scan: {stock_name} ({latest['Date']})
    - **Price**: ${price:.2f} ({trend} trend)
    - **Support/Resistance**: Support at ${s1:.2f}, Resistance at ${r1:.2f}
    - **RSI**: {rsi:.2f} ({'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'})
    - **Recommendation**: {'Buy near support ($' + f'{s1:.2f}) for a bounce to ${r1:.2f}' if rsi < 30 else 'Wait for breakout above $' + f'{sma_20:.2f}'}
    """

    moderate_detail = f"""
    ### Moderate Detail: {stock_name} ({latest['Date']})
    - **Price Trend**: ${price:.2f}, {trend} (Price vs. SMA20: ${sma_20:.2f}, SMA50: ${sma_50:.2f})
    - **Momentum**:
      - RSI: {rsi:.2f} ({'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'})
      - MACD: {macd:.2f} (Signal: {macd_signal:.2f}, {'Bearish' if macd < macd_signal else 'Bullish'})
    - **Bollinger Bands**: Price near {'lower' if price < bb_middle else 'upper'} band (Lower: ${bb_lower:.2f}, Upper: ${bb_upper:.2f})
    - **Key Levels**: Support: ${s1:.2f}, Resistance: ${r1:.2f}, Fib 61.8%: ${fib_618:.2f}
    - **Recommendation**: 
      - Traders: {'Buy near $' + f'{s1:.2f} for bounce to ${r1:.2f}' if rsi < 30 or price < bb_middle else 'Wait for breakout above $' + f'{r1:.2f}'}
      - Investors: Confirm trend reversal above ${sma_20:.2f}.
    """

    in_depth = f"""
    ### In-Depth Analysis: {stock_name} ({latest['Date']})
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
    
    #### Ichimoku Cloud
    - **Tenkan/Kijun**: {ichimoku_tenkan:.2f}/{ichimoku_kijun:.2f} ({'Bullish' if ichimoku_tenkan > ichimoku_kijun else 'Bearish'})
    - **Cloud**: Price {'below' if price < ichimoku_senkou_a else 'above'} Senkou A (${ichimoku_senkou_a:.2f}), Senkou B (${ichimoku_senkou_b:.2f})
    
    #### Volume
    - **OBV**: {obv:,.0f} ({'Declining' if obv < prev['OBV'] else 'Rising'})
    - **Volume**: {volume:,.0f} (Recent trend: {'Low' if volume < df['Volume'].mean() else 'High'})
    
    #### Recommendation
    - **Conservative Investors**: Wait for price to break above SMA20 (${sma_20:.2f}) or Senkou A (${ichimoku_senkou_a:.2f}).
    - **Traders**: {'Buy near support ($' + f'{s1:.2f}) for bounce to ${r1:.2f}' if rsi < 30 or stoch_k < 20 else 'Wait for breakout above $' + f'{r1:.2f}'}.
    - **Key Levels**: Support: ${s1:.2f}, Resistance: ${r1:.2f}, Fib 61.8%: ${fib_618:.2f}.
    - **Risk**: {'High' if volume < df['Volume'].mean() else 'Moderate'} due to {'low volume' if volume < df['Volume'].mean() else 'market volatility'}.
    """

    return quick_scan, moderate_detail, in_depth, stock_name, df

# Streamlit UI
st.title("📈 Stock Technical Analysis")
st.markdown("Upload a CSV file with technical indicators to generate a stock analysis report.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)
        required_columns = ['Date', 'Close', 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'BB_Upper', 'BB_Middle', 'BB_Lower', 'Stoch_K', 'Williams_R', 'CCI', 'Momentum', 'ROC', 'OBV', 'Volume', 'Pivot', 'R1', 'S1', 'Fib_236', 'Fib_382', 'Fib_618', 'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_A', 'Ichimoku_Senkou_B', 'PSAR']
        if not all(col in df.columns for col in required_columns):
            st.error("CSV must contain required columns: " + ", ".join(required_columns))
        else:
            # Convert Date to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Analyze data
            quick_scan, moderate_detail, in_depth, stock_name, df = analyze_stock_data(df)
            
            # Report selection
            report_type = st.selectbox("Select Report Type", ["Quick Scan", "Moderate Detail", "In-Depth Analysis", "Visual Summary"])
            
            # Display report
            st.markdown("<div class='report-container'>", unsafe_allow_html=True)
            if report_type == "Quick Scan":
                st.markdown(quick_scan)
            elif report_type == "Moderate Detail":
                st.markdown(moderate_detail)
            elif report_type == "In-Depth Analysis":
                st.markdown(in_depth)
            else:
                # Visual Summary
                st.markdown(f"### Visual Summary: {stock_name} ({df['Date'].iloc[-1].strftime('%Y-%m-%d')})")
                st.write(f"**Price**: ${df['Close'].iloc[-1]:.2f}")
                st.write(f"**Trend**: {'Bearish' if df['Close'].iloc[-1] < df['SMA_20'].iloc[-1] else 'Bullish'}")
                
                # Price trend chart
                fig = px.line(df.tail(30), x='Date', y='Close', title='Price Trend (Last 30 Days)')
                fig.add_scatter(x=df['Date'], y=df['SMA_20'], name='SMA20', line=dict(color='orange'))
                fig.add_scatter(x=df['Date'], y=df['SMA_50'], name='SMA50', line=dict(color='green'))
                st.plotly_chart(fig, use_container_width=True)
                
                # RSI chart
                fig_rsi = px.line(df.tail(30), x='Date', y='RSI', title='RSI (Last 30 Days)')
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                st.markdown(f"- **Support**: ${df['S1'].iloc[-1]:.2f}, **Resistance**: ${df['R1'].iloc[-1]:.2f}")
                st.markdown(f"- **RSI**: {df['RSI'].iloc[-1]:.2f} ({'Oversold' if df['RSI'].iloc[-1] < 30 else 'Overbought' if df['RSI'].iloc[-1] > 70 else 'Neutral'})")
                st.markdown(f"- **Recommendation**: {'Buy near support' if df['RSI'].iloc[-1] < 30 else 'Wait for breakout'}")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Download report
            report_content = quick_scan if report_type == "Quick Scan" else moderate_detail if report_type == "Moderate Detail" else in_depth
            buffer = io.StringIO()
            buffer.write(report_content)
            st.download_button(
                label="Download Report",
                data=buffer.getvalue(),
                file_name=f"{stock_name}_{report_type.replace(' ', '_')}_{df['Date'].iloc[-1].strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
else:
    st.info("Please upload a CSV file to begin analysis.")
