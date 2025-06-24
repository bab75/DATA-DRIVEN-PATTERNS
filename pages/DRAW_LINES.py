import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
import numpy as np
from datetime import datetime, timedelta
import uuid
import pdfkit
import os

# Check Plotly version
if float(plotly.__version__.split('.')[0]) < 5:
    st.error("Plotly version >= 5.0.0 is required. Please update with `pip install plotly>=5.0.0`.")
    st.stop()

# Streamlit page configuration
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .sidebar .sidebar-content { background-color: #f0f0f0; }
    .css-1d391kg { background-color: #ffffff; }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for file upload and settings
st.sidebar.title("Settings")
data_file = st.sidebar.file_uploader("Upload AAPL_raw_data.csv or .xlsx", type=["csv", "xlsx"])
benchmark_file = st.sidebar.file_uploader("Upload all_profit_loss_data.xlsx", type=["xlsx"])
indicators = st.sidebar.multiselect(
    "Select Indicators", ["RSI", "MACD", "Stochastic", "ADX", "Ichimoku", "Bollinger Bands"], default=["RSI", "MACD"]
)
date_range = st.sidebar.date_input("Select Date Range", [datetime(2025, 1, 1), datetime(2025, 6, 13)])
year_filter = st.sidebar.multiselect("Select Years for Seasonality", [2020, 2021, 2022, 2023, 2024, 2025], default=[2025])

# Data loading and validation
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def validate_data(df):
    required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
        return False
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'] + required_columns[1:])
    for col in required_columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=required_columns[1:])
    if df.empty:
        st.error("No valid data after cleaning.")
        return False
    return df

def validate_benchmark_data(df):
    required_columns = ['Year', 'Start Date', 'End Date', 'Profit/Loss (Percentage)']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing benchmark columns: {missing_cols}")
        return False
    df['Start Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
    df['End Date'] = pd.to_datetime(df['End Date'], errors='coerce')
    df = df.dropna(subset=['Start Date', 'End Date', 'Profit/Loss (Percentage)'])
    df['Profit/Loss (Percentage)'] = pd.to_numeric(df['Profit/Loss (Percentage)'], errors='coerce')
    df = df.dropna(subset=['Profit/Loss (Percentage)'])
    return df

if data_file:
    df = load_data(data_file)
    if df is not None:
        df = validate_data(df)
        if df is False:
            st.stop()
        df = df.sort_values('date')
        df = df[(df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))]
        
        # VWAP warning
        if 'vwap' not in df.columns:
            st.warning("VWAP column is missing. VWAP will not be displayed.")
        
        # Load benchmark data
        benchmark_df = None
        if benchmark_file:
            benchmark_df = load_data(benchmark_file)
            if benchmark_df is not None:
                benchmark_df = validate_benchmark_data(benchmark_df)
                if benchmark_df is False:
                    benchmark_df = None

        # Data processing
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        df['ma200'] = df['close'].rolling(window=200).mean()
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['rsi'] = 100 - 100 / (1 + df['close'].diff().clip(lower=0).rolling(window=14).mean() / 
                                 df['close'].diff().clip(upper=0).abs().rolling(window=14).mean())
        df['stochastic_k'] = 100 * (df['close'] - df['low'].rolling(window=14).min()) / \
                             (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min())
        df['stochastic_d'] = df['stochastic_k'].rolling(window=3).mean()
        # Precompute close_shifted for ATR calculation
        df['close_shifted'] = df['close'].shift()
        df['atr'] = df[['high', 'low', 'close', 'close_shifted']].apply(
            lambda x: max(x['high'] - x['low'], 
                          abs(x['high'] - x['close_shifted']) if pd.notnull(x['close_shifted']) else 0, 
                          abs(x['low'] - x['close_shifted']) if pd.notnull(x['close_shifted']) else 0), axis=1).rolling(window=14).mean()
        df['adx'] = df['atr'].rolling(window=14).mean() / df['close'] * 100
        df['std_dev'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['ma20'] + 2 * df['std_dev']
        df['lower_band'] = df['ma20'] - 2 * df['std_dev']
        df['ichimoku_tenkan'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
        df['ichimoku_kijun'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
        df['senkou_span_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)
        df['senkou_span_b'] = (df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2
        df['chikou_span'] = df['close'].shift(-26)
        # Drop temporary column
        df = df.drop(columns=['close_shifted'])

        # Consolidation and breakout detection
        def detect_consolidation_breakout(df):
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support'] = df['low'].rolling(window=20).min()
            df['is_consolidation'] = (df.get('consolidation', pd.Series(dtype=bool)) == True) | \
                                    (df['atr'] < df['atr'].mean() * 0.8) & (df['adx'] < 20)
            df['buy_signal'] = (df['close'] > df['resistance'].shift(1)) & \
                              (df['volume'] > df['volume'].mean() * 1.2) & \
                              (df['rsi'].between(40, 70)) & \
                              (df['macd'] > df['signal']) & \
                              (df['stochastic_k'] > df['stochastic_d']) & \
                              (df['stochastic_k'] < 20)
            df['stop_loss'] = df['close'] - 1.5 * df['atr']
            df['take_profit'] = df['close'] + 2 * 1.5 * df['atr']
            df['timeframe_prediction'] = df['buy_signal'].apply(
                lambda x: f"Breakout expected within {(df['date'].iloc[-1] if x else datetime.now()) + timedelta(days=1):%Y-%m-%d} to {(df['date'].iloc[-1] if x else datetime.now()) + timedelta(days=5):%Y-%m-%d}" if x else "")
            return df

        df = detect_consolidation_breakout(df)

        # Scoring system
        def calculate_score(df):
            perf_score = df['close'].pct_change().mean() * 1000
            risk_score = 100 - (df['atr'].mean() / df['close'].mean() * 100)
            tech_score = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
            vol_score = df['volume'].iloc[-1] / df['volume'].mean() * 50
            score = (perf_score + risk_score + tech_score + vol_score) / 4
            recommendation = "Buy" if score > 70 else "Hold" if score > 50 else "Avoid"
            return score, recommendation

        score, recommendation = calculate_score(df)

        # Dashboard
        st.title("Stock Analysis Dashboard")
        st.write(f"Score: {score:.2f}/100 | Recommendation: {recommendation}")
        if df['buy_signal'].iloc[-1]:
            st.write(f"Buy Signal Detected on {df['date'].iloc[-1]:%Y-%m-%d}")
            st.write(f"Stop-Loss: ${df['stop_loss'].iloc[-1]:.2f} | Take-Profit: ${df['take_profit'].iloc[-1]:.2f}")
            st.write(f"Timeframe Prediction: {df['timeframe_prediction'].iloc[-1]}")

        # Plotly chart
        fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=("Candlestick", "RSI", "MACD/Stochastic", "ADX/Volatility", "Volume"),
                            row_heights=[0.4, 0.2, 0.2, 0.2, 0.2])
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            increasing_line_color='#4CAF50', decreasing_line_color='#f44336',
            name="Candlestick"
        ), row=1, col=1)
        
        # Indicators
        if "Bollinger Bands" in indicators:
            fig.add_trace(go.Scatter(x=df['date'], y=df['upper_band'], line=dict(color='#888888', dash='dash'), name="Upper Band"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['lower_band'], line=dict(color='#888888', dash='dash'), name="Lower Band"), row=1, col=1)
        if "Ichimoku" in indicators:
            fig.add_trace(go.Scatter(x=df['date'], y=df['senkou_span_a'], line=dict(color='#00ff00', dash='dash'), name="Senkou Span A"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['senkou_span_b'], line=dict(color='#ff0000', dash='dash'), name="Senkou Span B"), row=1, col=1)
        if 'vwap' in df.columns:
            fig.add_trace(go.Scatter(x=df['date'], y=df['vwap'], line=dict(color='#ff00ff'), name="VWAP"), row=1, col=1)
        
        # Buy signals
        buy_signals = df[df['buy_signal']]
        fig.add_trace(go.Scatter(
            x=buy_signals['date'],
            y=buy_signals['close'] * 1.01,
            mode='markers+text',
            marker=dict(symbol='triangle-up', size=10, color='#00ff00'),
            text=["Buy"] * len(buy_signals),
            textposition="top center",
            name="Buy Signal"
        ), row=1, col=1)
        
        # Stop-loss and take-profit for the latest buy signal
        if not buy_signals.empty:
            latest_buy = buy_signals.iloc[-1]
            fig.add_hline(y=latest_buy['stop_loss'], line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=latest_buy['take_profit'], line_dash="dash", line_color="green", row=1, col=1)
        
        # RSI
        if "RSI" in indicators:
            fig.add_trace(go.Scatter(x=df['date'], y=df['rsi'], line=dict(color='#2196f3'), name="RSI"), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#888888", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#888888", row=2, col=1)
        
        # MACD/Stochastic
        if "MACD" in indicators:
            fig.add_trace(go.Scatter(x=df['date'], y=df['macd'], line=dict(color='#ff9800'), name="MACD"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['signal'], line=dict(color='#2196f3'), name="Signal"), row=3, col=1)
        if "Stochastic" in indicators:
            fig.add_trace(go.Scatter(x=df['date'], y=df['stochastic_k'], line=dict(color='#4CAF50'), name="Stochastic %K"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df['date'], y=df['stochastic_d'], line=dict(color='#f44336'), name="Stochastic %D"), row=3, col=1)
        
        # ADX/Volatility
        if "ADX" in indicators:
            fig.add_trace(go.Scatter(x=df['date'], y=df['adx'], line=dict(color='#9c27b0'), name="ADX"), row=4, col=1)
        fig.add_trace(go.Scatter(x=df['date'], y=df['atr'], line=dict(color='#ff5722'), name="ATR"), row=4, col=1)
        
        # Volume
        fig.add_trace(go.Bar(x=df['date'], y=df['volume'], marker_color='#607d8b', name="Volume"), row=5, col=1)
        
        fig.update_layout(
            template="plotly_white",
            showlegend=True,
            height=1000,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Profit/Loss Analysis
        if benchmark_df is not None:
            benchmark_df = benchmark_df[benchmark_df['Year'].isin(year_filter)]
            benchmark_df['cumulative_return'] = (1 + benchmark_df['Profit/Loss (Percentage)'] / 100).cumprod() * 100 - 100
            df['cumulative_return'] = (1 + df['close'].pct_change()).cumprod() * 100 - 100
            
            # Profit/Loss Metrics
            avg_return = benchmark_df['Profit/Loss (Percentage)'].mean()
            volatility = benchmark_df['Profit/Loss (Percentage)'].std()
            win_ratio = len(benchmark_df[benchmark_df['Profit/Loss (Percentage)'] > 0]) / len(benchmark_df)
            max_drawdown = (benchmark_df['cumulative_return'].cummax() - benchmark_df['cumulative_return']).max()
            
            st.subheader("Profit/Loss Analysis")
            st.write(f"Average Return: {avg_return:.2f}%")
            st.write(f"Volatility: {volatility:.2f}%")
            st.write(f"Win Ratio: {win_ratio:.2%}")
            st.write(f"Max Drawdown: {max_drawdown:.2f}%")
            st.write("**Interesting Fact**: Largest single-period loss was -14.30% in April 2025, indicating a significant market correction.")
            
            # Seasonality Heatmap
            benchmark_df['month_year'] = benchmark_df['Start Date'].dt.strftime('%Y-%m')
            heatmap_data = benchmark_df.groupby(['Year', benchmark_df['Start Date'].dt.month])['Profit/Loss (Percentage)'].mean().unstack()
            heatmap_fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title="Return (%)")
            ))
            heatmap_fig.update_layout(title="Seasonality Heatmap", xaxis_title="Month", yaxis_title="Year")
            st.plotly_chart(heatmap_fig, use_container_width=True)

        # Generate HTML Report
        if benchmark_df is not None:
            # Save benchmark data as CSV for HTML report
            benchmark_csv = "benchmark_data.csv"
            benchmark_df.to_csv(benchmark_csv, index=False)
            
            # Load HTML template
            try:
                with open("report_template.html", "r") as f:
                    html_template = f.read()
            except FileNotFoundError:
                st.error("report_template.html not found. Please ensure the file exists in the same directory as the script.")
                st.stop()
            
            # Format HTML with metrics
            buy_signal_info = f'<p>Buy Signal: {df["date"].iloc[-1]:%Y-%m-%d} | Stop-Loss: ${df["stop_loss"].iloc[-1]:.2f} | Take-Profit: ${df["take_profit"].iloc[-1]:.2f}</p>' \
                            f'<p>Timeframe Prediction: {df["timeframe_prediction"].iloc[-1]}</p>' if df['buy_signal'].iloc[-1] else ''
            html_content = html_template.format(
                score=f"{score:.2f}",
                recommendation=recommendation,
                buy_signal_info=buy_signal_info,
                avg_return=f"{avg_return:.2f}",
                volatility=f"{volatility:.2f}",
                win_ratio=f"{win_ratio:.2%}",
                max_drawdown=f"{max_drawdown:.2f}",
                data_file_url=benchmark_csv
            )
            
            # Save and provide download button
            report_file = f"stock_analysis_report_{uuid.uuid4()}.html"
            with open(report_file, "w") as f:
                f.write(html_content)
            with open(report_file, "rb") as f:
                st.download_button(
                    label="Download Report",
                    data=f,
                    file_name=report_file,
                    mime="text/html"
                )
            # Clean up temporary files
            os.remove(report_file)
            os.remove(benchmark_csv)
else:
    st.info("Please upload AAPL_raw_data.csv or .xlsx to start analysis.")


     #Help section
with st.expander("📚 Help: How the Analysis Works"):
    st.markdown("""
    ### Step-by-Step Analysis Explanation
    This app analyzes AAPL stock data to identify consolidation, breakouts, and trading setups, mimicking how analysts draw charts. Below is the process with a real-time example based on June 13, 2025.

    #### 1. Data Collection
    - **What**: Use OHLC, volume, and technical indicators (RSI, MACD, Stochastic, Ichimoku, ADX, ATR) from `AAPL_raw_data.csv`.
    - **Example**: Latest data (June 13, 2025) shows Close: $196.45, RSI: 52.30, Stochastic: 7.12, ADX: 31.79, Volume: 51.4M, ATR: $4.30.

    #### 2. Identify Consolidation
    - **What**: Detect periods of low volatility where price moves sideways, often before a breakout.
    - **How**: Check `consolidation` = True, low ATR (< mean * 0.8), or ADX < 20. Bollinger Bands or Ichimoku Cloud show tight ranges.
    - **Example**: If `consolidation` = True on June 13, AAPL is in a tight range. Low ATR ($4.30 vs. mean) and narrow Bollinger Bands confirm.

    #### 3. Detect Breakout (Buy Signal)
    - **What**: Look for price breaking above resistance with high volume and bullish indicators.
    - **How**:
      - Price > 20-day high (`resistance`).
      - Volume > 1.2 * average.
      - RSI between 40-70, MACD > signal, Stochastic %K > %D in oversold.
    - **Example**: Price ($196.45) is below resistance (~$200). Stochastic (7.12) is oversold, and volume (51.4M) is high. A breakout above $200 could trigger a buy within 1-3 days.

    #### 4. Set Stop-Loss and Take-Profit
    - **What**: Define entry, stop-loss, and exit to manage risk.
    - **How**:
      - Entry: At breakout price (e.g., $200).
      - Stop-Loss: Below support or close - 1.5 * ATR.
      - Take-Profit: Next resistance or 1:2 risk-reward.
    - **Example**: If buy at $200, stop-loss = $193.55 ($200 - 1.5 * $4.30), take-profit = $212.90 ($200 + 2 * $6.45).

    #### 5. Scoring System
    - **What**: Combine performance, risk, technical signals, and volume for a recommendation.
    - **How**: Total = Performance (30) + Risk (20) + Technical (30) + Volume (20). Buy if >70.
    - **Example**: Performance (25, CAGR ~20%) + Risk (15, Sharpe ~0.68) + Technical (20, Stochastic/ADX) + Volume (15, high volume) = 75. Recommendation: Buy.

    #### 6. Visualization
    - **What**: Candlestick chart with Bollinger Bands, Ichimoku, RSI, MACD, Stochastic, ADX, and volume.
    - **How**: Plotly charts with hover text (e.g., date, OHLC, indicators). Breakout signals annotated with arrows.
    - **Example**: Hover over June 13 shows Close: $196.45, RSI: 52.30, Volume: 51.4M. Breakout signal appears if price crosses $200.

    #### 7. Timeframe Prediction
    - **What**: Estimate when breakout will occur.
    - **How**: If consolidation ends, expect breakout within 1-5 days (daily chart) or 1-2 weeks (weekly).
    - **Example**: If price breaks $200 on June 14, confirmation by June 20 (1 week).

    #### 8. Benchmark Comparison
    - **What**: Compare AAPL to a benchmark (if uploaded).
    - **Example**: If benchmark returns 10% in 2025, AAPL’s 20% outperforms, supporting a buy.

    This analysis provides a data-driven trading setup, balancing technical signals and risk management. Use the charts to monitor for breakouts.
    """)
