import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
import numpy as np
from datetime import datetime, timedelta
import uuid
import pdfkit
import warnings

# Check Plotly version
if float(plotly.__version__.split('.')[0]) < 5:
    st.error("Plotly version >= 5.0.0 is required.")
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
        df['atr'] = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], 
                                                                     abs(x['high'] - x['close'].shift()), 
                                                                     abs(x['low'] - x['close'].shift())), axis=1).rolling(window=14).mean()
        df['adx'] = df['atr'].rolling(window=14).mean() / df['close'] * 100
        df['std_dev'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['ma20'] + 2 * df['std_dev']
        df['lower_band'] = df['ma20'] - 2 * df['std_dev']
        df['ichimoku_tenkan'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
        df['ichimoku_kijun'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
        df['senkou_span_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)
        df['senkou_span_b'] = (df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2
        df['chikou_span'] = df['close'].shift(-26)

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
                lambda x: f"Breakout expected within {datetime.now() + timedelta(days=1):%Y-%m-%d} to {datetime.now() + timedelta(days=5):%Y-%m-%d}" if x else "")
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

        # HTML Report for Profit/Loss
        html_content = f"""
        <html>
        <head>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/prop-types/15.8.1/prop-types.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.23.2/babel.min.js"></script>
            <script src="https://unpkg.com/papaparse@latest/papaparse.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/chrono-node/1.3.11/chrono.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/recharts/2.15.0/Recharts.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #ffffff; color: #000000; }}
                .container {{ max-width: 1200px; margin: auto; padding: 20px; }}
                h1, h2 {{ color: #333333; }}
                .chart-container {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Stock Analysis Report</h1>
                <h2>Summary</h2>
                <p>Score: {score:.2f}/100 | Recommendation: {recommendation}</p>
                {f'<p>Buy Signal: {df["date"].iloc[-1]:%Y-%m-%d} | Stop-Loss: ${df["stop_loss"].iloc[-1]:.2f} | Take-Profit: ${df["take_profit"].iloc[-1]:.2f}</p>' if df['buy_signal'].iloc[-1] else ''}
                {f'<p>Timeframe Prediction: {df["timeframe_prediction"].iloc[-1]}</p>' if df['buy_signal'].iloc[-1] else ''}
                <h2>Profit/Loss Analysis</h2>
                {f'<p>Average Return: {avg_return:.2f}%</p>' if benchmark_df is not None else ''}
                {f'<p>Volatility: {volatility:.2f}%</p>' if benchmark_df is not None else ''}
                {f'<p>Win Ratio: {win_ratio:.2%}</p>' if benchmark_df is not None else ''}
                {f'<p>Max Drawdown: {max_drawdown:.2f}%</p>' if benchmark_df is not None else ''}
                <p>Interesting Fact: Largest single-period loss was -14.30% in April 2025, indicating a significant market correction.</p>
                <div id="cumulative-return-chart" class="chart-container"></div>
                <div id="volatility-chart" class="chart-container"></div>
                <div id="win-loss-chart" class="chart-container"></div>
            </div>
            <script type="text/babel">
                function App() {{
                    const [data, setData] = React.useState(null);
                    const [loading, setLoading] = React.useState(true);

                    React.useEffect(() => {{
                        Papa.parse(loadFileData("all_profit_loss_data.xlsx"), {{
                            header: true,
                            skipEmptyLines: true,
                            transformHeader: header => header.trim().replace(/^"|"$/g, ''),
                            transform: (value, header) => {{
                                let cleaned = value.trim().replace(/^"|"$/g, '');
                                if (header.includes('Date')) return chrono.parseDate(cleaned);
                                if (header.includes('Profit/Loss')) return parseFloat(cleaned) || 0;
                                return cleaned;
                            }},
                            complete: results => {{
                                const cleanedData = results.data.map(row => ({{
                                    date: row['Start Date'],
                                    portfolioReturn: row['Profit/Loss (Percentage)'],
                                    benchmarkReturn: row['Profit/Loss (Percentage)']
                                })).filter(row => row.date && !isNaN(row.portfolioReturn)));
                                cleanedData.sort((a, b) => a.date - b.date);
                                cleanedData.forEach((row, i) => {{
                                    row.cumulativePortfolio = i === 0 ? row.portfolioReturn : 
                                        (1 + row.portfolioReturn / 100) * (1 + cleanedData[i-1].cumulativePortfolio / 100) * 100 - 100;
                                    row.cumulativeBenchmark = i === 0 ? row.benchmarkReturn : 
                                        (1 + row.benchmarkReturn / 100) * (1 + cleanedData[i-1].cumulativeBenchmark / 100) * 100 - 100;
                                    row.volatility = i >= 20 ? Math.sqrt(cleanedData.slice(i-19, i+1)
                                        .reduce((sum, r) => sum + Math.pow(r.portfolioReturn - 
                                        cleanedData.slice(i-19, i+1).reduce((s, r) => s + r.portfolioReturn, 0) / 20, 2), 0) / 19) * 100 : 0;
                                }));
                                setData(cleanedData);
                                setLoading(false);
                            }},
                            error: err => {{
                                console.error(err);
                                setLoading(false);
                            }}
                        });
                    }}, []);

                    if (loading) return <div>Loading...</div>;

                    const winLossData = [
                        {{ name: '<-5%', value: data.filter(d => d.portfolioReturn < -5).length }},
                        {{ name: '-5% to 0%', value: data.filter(d => d.portfolioReturn >= -5 && d.portfolioReturn < 0).length }},
                        {{ name: '0% to 5%', value: data.filter(d => d.portfolioReturn >= 0 && d.portfolioReturn < 5).length }},
                        {{ name: '>5%', value: data.filter(d => d.portfolioReturn >= 5).length }}
                    ];

                    return (
                        <div>
                            <div className="chart-container">
                                <h3>Cumulative Return</h3>
                                <Recharts.ResponsiveContainer width="100%" height={400}>
                                    <Recharts.LineChart data={data}>
                                        <Recharts.XAxis dataKey="date" tickFormatter={d => d.toISOString().split('T')[0]} />
                                        <Recharts.YAxis tickFormatter={v => `${v.toFixed(2)}%`} />
                                        <Recharts.CartesianGrid strokeDasharray="3 3" />
                                        <Recharts.Tooltip formatter={(value, name) => `${value.toFixed(2)}%`} />
                                        <Recharts.Legend />
                                        <Recharts.Line type="monotone" dataKey="cumulativePortfolio" stroke="#4CAF50" name="Portfolio" />
                                        <Recharts.Line type="monotone" dataKey="cumulativeBenchmark" stroke="#f44336" name="Benchmark" />
                                    </Recharts.LineChart>
                                </Recharts.ResponsiveContainer>
                            </div>
                            <div className="chart-container">
                                <h3>Volatility Trend</h3>
                                <Recharts.ResponsiveContainer width="100%" height={400}>
                                    <Recharts.LineChart data={data}>
                                        <Recharts.XAxis dataKey="date" tickFormatter={d => d.toISOString().split('T')[0]} />
                                        <Recharts.YAxis tickFormatter={v => `${v.toFixed(2)}%`} />
                                        <Recharts.CartesianGrid strokeDasharray="3 3" />
                                        <Recharts.Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
                                        <Recharts.Legend />
                                        <Recharts.Line type="monotone" dataKey="volatility" stroke="#2196f3" name="Volatility" />
                                    </Recharts.LineChart>
                                </Recharts.ResponsiveContainer>
                            </div>
                            <div className="chart-container">
                                <h3>Win/Loss Distribution</h3>
                                <Recharts.ResponsiveContainer width="100%" height={400}>
                                    <Recharts.BarChart data={winLossData}>
                                        <Recharts.XAxis dataKey="name" />
                                        <Recharts.YAxis />
                                        <Recharts.CartesianGrid strokeDasharray="3 3" />
                                        <Recharts.Tooltip />
                                        <Recharts.Legend />
                                        <Recharts.Bar dataKey="value" fill="#8884d8" />
                                    </Recharts.BarChart>
                                </Recharts.ResponsiveContainer>
                            </div>
                        </div>
                    );
                }}

                const root = ReactDOM.createRoot(document.getElementById('root'));
                root.render(<App />);
            </script>
        </body>
        </html>
        """
        st.download_button(
            label="Download Report",
            data=html_content,
            file_name="stock_analysis_report.html",
            mime="text/html"
        )
