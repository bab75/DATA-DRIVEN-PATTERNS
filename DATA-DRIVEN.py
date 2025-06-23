import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

# Set Streamlit page config
st.set_page_config(page_title="Stock Pattern Analyzer", layout="wide")

# Sidebar: Control Panel
st.sidebar.header("Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
days_to_analyze = st.sidebar.number_input("Days to Analyze (1-365)", min_value=1, max_value=365, value=60)
days_to_predict = st.sidebar.number_input("Days to Predict (1-30)", min_value=1, max_value=30, value=5)  # Set to 5 days
analysis_mode = st.sidebar.radio("Analysis Mode", ["Raw Data Only", "Use Technical Indicators"])
show_indicators = st.sidebar.checkbox("Show Technical Indicators on Chart", value=False)
show_candlestick = st.sidebar.checkbox("Show Candlestick Chart", value=False)
initial_investment = st.sidebar.number_input("Initial Investment ($)", min_value=1.0, value=100.0)
run_analysis = st.sidebar.button("Run Analysis")

# Function to load and preprocess data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        required_columns = ['date', 'close']
        if analysis_mode == "Use Technical Indicators":
            required_columns.extend(['macd', 'rsi', 'atr', 'vwap'])
        if show_candlestick:
            required_columns.extend(['open', 'high', 'low'])
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            st.error(f"Missing required columns: {', '.join(missing)}")
            return None
        if len(df) < days_to_analyze:
            st.error(f"Dataset has {len(df)} rows, but at least {days_to_analyze} are required.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Function to compute similarity between two series
def compute_similarity(series1, series2, method='pearson'):
    if len(series1) != len(series2) or len(series1) == 0:
        return np.nan
    try:
        if method == 'pearson':
            return pearsonr(series1, series2)[0]
        else:  # cosine
            return cosine_similarity([series1], [series2])[0][0]
    except:
        return np.nan

# Function to extract pattern and compare
def find_similar_patterns(df, days_analyze, days_predict, mode):
    recent_pattern = df.tail(days_analyze).copy()
    recent_date = recent_pattern['date'].iloc[-1]
    recent_close = recent_pattern['close'].pct_change().fillna(0)
    
    matches = []
    
    # Define features based on mode
    if mode == "Raw Data Only":
        features = ['close']
    else:
        features = ['close', 'macd', 'rsi', 'atr', 'vwap']
    
    # Slide through historical data with ±5-day window
    for year in range(df['date'].dt.year.min(), df['date'].dt.year.max() + 1):
        for day_offset in range(-5, 6):  # ±5-day window
            target_date = recent_date.replace(year=year) + timedelta(days=day_offset)
            if target_date in df['date'].values:
                idx = df[df['date'] == target_date].index[0]
                if idx >= days_analyze - 1:
                    historical = df.iloc[idx-days_analyze+1:idx+1].copy()
                    if len(historical) == len(recent_pattern):
                        similarity_scores = []
                        for feature in features:
                            hist_series = historical[feature].pct_change().fillna(0)
                            recent_series = recent_pattern[feature].pct_change().fillna(0)
                            score = compute_similarity(hist_series, recent_series)
                            if not np.isnan(score):
                                similarity_scores.append(score)
                        if similarity_scores:
                            avg_similarity = np.nanmean(similarity_scores)
                            if idx + days_predict < len(df):
                                forward_data = df.iloc[idx:idx+days_predict+1]
                                forward_return = (forward_data['close'].iloc[-1] / forward_data['close'].iloc[0] - 1) * 100
                                matches.append({
                                    'year': year,
                                    'target_date': target_date,
                                    'similarity': avg_similarity,
                                    'forward_return': forward_return,
                                    'historical_pattern': historical,
                                    'forward_data': forward_data
                                })
    
    matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)
    return matches, recent_pattern

# Function to create interactive Plotly chart
def create_chart(df, recent_pattern, matches, show_indicators, show_candlestick):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        subplot_titles=['Price Patterns', 'Volume', 'Technical Indicators'],
                        row_heights=[0.5, 0.2, 0.3])
    
    if not recent_pattern.empty:
        if show_candlestick and all(col in recent_pattern.columns for col in ['open', 'high', 'low', 'close']):
            fig.add_trace(go.Candlestick(
                x=recent_pattern['date'],
                open=recent_pattern['open'],
                high=recent_pattern['high'],
                low=recent_pattern['low'],
                close=recent_pattern['close'],
                name='Current Pattern'
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(
                x=recent_pattern['date'], y=recent_pattern['close'],
                mode='lines', name='Current Pattern', line=dict(color='blue', width=2)
            ), row=1, col=1)
    
    colors = ['red', 'green', 'purple', 'orange', 'pink', 'cyan']
    for i, match in enumerate(matches[:6]):
        hist = match['historical_pattern']
        if not hist.empty:
            if show_candlestick and all(col in hist.columns for col in ['open', 'high', 'low', 'close']):
                fig.add_trace(go.Candlestick(
                    x=hist['date'],
                    open=hist['open'],
                    high=hist['high'],
                    low=hist['low'],
                    close=hist['close'],
                    name=f"Match {match['year']} (Sim: {match['similarity']:.2f})",
                    increasing_line_color=colors[i % len(colors)], 
                    decreasing_line_color=colors[i % len(colors)], 
                    opacity=0.5
                ), row=1, col=1)
            else:
                fig.add_trace(go.Scatter(
                    x=hist['date'], y=hist['close'],
                    mode='lines', name=f"Match {match['year']} (Sim: {match['similarity']:.2f})",
                    line=dict(color=colors[i % len(colors)], width=1, dash='dash')
                ), row=1, col=1)
            if 'volume' in hist.columns:
                fig.add_trace(go.Bar(
                    x=hist['date'], y=hist['volume'], name=f"Vol {match['year']}",
                    marker_color=colors[i % len(colors)], opacity=0.3, showlegend=False
                ), row=2, col=1)
    
    if 'volume' in recent_pattern.columns and not recent_pattern.empty:
        fig.add_trace(go.Bar(
            x=recent_pattern['date'], y=recent_pattern['volume'],
            name='Current Volume', marker_color='blue', opacity=0.5
        ), row=2, col=1)
    
    if show_indicators:
        indicators = ['ma20', 'ma50', 'macd', 'rsi']
        ind_colors = ['orange', 'purple', 'pink', 'cyan']
        for i, col in enumerate(indicators):
            if col in recent_pattern.columns and not recent_pattern.empty:
                fig.add_trace(go.Scatter(
                    x=recent_pattern['date'], y=recent_pattern['close'],
                    mode='lines', name=col.upper(),
                    line=dict(width=1, dash='dot', color=ind_colors[i]),
                    yaxis='y3'
                ), row=3, col=1)
    
    fig.update_layout(
        title="Stock Pattern Comparison",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2_title="Volume",
        yaxis3_title="Indicators",
        hovermode="x unified",
        showlegend=True,
        height=900,
        template="plotly_white"
    )
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="Indicators", row=3, col=1, secondary_y=True)
    
    return fig

# Function to calculate profit/loss and train ML model
def calculate_profit_loss(df, initial_investment, days_predict):
    profit_loss_data = []
    years = range(2010, 2026)  # 2010 to 2025
    for year in years:
        start_date = datetime(year, 6, 21)
        end_date = start_date + timedelta(days=days_predict)
        start_price = df[df['date'] == start_date]['close'].iloc[0] if start_date in df['date'].values else np.nan
        end_price = df[df['date'] == end_date]['close'].iloc[0] if end_date in df['date'].values else np.nan
        if not np.isnan(start_price) and not np.isnan(end_price):
            profit_loss = (end_price - start_price) / start_price * 100
            dollar_change = (end_price - start_price) * (initial_investment / start_price)
            profit_loss_data.append({
                'Year': year,
                'Start Price': start_price,
                'End Price': end_price,
                'Profit/Loss (%)': profit_loss,
                'Profit/Loss ($)': dollar_change
            })
    
    # ML Prediction for 2025
    historical_data = pd.DataFrame(profit_loss_data[:-1])  # Exclude 2025 for training
    if len(historical_data) > 1 and 'Start Price' in historical_data.columns and 'End Price' in historical_data.columns:
        X = historical_data[['Start Price']].values
        y = historical_data['End Price'].values
        model = LinearRegression()
        model.fit(X, y)
        predicted_end_price_2025 = model.predict([[profit_loss_data[-1]['Start Price']]])
        profit_loss_data[-1]['End Price'] = predicted_end_price_2025[0]
        profit_loss_data[-1]['Profit/Loss (%)'] = ((predicted_end_price_2025[0] - profit_loss_data[-1]['Start Price']) / profit_loss_data[-1]['Start Price']) * 100
        profit_loss_data[-1]['Profit/Loss ($)'] = (predicted_end_price_2025[0] - profit_loss_data[-1]['Start Price']) * (initial_investment / profit_loss_data[-1]['Start Price'])
    
    return profit_loss_data

# Main app logic
if uploaded_file and run_analysis:
    st.header("Stock Pattern Analysis Results")
    
    # Load data
    df = load_data(uploaded_file)
    if df is None:
        st.stop()
    
    # Run pattern analysis
    matches, recent_pattern = find_similar_patterns(df, days_to_analyze, days_predict, analysis_mode)
    
    # Create and display chart
    fig = create_chart(df, recent_pattern, matches, show_indicators, show_candlestick)
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction summary in expandable section
    with st.expander("Prediction and Profit/Loss Summary"):
        if matches:
            st.subheader("Pattern Matches")
            summary_data = [{
                'Year': m['year'],
                'Target Date': m['target_date'].strftime('%Y-%m-%d'),
                'Correlation Score': f"{m['similarity']:.2f}",
                'Forward Movement (%)': f"{m['forward_return']:.2f}"
            } for m in matches]
            st.table(summary_data)
        
        # Calculate and display profit/loss
        st.subheader("Profit/Loss Comparison (June 21 to June 26)")
        profit_loss_data = calculate_profit_loss(df, initial_investment, days_predict)
        if profit_loss_data:
            profit_loss_df = pd.DataFrame(profit_loss_data)
            st.table(profit_loss_df)
            predicted_return = profit_loss_data[-1]['Profit/Loss (%)']
            st.write(f"Projected {days_predict}-day movement for 2025: {predicted_return:.2f}%")
        
        # Download predicted data
        pred_df = pd.DataFrame({
            'Date': [datetime(2025, 6, 21) + timedelta(days=i) for i in range(days_predict + 1)],
            'Predicted_Close': [profit_loss_data[-1]['Start Price']] + [profit_loss_data[-1]['End Price']] * days_predict
        })
        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="Download Predicted Data",
            data=csv,
            file_name="predicted_stock_data.csv",
            mime="text/csv"
        )
    
    if not matches:
        st.error("No matching patterns found. Suggestions:\n"
                 "- Ensure your dataset spans multiple years (at least 2010–2025).\n"
                 "- Adjust 'Days to Analyze' to a smaller or larger window.\n"
                 "- Verify the date column is in a valid format (e.g., YYYY-MM-DD).\n"
                 "- Check for sufficient data points.")
elif uploaded_file:
    st.info("Please click 'Run Analysis' to process the uploaded data.")
else:
    st.info("Please upload a CSV or Excel file to begin analysis.")
