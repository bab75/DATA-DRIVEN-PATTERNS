import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
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
days_to_predict = st.sidebar.number_input("Days to Predict (1-30)", min_value=1, max_value=30, value=10)
analysis_mode = st.sidebar.radio("Analysis Mode", ["Raw Data Only", "Use Technical Indicators"])
show_indicators = st.sidebar.checkbox("Show Technical Indicators on Chart", value=False)
run_analysis = st.sidebar.button("Run Analysis")

# Function to load and preprocess data
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

# Function to compute similarity between two series
def compute_similarity(series1, series2, method='pearson'):
    if method == 'pearson':
        return pearsonr(series1, series2)[0]
    else:  # cosine
        return cosine_similarity([series1], [series2])[0][0]

# Function to extract pattern and compare
def find_similar_patterns(df, days_analyze, days_predict, mode):
    recent_pattern = df.tail(days_analyze).copy()
    recent_date = recent_pattern['date'].iloc[-1]
    recent_close = recent_pattern['close'].pct_change().fillna(0)
    recent_volatility = recent_pattern['close'].std()
    
    matches = []
    
    # Define features based on mode
    if mode == "Raw Data Only":
        features = ['close']
    else:
        features = ['close', 'macd', 'rsi', 'atr', 'vwap']
    
    # Slide through historical data
    for year in range(df['date'].dt.year.min(), df['date'].dt.year.max()):
        target_date = recent_date.replace(year=year)
        if target_date in df['date'].values:
            idx = df[df['date'] == target_date].index[0]
            if idx >= days_analyze:
                historical = df.iloc[idx-days_analyze:idx+1].copy()
                if len(historical) == len(recent_pattern):
                    # Compute similarity
                    similarity_scores = []
                    for feature in features:
                        hist_series = historical[feature].pct_change().fillna(0)
                        recent_series = recent_pattern[feature].pct_change().fillna(0)
                        score = compute_similarity(hist_series, recent_series)
                        similarity_scores.append(score)
                    avg_similarity = np.nanmean(similarity_scores)
                    
                    # Get forward movement
                    if idx + days_predict < len(df):
                        forward_data = df.iloc[idx:idx+days_predict+1]
                        forward_return = (forward_data['close'].iloc[-1] / forward_data['close'].iloc[0] - 1) * 100
                        matches.append({
                            'year': year,
                            'similarity': avg_similarity,
                            'forward_return': forward_return,
                            'historical_pattern': historical,
                            'forward_data': forward_data
                        })
    
    # Sort and select top matches
    matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)[:3]
    return matches, recent_pattern

# Function to create interactive Plotly chart
def create_chart(df, recent_pattern, matches, show_indicators):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, subplot_titles=['Price Patterns', 'Volume'],
                        row_heights=[0.7, 0.3])
    
    # Plot recent pattern
    fig.add_trace(go.Scatter(x=recent_pattern['date'], y=recent_pattern['close'],
                             mode='lines', name='Current Pattern', line=dict(color='blue', width=2)),
                  row=1, col=1)
    
    # Plot historical matches
    colors = ['red', 'green', 'purple']
    for i, match in enumerate(matches):
        hist = match['historical_pattern']
        fig.add_trace(go.Scatter(x=hist['date'], y=hist['close'],
                                 mode='lines', name=f"Match {match['year']} (Sim: {match['similarity']:.2f})",
                                 line=dict(color=colors[i], width=1, dash='dash')),
                      row=1, col=1)
        fig.add_trace(go.Bar(x=hist['date'], y=hist['volume'], name=f"Vol {match['year']}",
                             marker_color=colors[i], opacity=0.3, showlegend=False),
                      row=2, col=1)
    
    # Add technical indicators if selected
    if show_indicators:
        for col in ['ma20', 'ma50', 'macd', 'rsi']:
            if col in recent_pattern.columns:
                fig.add_trace(go.Scatter(x=recent_pattern['date'], y=recent_pattern[col],
                                         mode='lines', name=col.upper(),
                                         line=dict(width=1, dash='dot')),
                              row=1, col=1)
    
    # Add volume for recent pattern
    fig.add_trace(go.Bar(x=recent_pattern['date'], y=recent_pattern['volume'],
                         name='Current Volume', marker_color='blue', opacity=0.5),
                  row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title="Stock Pattern Comparison",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        showlegend=True,
        height=800,
        template="plotly_white"
    )
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

# Main app logic
if uploaded_file and run_analysis:
    st.header("Stock Pattern Analysis Results")
    
    # Load data
    df = load_data(uploaded_file)
    
    # Run analysis
    matches, recent_pattern = find_similar_patterns(df, days_to_analyze, days_to_predict, analysis_mode)
    
    if matches:
        # Create and display chart
        fig = create_chart(df, recent_pattern, matches, show_indicators)
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction summary
        st.subheader("Prediction Summary")
        summary_data = [{
            'Year': m['year'],
            'Correlation Score': f"{m['similarity']:.2f}",
            'Forward Movement (%)': f"{m['forward_return']:.2f}"
        } for m in matches]
        st.table(summary_data)
        
        # Generate predicted path
        predicted_returns = np.mean([m['forward_return'] for m in matches])
        st.write(f"Projected {days_to_predict}-day movement: {predicted_returns:.2f}%")
        
        # Download predicted data
        pred_df = pd.DataFrame({
            'Date': [recent_pattern['date'].iloc[-1] + timedelta(days=i) for i in range(1, days_to_predict+1)],
            'Predicted_Close': [recent_pattern['close'].iloc[-1] * (1 + predicted_returns/100) for _ in range(days_to_predict)]
        })
        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="Download Predicted Data",
            data=csv,
            file_name="predicted_stock_data.csv",
            mime="text/csv"
        )
        
        # Download chart
        img_bytes = fig.to_image(format="png")
        st.download_button(
            label="Download Chart",
            data=img_bytes,
            file_name="stock_pattern_chart.png",
            mime="image/png"
        )
    else:
        st.error("No matching patterns found. Try adjusting the analysis window or uploading a different dataset.")
elif uploaded_file:
    st.info("Please click 'Run Analysis' to process the uploaded data.")
else:
    st.info("Please upload a CSV or Excel file to begin analysis.")
