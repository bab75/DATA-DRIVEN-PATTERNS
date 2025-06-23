import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

st.set_page_config(page_title="Stock Pattern Predictor", layout="wide")

# --- Sidebar Inputs ---
st.sidebar.title("📊 Stock Pattern Analysis")

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])

pattern_mode = st.sidebar.radio("Select Data Mode:", ["Raw Data", "With Technical Indicators"])

pattern_days = st.sidebar.number_input("Days to Analyze", min_value=1, max_value=365, value=30)
predict_days = st.sidebar.number_input("Days to Predict", min_value=1, max_value=30, value=5)

show_indicators = st.sidebar.checkbox("Show Technical Indicator Overlays")

submit_button = st.sidebar.button("Run Pattern Analysis")

# --- Helper Functions ---
def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def calculate_similarity_matrix(data, base_vector):
    return cosine_similarity(data, base_vector.reshape(1, -1)).flatten()

def extract_feature_vector(df, start_idx, window, cols):
    return df.iloc[start_idx:start_idx + window][cols].values.flatten()

def pattern_match(df, window, use_indicators):
    feature_cols = ['close'] if not use_indicators else [
        'close', 'ma20', 'ma50', 'ema12', 'ema26', 'macd', 'rsi', 'vwap', 'atr'
    ]

    latest_vector = extract_feature_vector(df, len(df) - window - predict_days, window, feature_cols)
    similarities = []
    match_indices = []

    for i in range(len(df) - window - predict_days):
        historical_vector = extract_feature_vector(df, i, window, feature_cols)
        if historical_vector.shape == latest_vector.shape:
            score = cosine_similarity([historical_vector], [latest_vector])[0][0]
            similarities.append(score)
            match_indices.append(i)

    return match_indices, similarities

def get_prediction_result(df, match_indices, similarities, window, predict_days):
    result = []
    for idx, score in sorted(zip(match_indices, similarities), key=lambda x: x[1], reverse=True)[:3]:
        future_data = df.iloc[idx + window:idx + window + predict_days]
        actual_data = df.iloc[idx:idx + window]
        pct_change = (future_data['close'].iloc[-1] - future_data['close'].iloc[0]) / future_data['close'].iloc[0] * 100
        result.append({
            "Match Start": df.iloc[idx]['date'].date(),
            "Score": round(score, 4),
            "Next {} Days % Change".format(predict_days): round(pct_change, 2)
        })
    return pd.DataFrame(result)

def plot_comparison(df, recent_window, match_idx, window, title):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recent_window['date'], y=recent_window['close'],
        mode='lines+markers', name='Current Pattern', line=dict(color='blue')
    ))

    for idx in match_idx:
        past = df.iloc[idx:idx + window]
        fig.add_trace(go.Scatter(
            x=past['date'], y=past['close'],
            mode='lines', name=f"Match {past['date'].iloc[0].date()}"
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date", yaxis_title="Close Price",
        hovermode="x unified"
    )
    return fig

# --- Main App Execution ---
if submit_button and uploaded_file:
    df = load_data(uploaded_file)

    if len(df) < pattern_days + predict_days:
        st.error("Not enough data for the selected window.")
    else:
        st.success("Data loaded successfully.")
        use_indicators = pattern_mode == "With Technical Indicators"

        match_indices, similarities = pattern_match(df, pattern_days, use_indicators)
        result_df = get_prediction_result(df, match_indices, similarities, pattern_days, predict_days)

        # Show Comparison Chart
        st.subheader("📈 Pattern Match Comparison")
        recent_data = df.iloc[-(pattern_days + predict_days):-predict_days]
        fig = plot_comparison(df, recent_data, match_indices[:3], pattern_days, "Pattern Match vs Current")
        st.plotly_chart(fig, use_container_width=True)

        # Show Prediction Table
        st.subheader("🔮 Prediction Summary")
        st.dataframe(result_df)

        # Download Prediction
        csv_download = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Prediction CSV", csv_download, "predicted_results.csv", "text/csv")

else:
    st.info("Please upload a file and configure inputs to start the analysis.")
