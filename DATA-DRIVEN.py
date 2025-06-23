import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

st.set_page_config(page_title="Pattern Predictor App", layout="wide")

# --- Sidebar Inputs ---
st.sidebar.title("📊 Pattern Prediction Engine")

uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])
load_file = st.sidebar.button("📂 Load File")

if load_file and uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        st.session_state['df'] = df
        st.success(f"✅ File loaded: {uploaded_file.name}")
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")

# Proceed only if file is loaded
if 'df' in st.session_state:
    df = st.session_state['df']

    st.sidebar.markdown("---")
    pattern_mode = st.sidebar.radio("Data Mode:", ["Raw Data", "With Technical Indicators"])
    pattern_days = st.sidebar.number_input("Days to Analyze", min_value=1, max_value=365, value=30)
    predict_days = st.sidebar.number_input("Days to Predict", min_value=1, max_value=60, value=5)
    show_indicators = st.sidebar.checkbox("Overlay Technical Indicators")
    analyze_button = st.sidebar.button("▶️ Run Analysis")

    # --- Feature Columns ---
    raw_features = ['close']
    technical_features = [
        'close', 'ma20', 'ma50', 'ma200', 'ema12', 'ema26', 'macd', 'signal', 'high_diff', 'low_diff',
        'plus_dm', 'minus_dm', 'tr', 'atr', 'plus_di', 'minus_di', 'adx', 'std_dev', 'upper_band',
        'lower_band', 'rsi', 'stochastic_k', 'stochastic_d', 'vwap', 'tenkan_sen', 'kijun_sen',
        'senkou_span_a', 'senkou_span_b', 'chikou_span', 'price_change'
    ]

    # --- Helper Functions ---
    def extract_vector(df, start, window, cols):
        return df.iloc[start:start+window][cols].values.flatten()

    def match_patterns(df, window, use_tech):
        cols = technical_features if use_tech else raw_features
        if df[cols].isnull().any().any():
            df = df.dropna(subset=cols)

        base_vector = extract_vector(df, len(df)-window-predict_days, window, cols)
        matches, scores = [], []
        for i in range(len(df) - window - predict_days):
            hist_vector = extract_vector(df, i, window, cols)
            if hist_vector.shape == base_vector.shape:
                score = cosine_similarity([hist_vector], [base_vector])[0][0]
                matches.append(i)
                scores.append(score)
        return matches, scores

    def predict_outcome(df, match_indices, scores):
        result = []
        for idx, score in sorted(zip(match_indices, scores), key=lambda x: x[1], reverse=True)[:3]:
            future = df.iloc[idx + pattern_days : idx + pattern_days + predict_days]
            pct_change = (future['close'].iloc[-1] - future['close'].iloc[0]) / future['close'].iloc[0] * 100
            result.append({
                "Match Date": df.iloc[idx]['date'].date(),
                "Similarity Score": round(score, 4),
                f"Next {predict_days} Days % Change": round(pct_change, 2)
            })
        return pd.DataFrame(result)

    def plot_patterns(df, recent, matched_idx):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent['date'], y=recent['close'],
            mode='lines+markers', name='Current Pattern',
            line=dict(color='blue'), hovertemplate="Date: %{x}<br>Close: %{y}"
        ))
        for i in matched_idx[:3]:
            match_data = df.iloc[i:i+pattern_days]
            fig.add_trace(go.Scatter(
                x=match_data['date'], y=match_data['close'],
                mode='lines', name=f"Match {match_data['date'].iloc[0].date()}",
                hovertemplate="Date: %{x}<br>Close: %{y}"
            ))
        fig.update_layout(title="📊 Pattern Comparison", hovermode="x unified")
        return fig

    # --- Analysis Execution ---
    if analyze_button:
        if len(df) < pattern_days + predict_days:
            st.error("Not enough data for selected analysis window.")
        else:
            st.subheader("🔍 Pattern Matching and Prediction")
            use_indicators = pattern_mode == "With Technical Indicators"
            match_idx, similarity = match_patterns(df, pattern_days, use_indicators)
            pred_df = predict_outcome(df, match_idx, similarity)

            st.plotly_chart(
                plot_patterns(df, df.iloc[-(pattern_days + predict_days):-predict_days], match_idx),
                use_container_width=True
            )
            st.dataframe(pred_df)

            csv = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions CSV", csv, "predictions.csv", "text/csv")

else:
    st.info("📁 Please upload a file and click 'Load File' to begin.")
