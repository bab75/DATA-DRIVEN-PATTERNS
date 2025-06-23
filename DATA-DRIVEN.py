import streamlit as st
import pandas as pd
import numpy as np
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
compare_days = st.sidebar.number_input("Compare Days (1-30)", min_value=1, max_value=30, value=6)
analysis_mode = st.sidebar.radio("Analysis Mode", ["Raw Data (Open vs. Close)", "Open/Close/High/Low", "Technical Indicators"])
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
        required_columns = ['date', 'open', 'close']
        if analysis_mode == "Open/Close/High/Low":
            required_columns.extend(['high', 'low'])
        elif analysis_mode == "Technical Indicators":
            required_columns.extend(['high', 'low', 'ma20', 'ma50', 'macd', 'rsi', 'atr', 'vwap'])
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            st.error(f"Missing required columns for {analysis_mode}: {', '.join(missing)}")
            return None
        if len(df) < compare_days:
            st.error(f"Dataset has {len(df)} rows, but at least {compare_days} are required.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Function to calculate rolling profit/loss with ML prediction
def calculate_rolling_profit_loss(df, compare_days, mode):
    profit_loss_data = []
    current_year = datetime.now().year  # 2025
    years = range(df['date'].dt.year.min(), current_year + 1)
    
    for year in years:
        year_df = df[df['date'].dt.year == year].copy()
        if len(year_df) < compare_days:
            continue
        for i in range(len(year_df) - compare_days + 1):
            start_date = year_df['date'].iloc[i]
            end_date = year_df['date'].iloc[i + compare_days - 1]
            start_price = year_df['open'].iloc[i]
            end_price = year_df['close'].iloc[i + compare_days - 1]
            profit_loss_percent = ((end_price - start_price) / start_price) * 100
            features = {'Start Open': start_price, 'End Close': end_price}
            if mode == "Open/Close/High/Low":
                features.update({
                    'High': year_df['high'].iloc[i + compare_days - 1],
                    'Low': year_df['low'].iloc[i + compare_days - 1]
                })
            elif mode == "Technical Indicators":
                features.update({
                    'High': year_df['high'].iloc[i + compare_days - 1],
                    'Low': year_df['low'].iloc[i + compare_days - 1],
                    'MA20': year_df['ma20'].iloc[i + compare_days - 1],
                    'MA50': year_df['ma50'].iloc[i + compare_days - 1],
                    'MACD': year_df['macd'].iloc[i + compare_days - 1],
                    'RSI': year_df['rsi'].iloc[i + compare_days - 1],
                    'ATR': year_df['atr'].iloc[i + compare_days - 1],
                    'VWAP': year_df['vwap'].iloc[i + compare_days - 1]
                })
            profit_loss_data.append({
                'Year': year,
                'Start Date': start_date,
                'End Date': end_date,
                'Start Open Price': start_price,
                'End Close Price': end_price,
                'Profit/Loss (%)': profit_loss_percent,
                **features
            })
    
    # ML Prediction for 2025 future dates
    historical_data = pd.DataFrame(profit_loss_data[:-len([d for d in profit_loss_data if d['Year'] == current_year])])
    future_data = []
    if len(historical_data) > compare_days:
        X = historical_data[['Start Open Price'] + [k for k in features.keys() if k not in ['Start Open', 'End Close']]].values
        y = historical_data['End Close Price'].values
        model = LinearRegression()
        try:
            model.fit(X, y)
            last_date = df['date'].iloc[-1]
            if last_date.year == current_year:
                start_idx = df[df['date'] == last_date].index[0]
                if start_idx + compare_days - 1 < len(df):
                    start_price = df['open'].iloc[start_idx]
                    features_dict = {'Start Open': start_price}
                    if mode == "Open/Close/High/Low":
                        features_dict.update({
                            'High': df['high'].iloc[start_idx + compare_days - 1] if start_idx + compare_days - 1 < len(df) else np.nan,
                            'Low': df['low'].iloc[start_idx + compare_days - 1] if start_idx + compare_days - 1 < len(df) else np.nan
                        })
                    elif mode == "Technical Indicators":
                        features_dict.update({
                            'High': df['high'].iloc[start_idx + compare_days - 1] if start_idx + compare_days - 1 < len(df) else np.nan,
                            'Low': df['low'].iloc[start_idx + compare_days - 1] if start_idx + compare_days - 1 < len(df) else np.nan,
                            'MA20': df['ma20'].iloc[start_idx + compare_days - 1] if start_idx + compare_days - 1 < len(df) else np.nan,
                            'MA50': df['ma50'].iloc[start_idx + compare_days - 1] if start_idx + compare_days - 1 < len(df) else np.nan,
                            'MACD': df['macd'].iloc[start_idx + compare_days - 1] if start_idx + compare_days - 1 < len(df) else np.nan,
                            'RSI': df['rsi'].iloc[start_idx + compare_days - 1] if start_idx + compare_days - 1 < len(df) else np.nan,
                            'ATR': df['atr'].iloc[start_idx + compare_days - 1] if start_idx + compare_days - 1 < len(df) else np.nan,
                            'VWAP': df['vwap'].iloc[start_idx + compare_days - 1] if start_idx + compare_days - 1 < len(df) else np.nan
                        })
                    X_predict = [v for k, v in features_dict.items() if k != 'Start Open']
                    predicted_end_price = model.predict([X_predict])[0]
                    end_date = last_date + timedelta(days=compare_days)
                    profit_loss_percent = ((predicted_end_price - start_price) / start_price) * 100
                    future_data.append({
                        'Year': current_year,
                        'Start Date': last_date,
                        'End Date': end_date,
                        'Start Open Price': start_price,
                        'End Close Price': predicted_end_price,
                        'Profit/Loss (%)': profit_loss_percent,
                        **features_dict
                    })
        except Exception as e:
            st.warning(f"ML prediction failed: {str(e)}. Using last known value.")
            predicted_end_price = start_price  # Fallback
        profit_loss_data.extend(future_data)
    
    return profit_loss_data

# Function to create interactive Plotly chart
def create_chart(df, profit_loss_data, mode):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        subplot_titles=['Price Patterns', 'Profit/Loss'],
                        row_heights=[0.7, 0.3])
    
    for year in set(d['Year'] for d in profit_loss_data):
        year_data = [d for d in profit_loss_data if d['Year'] == year]
        if year_data:
            dates = [d['Start Date'] for d in year_data] + [year_data[-1]['End Date']]
            prices = [d['Start Open Price'] for d in year_data] + [year_data[-1]['End Close Price']]
            fig.add_trace(go.Scatter(
                x=dates, y=prices,
                mode='lines+markers', name=f"Year {year}",
                line=dict(width=1, dash='dash' if year == 2025 else 'solid')
            ), row=1, col=1)
    
    profits = [d['Profit/Loss (%)'] for d in profit_loss_data]
    dates = [d['End Date'] for d in profit_loss_data]
    fig.add_trace(go.Bar(
        x=dates, y=profits,
        name='Profit/Loss (%)',
        marker_color=['green' if p >= 0 else 'red' for p in profits],
        opacity=0.7
    ), row=2, col=1)
    
    fig.update_layout(
        title=f"Stock Price and Profit/Loss Analysis ({mode})",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2_title="Profit/Loss (%)",
        hovermode="x unified",
        showlegend=True,
        height=800,
        template="plotly_white"
    )
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Profit/Loss (%)", row=2, col=1)
    
    return fig

# Main app logic
if uploaded_file and run_analysis:
    st.header("Stock Pattern Analysis Results")
    
    # Load data
    df = load_data(uploaded_file)
    if df is None:
        st.stop()
    
    # Calculate rolling profit/loss
    profit_loss_data = calculate_rolling_profit_loss(df, compare_days, analysis_mode)
    
    # Create and display chart
    fig = create_chart(df, profit_loss_data, analysis_mode)
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction summary in expandable section
    with st.expander("Profit/Loss Summary"):
        if profit_loss_data:
            profit_loss_df = pd.DataFrame(profit_loss_data)
            styled_df = profit_loss_df.style.apply(
                lambda x: ['background-color: green' if v >= 0 else 'background-color: red' for v in x['Profit/Loss (%)']],
                subset=['Profit/Loss (%)']
            )
            st.table(styled_df)
            if any(d['Year'] == 2025 for d in profit_loss_data):
                predicted_pl = next(d for d in profit_loss_data if d['Year'] == 2025)['Profit/Loss (%)']
                st.write(f"Predicted Profit/Loss for 2025 ({compare_days}-day window): {predicted_pl:.2f}%")
        
        # Download predicted data
        pred_df = pd.DataFrame(profit_loss_data)
        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="Download Predicted Data",
            data=csv,
            file_name="profit_loss_data.csv",
            mime="text/csv"
        )
    
    if not profit_loss_data:
        st.error("No profit/loss data calculated. Suggestions:\n"
                 "- Ensure your dataset spans multiple years (at least 2010–2025).\n"
                 "- Verify the date column is in a valid format (e.g., YYYY-MM-DD).\n"
                 "- Check for sufficient data points (at least {compare_days} rows per year).")
elif uploaded_file:
    st.info("Please click 'Run Analysis' to process the uploaded data.")
else:
    st.info("Please upload a CSV or Excel file to begin analysis.")
