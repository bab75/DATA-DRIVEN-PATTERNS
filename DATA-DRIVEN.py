import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
from calendar import month_name

# Set Streamlit page config
st.set_page_config(page_title="Stock Pattern Analyzer", layout="wide")

# Sidebar: Control Panel
st.sidebar.header("Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
compare_days = st.sidebar.number_input("Compare Days (1-30)", min_value=1, max_value=30, value=2)
analysis_mode = st.sidebar.radio("Analysis Mode", ["Raw Data (Open vs. Close)", "Open/Close/High/Low", "Technical Indicators"])
run_analysis = st.sidebar.button("Run Analysis")

# Function to load and preprocess data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        df['date'] = pd.to_datetime(df['date']).dt.date  # Remove time component
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

# Function to format date as "Jan 2, 2020"
def format_date(date):
    if pd.isna(date):
        return ""
    day = date.day
    if 4 <= day <= 20 or 24 <= day <= 30:
        suffix = "th"
    else:
        suffix = ["st", "nd", "rd"][day % 10 - 1]
    return f"{month_name[date.month][:3]} {day}{suffix}, {date.year}"

# Function to get next available date
def get_next_available_date(df, current_date):
    next_date = current_date + timedelta(days=1)
    while next_date not in df['date'].values and (next_date - df['date'].iloc[-1]).days < 30:  # Limit to 30 days
        next_date += timedelta(days=1)
    return next_date if next_date in df['date'].values else None

# Function to calculate rolling profit/loss with ML prediction
def calculate_rolling_profit_loss(df, compare_days, mode):
    profit_loss_data = []
    current_year = datetime.now().year  # 2025
    years = sorted(set(df['date'].apply(lambda x: x.year)))

    for year in years:
        year_df = df[df['date'].apply(lambda x: x.year) == year].copy()
        if len(year_df) < compare_days:
            continue
        current_date = min(year_df['date'])
        while current_date and (current_date + timedelta(days=compare_days-1) <= max(year_df['date'])):
            start_date = current_date
            end_date = start_date + timedelta(days=compare_days-1)
            if end_date not in year_df['date'].values:
                end_date = get_next_available_date(year_df, end_date)
                if not end_date:
                    break
            start_idx = year_df.index[year_df['date'] == start_date][0]
            end_idx = year_df.index[year_df['date'] == end_date][0]
            start_price = year_df['open'].iloc[start_idx]
            end_price = year_df['close'].iloc[end_idx]
            profit_loss_percent = ((end_price - start_price) / start_price) * 100 if not np.isnan(end_price - start_price) else 0.0
            features = {'Start Open': start_price, 'End Close': end_price}
            if mode == "Open/Close/High/Low":
                features.update({
                    'High': year_df['high'].iloc[end_idx],
                    'Low': year_df['low'].iloc[end_idx]
                })
            elif mode == "Technical Indicators":
                features.update({
                    'High': year_df['high'].iloc[end_idx],
                    'Low': year_df['low'].iloc[end_idx],
                    'MA20': year_df['ma20'].iloc[end_idx],
                    'MA50': year_df['ma50'].iloc[end_idx],
                    'MACD': year_df['macd'].iloc[end_idx],
                    'RSI': year_df['rsi'].iloc[end_idx],
                    'ATR': year_df['atr'].iloc[end_idx],
                    'VWAP': year_df['vwap'].iloc[end_idx]
                })
            profit_loss_data.append({
                'Year': year,
                'Start Date': start_date,
                'End Date': end_date,
                'Start Open Price': start_price,
                'End Close Price': end_price,
                'Profit/Loss (%)': profit_loss_percent,
                **{k: v for k, v in features.items() if v is not np.nan}
            })
            current_date = get_next_available_date(year_df, end_date)

    # ML Prediction for 2025
    historical_data = pd.DataFrame([d for d in profit_loss_data if d['Year'] != current_year])
    future_data = []
    if len(historical_data) > compare_days and 'Profit/Loss (%)' in historical_data.columns:
        X = historical_data[['Start Open Price'] + [k for k in historical_data.columns if k not in ['Year', 'Start Date', 'End Date', 'Start Open Price', 'End Close Price', 'Profit/Loss (%)']]].values
        y = historical_data['End Close Price'].values
        model = LinearRegression()
        try:
            model.fit(X, y)
            last_date = df['date'].iloc[-1]
            if last_date.year == current_year:
                start_idx = df.index[df['date'] == last_date][0]
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
                X_predict = [v for k, v in features_dict.items() if k != 'Start Open' and k in historical_data.columns]
                if X_predict and len(X_predict) == X.shape[1]:
                    predicted_end_price = model.predict([X_predict])[0]
                else:
                    predicted_end_price = start_price
                end_date = last_date + timedelta(days=compare_days)
                profit_loss_percent = ((predicted_end_price - start_price) / start_price) * 100 if not np.isnan(predicted_end_price - start_price) else 0.0
                future_data.append({
                    'Year': current_year,
                    'Start Date': last_date,
                    'End Date': end_date,
                    'Start Open Price': start_price,
                    'End Close Price': predicted_end_price,
                    'Profit/Loss (%)': profit_loss_percent,
                    **{k: v for k, v in features_dict.items() if v is not np.nan and k in historical_data.columns}
                })
        except Exception as e:
            st.warning(f"ML prediction failed: {str(e)}. Using last known value.")
            predicted_end_price = start_price
            profit_loss_percent = 0.0
        profit_loss_data.extend(future_data)

    return profit_loss_data

# Function to create interactive Plotly chart
def create_chart(df, profit_loss_data, mode):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        subplot_titles=['Price Patterns', 'Profit/Loss'],
                        row_heights=[0.7, 0.3])
    
    unique_years = set(d['Year'] for d in profit_loss_data)
    for year in unique_years:
        year_data = [d for d in profit_loss_data if d['Year'] == year]
        if year_data:
            dates = [d['Start Date'] for d in year_data] + [year_data[-1]['End Date']]
            prices = [d['Start Open Price'] for d in year_data] + [year_data[-1]['End Close Price']]
            formatted_dates = [format_date(d) for d in dates]
            fig.add_trace(go.Scatter(
                x=formatted_dates, y=prices,
                mode='lines+markers', name=f"Year {year}",
                line=dict(width=1, dash='dash' if year == 2025 else 'solid')
            ), row=1, col=1)
    
    profits = [d['Profit/Loss (%)'] for d in profit_loss_data if 'Profit/Loss (%)' in d]
    dates = [format_date(d['End Date']) for d in profit_loss_data if 'Profit/Loss (%)' in d]
    fig.add_trace(go.Bar(
        x=dates, y=profits,
        name='Profit/Loss (%)',
        marker_color=['#90EE90' if p >= 0 else '#FFB6C1' for p in profits],
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

# Function to create styled table for a year
def create_year_table(year_data):
    if not year_data:
        return None
    year = year_data[0]['Year']
    columns = ['Year'] + [f"{format_date(d['Start Date'])} to {format_date(d['End Date'])}" for d in year_data]
    values = [year] + [d['Profit/Loss (%)'] for d in year_data]
    df = pd.DataFrame([values], columns=columns)
    styled_df = df.style.apply(
        lambda x: ['background-color: #90EE90' if v >= 0 else 'background-color: #FFB6C1' for v in x[1:]],
        subset=[col for col in df.columns if col != 'Year'],
        axis=1
    ).set_properties(**{'text-align': 'center'})
    return styled_df

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
    
    # Group data by year and display separate tables
    years = sorted(set(d['Year'] for d in profit_loss_data) - {2025})  # Exclude 2025 from historical tables
    for year in years:
        year_data = [d for d in profit_loss_data if d['Year'] == year]
        if year_data:
            st.subheader(f"Profit/Loss for Year {year}")
            styled_df = create_year_table(year_data)
            if styled_df is not None:
                st.dataframe(styled_df, use_container_width=True)
    
    # Display 2025 prediction
    current_year_data = [d for d in profit_loss_data if d['Year'] == 2025]
    if current_year_data:
        st.subheader("2025 Prediction")
        pred_df = pd.DataFrame(current_year_data).fillna(0)
        styled_pred_df = pred_df.style.apply(
            lambda x: ['background-color: #90EE90' if v >= 0 else 'background-color: #FFB6C1' for v in x['Profit/Loss (%)']],
            subset=['Profit/Loss (%)'],
            axis=1
        ).set_properties(**{'text-align': 'center'})
        st.dataframe(styled_pred_df, use_container_width=True)
        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="Download 2025 Prediction",
            data=csv,
            file_name="2025_prediction.csv",
            mime="text/csv"
        )
    
    # Download all data
    all_df = pd.DataFrame(profit_loss_data).fillna(0)
    csv_all = all_df.to_csv(index=False)
    st.download_button(
        label="Download All Data",
        data=csv_all,
        file_name="all_profit_loss_data.csv",
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
