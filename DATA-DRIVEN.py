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
        if 'close' not in df.columns:
            st.error("Missing required column: 'close'")
            return None
        if len(df) < compare_days:
            st.error(f"Dataset has {len(df)} rows, but at least {compare_days} are required.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Function to calculate rolling profit/loss
def calculate_rolling_profit_loss(df, compare_days, initial_investment):
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
            start_price = year_df['close'].iloc[i]
            end_price = year_df['close'].iloc[i + compare_days - 1]
            profit_loss_percent = ((end_price - start_price) / start_price) * 100
            profit_loss_dollar = (end_price - start_price) * (initial_investment / start_price)
            profit_loss_data.append({
                'Year': year,
                'Start Date': start_date,
                'End Date': end_date,
                'Start Price': start_price,
                'End Price': end_price,
                'Profit/Loss (%)': profit_loss_percent,
                'Profit/Loss ($)': profit_loss_dollar
            })
    
    # ML Prediction for 2025 future dates
    historical_data = pd.DataFrame(profit_loss_data)
    future_data = []
    if len(historical_data) > compare_days:
        X = historical_data[['Start Price']].values
        y = historical_data['End Price'].values
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict for 2025 from the last available date
        last_date = df['date'].iloc[-1]
        if last_date.year == current_year:
            start_idx = df[df['date'] == last_date].index[0]
            if start_idx + compare_days - 1 < len(df):
                start_price = df['close'].iloc[start_idx]
                predicted_end_price = model.predict([[start_price]])[0]
                end_date = last_date + timedelta(days=compare_days)
                profit_loss_percent = ((predicted_end_price - start_price) / start_price) * 100
                profit_loss_dollar = (predicted_end_price - start_price) * (initial_investment / start_price)
                future_data.append({
                    'Year': current_year,
                    'Start Date': last_date,
                    'End Date': end_date,
                    'Start Price': start_price,
                    'End Price': predicted_end_price,
                    'Profit/Loss (%)': profit_loss_percent,
                    'Profit/Loss ($)': profit_loss_dollar
                })
            profit_loss_data.extend(future_data)
    
    return profit_loss_data

# Function to create interactive Plotly chart
def create_chart(df, profit_loss_data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        subplot_titles=['Price Patterns', 'Profit/Loss'],
                        row_heights=[0.7, 0.3])
    
    # Plot price patterns
    for year in set(d['Year'] for d in profit_loss_data):
        year_data = [d for d in profit_loss_data if d['Year'] == year]
        if year_data:
            dates = [d['Start Date'] for d in year_data] + [year_data[-1]['End Date']]
            prices = [d['Start Price'] for d in year_data] + [year_data[-1]['End Price']]
            fig.add_trace(go.Scatter(
                x=dates, y=prices,
                mode='lines+markers', name=f"Year {year}",
                line=dict(width=1, dash='dash' if year == 2025 else 'solid')
            ), row=1, col=1)
    
    # Plot profit/loss
    profits = [d['Profit/Loss ($)'] for d in profit_loss_data]
    dates = [d['End Date'] for d in profit_loss_data]
    fig.add_trace(go.Bar(
        x=dates, y=profits,
        name='Profit/Loss ($)',
        marker_color=['green' if p >= 0 else 'red' for p in profits],
        opacity=0.7
    ), row=2, col=1)
    
    fig.update_layout(
        title="Stock Price and Profit/Loss Analysis",
        xaxis_title="Date",
        yaxis_title="Price",
        yaxis2_title="Profit/Loss ($)",
        hovermode="x unified",
        showlegend=True,
        height=800,
        template="plotly_white"
    )
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Profit/Loss ($)", row=2, col=1)
    
    return fig

# Main app logic
if uploaded_file and run_analysis:
    st.header("Stock Pattern Analysis Results")
    
    # Load data
    df = load_data(uploaded_file)
    if df is None:
        st.stop()
    
    # Calculate rolling profit/loss
    profit_loss_data = calculate_rolling_profit_loss(df, compare_days, initial_investment)
    
    # Create and display chart
    fig = create_chart(df, profit_loss_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction summary in expandable section
    with st.expander("Profit/Loss Summary"):
        if profit_loss_data:
            profit_loss_df = pd.DataFrame(profit_loss_data)
            styled_df = profit_loss_df.style.apply(
                lambda x: ['background-color: green' if v >= 0 else 'background-color: red' for v in x['Profit/Loss ($)']],
                subset=['Profit/Loss (%)', 'Profit/Loss ($)']
            )
            st.table(styled_df)
            if any(d['Year'] == 2025 for d in profit_loss_data):
                predicted_pl = next(d for d in profit_loss_data if d['Year'] == 2025)['Profit/Loss ($)']
                st.write(f"Predicted Profit/Loss for 2025 ({compare_days}-day window): ${predicted_pl:.2f}")
        
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
