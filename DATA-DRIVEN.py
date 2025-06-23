import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
from calendar import month_name
import time

# Custom CSS for beautiful design
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lora&display=swap');
    .stApp {
        background: linear-gradient(to right, #FFFFFF, #F0F8FF);
        font-family: 'Lora', serif;
    }
    .sidebar .sidebar-content {
        background-color: #F5F5F5;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .custom-table td {
        text-align: center;
        border: 1px solid #ddd;
        padding: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Set Streamlit page config
st.set_page_config(page_title="Stock Pattern Analyzer", layout="wide")

# Sidebar: Control Panel
with st.sidebar:
    st.header("Control Panel")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'], help="Upload a file with 'date', 'open', 'close' columns.")
    compare_days = st.number_input("Compare Days (1-30)", min_value=1, max_value=30, value=2, help="Number of days to compare for profit/loss.")
    analysis_mode = st.radio("Analysis Mode", ["Raw Data (Open vs. Close)", "Open/Close/High/Low", "Technical Indicators"],
                             help="Choose the data features for analysis.")
    run_analysis = st.button("Run Analysis")
    if st.button("Reset", key="reset"):
        st.session_state.clear()
    if st.button("Mode Description"):
        st.write("""
        - **Raw Data (Open vs. Close)**: Uses only 'open' and 'close' prices.
        - **Open/Close/High/Low**: Includes 'high' and 'low' prices.
        - **Technical Indicators**: Uses 'high', 'low', 'ma20', 'ma50', 'macd', 'rsi', 'atr', 'vwap'.
        """)

# Function to load and preprocess data
def load_data(file):
    with st.spinner("Loading data..."):
        time.sleep(1)  # Simulate loading
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        df['date'] = pd.to_datetime(df['date']).dt.date
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

# Function to format date range
def format_date_range(start_date, end_date):
    start_day = start_date.day
    end_day = end_date.day
    start_suffix = "th" if 4 <= start_day <= 20 or 24 <= start_day <= 30 else ["st", "nd", "rd"][start_day % 10 - 1]
    end_suffix = "th" if 4 <= end_day <= 20 or 24 <= end_day <= 30 else ["st", "nd", "rd"][end_day % 10 - 1]
    start_month = month_name[start_date.month][:3]
    end_month = month_name[end_date.month][:3]
    if start_date.month == end_date.month:
        return f"{start_month} {start_day}{start_suffix} to {end_day}{end_suffix}"
    return f"{start_month} {start_day}{start_suffix} to {end_month} {end_day}{end_suffix}"

# Function to get next available date
def get_next_available_date(df, current_date):
    next_date = current_date + timedelta(days=1)
    while next_date not in df['date'].values and (next_date - df['date'].iloc[-1]).days < 30:
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
        while current_date:
            start_date = current_date
            end_date = start_date + timedelta(days=compare_days-1)
            if start_date in year_df['date'].values:
                end_date_idx = year_df.index[year_df['date'] >= end_date].min()
                if pd.notna(end_date_idx) and end_date_idx < len(year_df):
                    end_date = year_df['date'].iloc[end_date_idx]
                    break
            current_date = get_next_available_date(year_df, current_date)
            if not current_date or current_date > max(year_df['date']):
                break
        if not current_date:
            continue

        while current_date and current_date <= max(year_df['date']):
            start_date = current_date
            if start_date not in year_df['date'].values:
                current_date = get_next_available_date(year_df, current_date)
                continue
            end_date = start_date + timedelta(days=compare_days-1)
            if end_date not in year_df['date'].values:
                end_date_idx = year_df.index[year_df['date'] >= end_date].min()
                if pd.isna(end_date_idx) or end_date_idx >= len(year_df):
                    current_date = get_next_available_date(year_df, start_date)
                    continue
                end_date = year_df['date'].iloc[end_date_idx]
            start_idx = year_df.index[year_df['date'] == start_date][0]
            end_idx = year_df.index[year_df['date'] == end_date][0]
            if start_idx >= len(year_df) or end_idx >= len(year_df):
                st.warning(f"Index out of bounds for year {year} at {format_date_range(start_date, end_date)}. Data length: {len(year_df)}, Last date: {format_date_range(min(year_df['date']), max(year_df['date']))}. Skipping this iteration.")
                current_date = get_next_available_date(year_df, end_date)
                continue
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
        X = historical_data[['Start Open Price']].values
        y = historical_data['End Close Price'].values
        model = LinearRegression()
        try:
            model.fit(X, y)
            last_date = df['date'].iloc[-1]
            if last_date.year == current_year:
                start_idx = df.index[df['date'] == last_date][0]
                start_price = df['open'].iloc[start_idx]
                X_predict = [[start_price]]
                predicted_end_price = model.predict(X_predict)[0]
                end_date = last_date + timedelta(days=compare_days)
                profit_loss_percent = ((predicted_end_price - start_price) / start_price) * 100 if not np.isnan(predicted_end_price - start_price) else 0.0
                future_data.append({
                    'Year': current_year,
                    'Start Date': last_date,
                    'End Date': end_date,
                    'Start Open Price': start_price,
                    'End Close Price': predicted_end_price,
                    'Profit/Loss (%)': profit_loss_percent
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
    
    # Define color map for each year
    color_map = {
        2020: '#FF6B6B', 2021: '#4ECDC4', 2022: '#45B7D1', 2023: '#96CEB4',
        2024: '#FFEEAD', 2025: '#D4A5A5',
    }
    
    unique_years = set(d['Year'] for d in profit_loss_data)
    for year in unique_years:
        year_data = [d for d in profit_loss_data if d['Year'] == year]
        if year_data:
            dates = [d['Start Date'] for d in year_data] + [year_data[-1]['End Date']]
            prices = [d['Start Open Price'] for d in year_data] + [year_data[-1]['End Close Price']]
            formatted_dates = [format_date_range(d['Start Date'], d['End Date']) for d in year_data] + [format_date_range(year_data[-1]['Start Date'], year_data[-1]['End Date'])]
            fig.add_trace(go.Scatter(
                x=formatted_dates, y=prices,
                mode='lines+markers', name=f"Year {year}",
                line=dict(width=2, dash='dash' if year == 2025 else 'solid', color=color_map[year]),  # Enforce color
                hovertemplate='Date Range: %{x}<br>Price: %{y}<extra></extra>'
            ), row=1, col=1)
    
    profits = [d['Profit/Loss (%)'] for d in profit_loss_data if 'Profit/Loss (%)' in d]
    dates = [format_date_range(d['Start Date'], d['End Date']) for d in profit_loss_data if 'Profit/Loss (%)' in d]
    fig.add_trace(go.Bar(
        x=dates, y=profits,
        name='Profit/Loss (%)',
        marker_color=['#90EE90' if p >= 0 else '#FFB6C1' for p in profits],
        opacity=0.7,
        hovertemplate='Date Range: %{x}<br>Profit/Loss: %{y}%<extra></extra>'
    ), row=2, col=1)
    
    fig.update_layout(
        title=f"Stock Price and Profit/Loss Analysis ({mode})",
        yaxis_title="Price",
        yaxis2_title="Profit/Loss (%)",
        hovermode="x unified",
        showlegend=True,
        height=900,
        template="plotly_white",
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            buttons=list([
                dict(label="All Years", method="update", args=[{"visible": [True] * len(unique_years)}]),
                *[dict(label=f"Year {year}", method="update", args=[{"visible": [year == y for y in unique_years]}]) for year in unique_years]
            ]),
            x=1.1, y=1.1
        )]
    )
    fig.update_xaxes(rangeslider_visible=True, title_text="", tickangle=-45)  # Rotate labels for clarity
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Profit/Loss (%)", row=2, col=1)
    
    # Download chart button
    chart_div = fig.to_html(include_plotlyjs='cdn', full_html=False)
    st.download_button(label="Download Chart", data=chart_div, file_name="stock_chart.html", mime="text/html")
    
    return fig

# Function to create styled table for a year
def create_year_table(year_data, compare_days):
    if not year_data:
        return None
    # Create a dictionary to store data by date range
    data = {'Year': year_data[0]['Year']}
    for i, d in enumerate(year_data):
        date_range = format_date_range(d['Start Date'], d['End Date'])
        data[f"{date_range} - Profit/Loss (%)"] = [d['Profit/Loss (%)']]
    df = pd.DataFrame(data)
    # Transpose for horizontal layout
    df = df.T.reset_index()
    df.columns = ['Date Range - Metric', year_data[0]['Year']]
    # Apply simple conditional styling for Profit/Loss
    def color_profit(val):
        if isinstance(val, (int, float)) and not pd.isna(val):
            color = '#90EE90' if val >= 0 else '#FFB6C1'
            return f'background-color: {color}'
        return ''
    styled_df = df.style.applymap(color_profit, subset=[year_data[0]['Year']])
    return styled_df

# Function to create prediction card
def create_prediction_card(pred_data):
    if not pred_data:
        return None
    pred_df = pd.DataFrame(pred_data).fillna(0)
    def color_profit(val):
        color = '#90EE90' if val >= 0 else '#FFB6C1'
        return f'background-color: {color}'
    styled_df = pred_df.style.applymap(color_profit, subset=['Profit/Loss (%)'])
    return styled_df

# Main app logic
if uploaded_file and run_analysis:
    st.header("Stock Pattern Analyzer")
    st.write("Analyze stock patterns and predict future trends. Current date: June 23, 2025.")

    # Load data with progress
    progress = st.progress(0)
    df = load_data(uploaded_file)
    if df is None:
        st.stop()
    progress.progress(50)

    # Calculate rolling profit/loss
    with st.spinner("Calculating profit/loss..."):
        profit_loss_data = calculate_rolling_profit_loss(df, compare_days, analysis_mode)
    progress.progress(100)

    # Create and display chart
    st.subheader("Price and Profit/Loss Visualization")
    fig = create_chart(df, profit_loss_data, analysis_mode)
    st.plotly_chart(fig, use_container_width=True)

    # Search bar for tables
    search_term = st.text_input("Search by Year or Date Range", key="search")

    # Group data by year and display separate tables
    st.subheader("Historical Profit/Loss by Year")
    years = sorted(set(d['Year'] for d in profit_loss_data) - {2025})
    for year in years:
        year_data = [d for d in profit_loss_data if d['Year'] == year and (str(year) in search_term or any(search_term.lower() in format_date_range(d['Start Date'], d['End Date']).lower() for d in [d]))]
        if year_data:
            styled_df = create_year_table(year_data, compare_days)
            if styled_df is not None:
                st.dataframe(styled_df, use_container_width=True)
                st.button("Copy to Clipboard", key=f"copy_{year}", on_click=lambda: st.write(styled_df.to_html(), unsafe_allow_html=True))

    # Display 2025 prediction
    st.subheader("2025 Prediction")
    current_year_data = [d for d in profit_loss_data if d['Year'] == 2025]
    if current_year_data:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            styled_pred_df = create_prediction_card(current_year_data)
            if styled_pred_df is not None:
                st.dataframe(styled_pred_df, use_container_width=True)
                csv = pd.DataFrame(current_year_data).to_csv(index=False)
                st.download_button(label="Download 2025 Prediction", data=csv, file_name="2025_prediction.csv", mime="text/csv")
                if st.button("Predict Again", key="predict_again"):
                    st.experimental_rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # Download all data
    all_df = pd.DataFrame(profit_loss_data).fillna(0)
    csv_all = all_df.to_csv(index=False)
    try:
        import openpyxl
        output = io.BytesIO()
        all_df.to_excel(output, index=False)
        excel_all = output.getvalue()
        st.download_button(label="Download as Excel", data=excel_all, file_name="all_profit_loss_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except ImportError:
        st.warning("Please install 'openpyxl' to enable Excel export: `pip install openpyxl`")
    with st.expander("Download All Data"):
        st.download_button(label="Download as CSV", data=csv_all, file_name="all_profit_loss_data.csv", mime="text/csv")

    # Help section
    with st.expander("Help"):
        st.write("""
        **Usage Guide**:
        - Upload a CSV/Excel file with stock data.
        - Set 'Compare Days' and select an 'Analysis Mode'.
        - Click 'Run Analysis' to process data.
        - Explore charts, tables, and download results.
        **Troubleshooting**:
        - Ensure data spans 2010–2025 with valid dates (YYYY-MM-DD).
        - Check for missing columns based on the selected mode.
        - If 'Index out of bounds' warnings appear, verify data continuity (e.g., no large gaps).
        - Install 'openpyxl' for Excel export: `pip install openpyxl`.
        """)

    # Footer
    st.markdown('<div style="text-align: center; padding: 10px; background-color: #F5F5F5; border-radius: 5px;">Version 1.0 | Developed with ❤️ by xAI</div>', unsafe_allow_html=True)

elif uploaded_file:
    st.info("Please click 'Run Analysis' to process the uploaded data.")
else:
    st.info("Please upload a CSV or Excel file to begin analysis.")
