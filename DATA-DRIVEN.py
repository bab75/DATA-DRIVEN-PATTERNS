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
import pandas_market_calendars as mcal

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

# Initialize session state
if 'dframe' not in st.session_state:
    st.session_state.dframe = None
if 'profit_loss_data' not in st.session_state:
    st.session_state.profit_loss_data = None
if 'display_mode' not in st.session_state:
    st.session_state.display_mode = "Show All Columns"
if 'selected_month' not in st.session_state:
    st.session_state.selected_month = month_name[1][:3]
if 'transpose_table' not in st.session_state:
    st.session_state.transpose_table = False
if 'file_key' not in st.session_state:
    st.session_state.file_key = 0
if 'profit_loss_unit' not in st.session_state:
    st.session_state.profit_loss_unit = "Percentage"  # Default to percentage

# Sidebar: Control Panel
with st.sidebar:
    st.header("Control Panel")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'], help="Upload a file with 'date', 'open', 'close' columns.", key=f"file_uploader_{st.session_state.file_key}")
    compare_days = st.number_input("Compare Days (1-30)", min_value=1, max_value=30, value=2, help="Number of trading days to compare for profit/loss within each year.")
    analysis_mode = st.radio("Analysis Mode", ["Raw Data (Open vs. Close)", "Open/Close/High/Low", "Technical Indicators"],
                             help="Choose the data features for analysis.")
    run_analysis = st.button("Run Analysis")
    if st.button("Reset", key="reset"):
        st.session_state.clear()
        st.session_state.file_key += 1
        st.experimental_rerun()
    if st.button("Mode Description"):
        st.write("""
        - **Raw Data (Open vs. Close)**: Analyzes profit/loss using opening and closing prices.
        - **Open/Close/High/Low**: Considers high and low prices within the period for potential profit/loss.
        - **Technical Indicators**: Uses indicators (ma20, ma50, macd, rsi, atr, vwap) to adjust predictions.
        """)

# Function to get trading days
def get_trading_days(start_date, end_date, exchange='NYSE'):
    nyse = mcal.get_calendar(exchange)
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_days = mcal.date_range(schedule, frequency='1D').date
    return trading_days

# Function to get the nth trading day from a start date
def get_nth_trading_day(start_date, n, year, exchange='NYSE'):
    start_of_year = datetime(year, 1, 1).date()
    end_of_year = datetime(year, 12, 31).date()
    trading_days = get_trading_days(start_of_year, end_of_year)
    start_idx = trading_days.searchsorted(start_date)
    if start_idx + n - 1 >= len(trading_days):
        return None
    return trading_days[start_idx + n - 1]

# Function to load and preprocess data
def load_data(file):
    with st.spinner("Loading data..."):
        time.sleep(1)  # Simulate loading
    try:
        if file.name.endswith('.csv'):
            dframe = pd.read_csv(file)
        else:
            dframe = pd.read_excel(file)
        dframe['date'] = pd.to_datetime(dframe['date']).dt.date
        dframe = dframe.sort_values('date').reset_index(drop=True)
        required_columns = ['date', 'open', 'close']
        if analysis_mode == "Open/Close/High/Low":
            required_columns.extend(['high', 'low'])
        elif analysis_mode == "Technical Indicators":
            required_columns.extend(['high', 'low', 'ma20', 'ma50', 'macd', 'rsi', 'atr', 'vwap'])
        if not all(col in dframe.columns for col in required_columns):
            missing = [col for col in required_columns if col not in dframe.columns]
            st.error(f"Missing required columns for {analysis_mode}: {', '.join(missing)}")
            return None
        if len(dframe) < compare_days:
            st.error(f"Dataset has {len(dframe)} rows, but at least {compare_days} are required.")
            return None
        # Filter for trading days
        min_date = dframe['date'].min()
        max_date = dframe['date'].max()
        trading_days = get_trading_days(min_date, max_date)
        dframe = dframe[dframe['date'].isin(trading_days)].copy()
        # Additional validation to remove any residual weekend dates
        nyse = mcal.get_calendar('NYSE')
        schedule = nyse.schedule(start_date=min_date, end_date=max_date)
        valid_trading_days = mcal.date_range(schedule, frequency='1D').date
        dframe = dframe[dframe['date'].isin(valid_trading_days)]
        if len(dframe) < compare_days:
            st.error(f"After filtering for trading days, dataset has {len(dframe)} rows, but at least {compare_days} are required.")
            return None
        st.info(f"Loaded {len(dframe)} trading days with columns: {', '.join(dframe.columns)}")
        return dframe
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

# Function to get next available trading date
def get_next_trading_date(df, current_date, trading_days):
    year = current_date.year
    next_date = current_date + timedelta(days=1)
    while next_date.year == year and next_date not in trading_days and (next_date - df['date'].iloc[-1]).days < 365:
        next_date += timedelta(days=1)
    return next_date if next_date.year == year and next_date in df['date'].values else None

# Function to generate month-aligned trading periods
def generate_monthly_periods(compare_days, year=2024, exchange='NYSE'):
    periods = []
    nyse = mcal.get_calendar(exchange)
    start_date = datetime(year, 1, 1).date()
    end_date = datetime(year, 12, 31).date()
    trading_days = get_trading_days(start_date, end_date)
    if not trading_days.size:
        return periods
    i = 0
    while i < len(trading_days):
        start_trading_date = trading_days[i]
        end_trading_date = get_nth_trading_day(start_trading_date, compare_days, year)
        if end_trading_date:
            periods.append(format_date_range(start_trading_date, end_trading_date))
        i += 1
    return periods

# Function to calculate rolling profit/loss within each year
def calculate_rolling_profit_loss(dframe, compare_days, mode):
    profit_loss_data = []
    current_year = datetime.now().year  # 2025
    current_date = datetime.now().date()  # June 23, 2025, 10:21 PM EDT
    years = sorted(set(dframe['date'].apply(lambda x: x.year)))
    
    for year in years:
        year_df = dframe[dframe['date'].apply(lambda x: x.year) == year].copy()
        if len(year_df) < max(1, compare_days):
            st.warning(f"Year {year} has insufficient data ({len(year_df)} days). Minimum required: {max(1, compare_days)}")
            continue
        
        # Get trading days for the year
        start_date = datetime(year, 1, 1).date()
        end_date = datetime(year, 12, 31).date() if year < current_year else current_date
        trading_days = get_trading_days(start_date, end_date)
        
        for month in range(1, 13):
            month_start = datetime(year, month, 1).date()
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month-1]
            if year % 4 == 0 and month == 2:
                days_in_month = 29
            month_end = datetime(year, month, days_in_month).date()
            month_end = min(month_end, current_date) if year == current_year else month_end
            month_trading_days = trading_days[(trading_days >= month_start) & (trading_days <= month_end)]
            
            if not month_trading_days.size:
                continue
                
            last_end_date = month_start - timedelta(days=1)
            i = 0
            while i < len(month_trading_days):
                start_date = month_trading_days[i]
                if start_date <= last_end_date:
                    i += 1
                    continue
                end_date = get_nth_trading_day(start_date, compare_days, year)
                if not end_date or (year == current_year and end_date > current_date):
                    break
                
                # Find actual data for this period
                period_data = year_df[
                    (year_df['date'] >= start_date) & 
                    (year_df['date'] <= end_date)
                ]
                
                if len(period_data) >= compare_days:  # Require full compare_days
                    start_price = period_data.iloc[0]['open']
                    end_price = period_data.iloc[-1]['close']
                    # Debug output with validation
                    st.write(f"Debug - {year} {format_date_range(start_date, end_date)}: Open={start_price}, Close={end_price}, Data Points={len(period_data)}, In dframe={start_date in year_df['date'].values and end_date in year_df['date'].values}")
                    
                    # Mode-specific profit/loss calculation
                    if mode == "Raw Data (Open vs. Close)":
                        profit_loss_percent = ((end_price - start_price) / start_price) * 100 if not np.isnan(end_price - start_price) else None
                        profit_loss_value = end_price - start_price if not np.isnan(end_price - start_price) else None
                    elif mode == "Open/Close/High/Low":
                        max_high = period_data['high'].max()
                        min_low = period_data['low'].min()
                        profit_loss_percent = ((max_high - min_low) / start_price) * 100 if not np.isnan(max_high - min_low) else None
                        profit_loss_value = max_high - min_low if not np.isnan(max_high - min_low) else None
                    elif mode == "Technical Indicators":
                        rsi_avg = period_data['rsi'].mean()
                        macd_avg = period_data['macd'].mean()
                        weight = 1.0
                        if rsi_avg > 70:
                            weight *= 0.8
                        elif rsi_avg < 30:
                            weight *= 1.2
                        if macd_avg > 0:
                            weight *= 1.1
                        elif macd_avg < 0:
                            weight *= 0.9
                        profit_loss_percent = ((end_price - start_price) / start_price) * 100 * weight if not np.isnan(end_price - start_price) else None
                        profit_loss_value = (end_price - start_price) * weight if not np.isnan(end_price - start_price) else None
                    
                    if profit_loss_percent is not None and profit_loss_value is not None:
                        profit_loss_data.append({
                            'Year': year,
                            'Start Date': start_date,
                            'End Date': end_date,
                            'Profit/Loss (Percentage)': profit_loss_percent,
                            'Profit/Loss (Value)': profit_loss_value
                        })
                    last_end_date = end_date
                
                i += 1

    if not profit_loss_data:
        st.warning("No valid periods found for profit/loss calculation. Check data for gaps or insufficient trading days.")
        return []

    # ML Prediction for 2025
    historical_data = pd.DataFrame([d for d in profit_loss_data if d['Year'] != current_year])
    future_data = []
    if len(historical_data) > 5 and current_year in years:
        historical_data['period_index'] = range(len(historical_data))
        X = historical_data[['period_index']].values
        y_percent = historical_data['Profit/Loss (Percentage)'].values
        y_value = historical_data['Profit/Loss (Value)'].values
        model_percent = LinearRegression()
        model_value = LinearRegression()
        try:
            model_percent.fit(X, y_percent)
            model_value.fit(X, y_value)
            last_date = dframe[dframe['date'].apply(lambda x: x.year) == current_year]['date'].iloc[-1] if not dframe[dframe['date'].apply(lambda x: x.year) == current_year].empty else current_date
            if last_date.year == current_year:
                trading_days = get_trading_days(last_date, datetime(current_year, 12, 31).date())
                next_period_idx = len(historical_data)
                predicted_pl_percent = model_percent.predict([[next_period_idx]])[0]
                predicted_pl_value = model_value.predict([[next_period_idx]])[0]
                end_date = last_date
                for _ in range(compare_days - 1):
                    next_date = get_next_trading_date(dframe, end_date, trading_days)
                    if next_date:
                        end_date = next_date
                    else:
                        break
                future_data.append({
                    'Year': current_year,
                    'Start Date': last_date,
                    'End Date': end_date,
                    'Profit/Loss (Percentage)': predicted_pl_percent,
                    'Profit/Loss (Value)': predicted_pl_value,
                    'Prediction': True
                })
        except Exception as e:
            st.warning(f"ML prediction failed: {str(e)}. Using last known value.")
            predicted_pl_percent = 0.0
            predicted_pl_value = 0.0
    profit_loss_data.extend(future_data)

    return profit_loss_data

# Function to create interactive Plotly chart
def create_chart(dframe, profit_loss_data, mode, unit):
    if not profit_loss_data or dframe.empty:
        st.warning("No data available to display chart.")
        return None
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.1, 
                            subplot_titles=['Price Patterns', f'Profit/Loss ({unit})'],
                            row_heights=[0.7, 0.3])
        
        color_map = {
            2020: '#FF6B6B', 2021: '#4ECDC4', 2022: '#45B7D1', 2023: '#96CEB4',
            2024: '#FFEEAD', 2025: '#D4A5A5',
        }
        
        unique_years = set(d['Year'] for d in profit_loss_data)
        for year in unique_years:
            year_data = [d for d in profit_loss_data if d['Year'] == year and (not 'Prediction' in d or not d['Prediction'])]
            if year_data:
                dates = [d['Start Date'] for d in year_data]
                valid_dates = [d for d in dates if d in dframe['date'].values]
                if valid_dates:
                    if mode == "Raw Data (Open vs. Close)":
                        prices = [dframe[dframe['date'] == d]['open'].iloc[0] for d in valid_dates if d in dframe['date'].values and not np.isnan(dframe[dframe['date'] == d]['open'].iloc[0])]
                        if prices:
                            fig.add_trace(go.Scatter(
                                x=valid_dates,
                                y=prices,
                                mode='lines+markers', name=f"Year {year} (Open)",
                                line=dict(width=2, dash='solid', color=color_map[year]),
                                hovertemplate='Date: %{x}<br>Price: %{y}<extra></extra>'
                            ), row=1, col=1)
                    elif mode == "Open/Close/High/Low":
                        highs = [dframe[dframe['date'] == d]['high'].iloc[0] for d in valid_dates if d in dframe['date'].values and not np.isnan(dframe[dframe['date'] == d]['high'].iloc[0])]
                        lows = [dframe[dframe['date'] == d]['low'].iloc[0] for d in valid_dates if d in dframe['date'].values and not np.isnan(dframe[dframe['date'] == d]['low'].iloc[0])]
                        if highs and lows:
                            fig.add_trace(go.Scatter(
                                x=valid_dates,
                                y=highs,
                                mode='lines+markers', name=f"Year {year} (High)",
                                line=dict(width=2, dash='solid', color=color_map[year]),
                                hovertemplate='Date: %{x}<br>High: %{y}<extra></extra>'
                            ), row=1, col=1)
                            fig.add_trace(go.Scatter(
                                x=valid_dates,
                                y=lows,
                                mode='lines+markers', name=f"Year {year} (Low)",
                                line=dict(width=1, dash='dot', color=color_map[year]),
                                hovertemplate='Date: %{x}<br>Low: %{y}<extra></extra>'
                            ), row=1, col=1)
                    elif mode == "Technical Indicators":
                        ma20 = [dframe[dframe['date'] == d]['ma20'].iloc[0] for d in valid_dates if d in dframe['date'].values and not np.isnan(dframe[dframe['date'] == d]['ma20'].iloc[0])]
                        if ma20:
                            fig.add_trace(go.Scatter(
                                x=valid_dates,
                                y=ma20,
                                mode='lines+markers', name=f"Year {year} (MA20)",
                                line=dict(width=2, dash='solid', color=color_map[year]),
                                hovertemplate='Date: %{x}<br>MA20: %{y}<extra></extra>'
                            ), row=1, col=1)
        
        profits = [d[f'Profit/Loss ({unit})'] for d in profit_loss_data if d[f'Profit/Loss ({unit})'] is not None]
        dates = [d['Start Date'] for d in profit_loss_data if d[f'Profit/Loss ({unit})'] is not None]
        unit_symbol = "%" if unit == "Percentage" else ""
        if profits and dates:
            fig.add_trace(go.Bar(
                x=dates,
                y=profits,
                name=f'Profit/Loss ({unit})',
                marker_color=['#006400' if p >= 0 else '#8B0000' for p in profits],
                opacity=0.9,
                hovertemplate=f'Date: %{{x}}<br>Price: %{{y}}{unit_symbol}<extra></extra>'
            ), row=2, col=1)
        
        fig.update_layout(
            title=f"Stock Price and Profit/Loss Analysis ({mode})",
            yaxis_title="Price",
            yaxis2_title=f"Profit/Loss ({unit_symbol})",
            xaxis_title="Date",
            hovermode="x unified",
            showlegend=True,
            height=900,
            template="plotly_white",
            updatemenus=[dict(
                type="dropdown",
                direction="down",
                buttons=list([
                    dict(label="All Years", method="update", args=[{"visible": [True] * len(fig.data)}]),
                    *[dict(label=f"Year {year}", method="update", args=[{"visible": [d.name.startswith(f"Year {year}") for d in fig.data]}]) for year in unique_years]
                ]),
                x=1.1, y=1.1
            )]
        )
        fig.update_xaxes(tickformat="%Y-%m-%d", tickangle=45)  # Adjusted to "2020-01-10" for consistency
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text=f"Profit/Loss ({unit_symbol})", row=2, col=1)
        
        # Download chart button
        chart_div = fig.to_html(include_plotlyjs='cdn', full_html=False)
        st.download_button(label="Download Chart", data=chart_div, file_name="stock_chart.html", mime="text/html")
        
        return fig
    except Exception as e:
        st.error(f"Chart rendering failed: {str(e)}. Please check data or debug output.")
        return None

# Function to create styled table for all years
def create_year_table(profit_loss_data, compare_days, unit):
    if not profit_loss_data:
        return None, None
    
    # Generate month-aligned trading periods using trading days
    current_year = datetime.now().year
    calendar_columns = generate_monthly_periods(compare_days, year=current_year, exchange='NYSE')
    
    # Create pivot table
    years = sorted(set(d['Year'] for d in profit_loss_data if d['Year'] < datetime.now().year or (d['Year'] == datetime.now().year and not 'Prediction' in d)))
    table_data = {'Year': years}
    
    for col in calendar_columns:
        table_data[col] = [0] * len(years)
    
    # Fill with actual data
    for d in profit_loss_data:
        if d['Year'] < datetime.now().year or (d['Year'] == datetime.now().year and not 'Prediction' in d):
            year_idx = years.index(d['Year'])
            date_range = format_date_range(d['Start Date'], d['End Date'])
            if date_range in table_data:
                table_data[date_range][year_idx] = d[f'Profit/Loss ({unit})'] if d[f'Profit/Loss ({unit})'] is not None else 0
    
    df = pd.DataFrame(table_data)
    
    # Apply color styling
    numeric_cols = [col for col in df.columns if col != 'Year']
    styled_df = df.style.applymap(color_profit, subset=numeric_cols)
    
    return df, styled_df

# Function to create prediction card
def create_prediction_card(pred_data, unit):
    if not pred_data:
        return None
    pred_df = pd.DataFrame(pred_data).fillna(0)
    pred_df = pred_df[['Year', 'Start Date', 'End Date', f'Profit/Loss ({unit})']]
    styled_df = pred_df.style.applymap(color_profit, subset=[f'Profit/Loss ({unit})'])
    return styled_df

# Global color_profit function
def color_profit(val):
    """Apply color styling to profit/loss values"""
    if isinstance(val, (int, float)) and not pd.isna(val):
        return 'background-color: #90EE90' if val >= 0 else 'background-color: #FFB6C1'
    return ''

# Main app logic
if uploaded_file and run_analysis:
    # Load and process data
    progress = st.progress(0)
    dframe = load_data(uploaded_file)
    if dframe is None:
        st.stop()
    st.session_state.dframe = dframe
    progress.progress(50)

    # Calculate rolling profit/loss
    with st.spinner("Calculating profit/loss..."):
        profit_loss_data = calculate_rolling_profit_loss(dframe, compare_days, analysis_mode)
    st.session_state.profit_loss_data = pd.DataFrame(profit_loss_data)
    progress.progress(100)

# Display results
if st.session_state.dframe is not None and st.session_state.profit_loss_data is not None:
    st.header("Stock Pattern Analyzer")
    st.write(f"Analyze stock patterns and predict future trends. Current date: June 23, 2025, 10:21 PM EDT")

    # Profit/Loss unit selection
    def update_profit_loss_unit():
        st.session_state.profit_loss_unit = "Percentage" if st.session_state.profit_loss_unit_check else "Value"
    st.checkbox("Show Profit/Loss as Percentage (uncheck for Absolute Value)", value=st.session_state.profit_loss_unit == "Percentage", key="profit_loss_unit_check", on_change=update_profit_loss_unit)
    unit = st.session_state.profit_loss_unit

    # Create and display chart
    st.subheader("Price and Profit/Loss Visualization")
    fig = create_chart(st.session_state.dframe, st.session_state.profit_loss_data.to_dict('records'), analysis_mode, unit)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Chart could not be generated. Please check the data or debug output.")

    # Display table for all years
    st.subheader("Historical Profit/Loss by Year")
    years = sorted(set(st.session_state.profit_loss_data['Year'].unique()))
    for year in years:
        if year < datetime.now().year:
            year_data = st.session_state.profit_loss_data[st.session_state.profit_loss_data['Year'] == year].to_dict('records')
        elif year == datetime.now().year:
            year_data = [d for d in st.session_state.profit_loss_data.to_dict('records') if d['Year'] == year and not 'Prediction' in d]
        else:
            continue
        if year_data:
            df, styled_df = create_year_table(year_data, compare_days, unit)
            if df is not None and not df.empty:
                st.write(f"📊 **Table Summary for Year {year}**: {len(df.columns)-1} periods")
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.warning(f"No table data available for year {year}.")

    # Display 2025 prediction
    current_year = datetime.now().year
    if current_year in years:
        st.subheader("2025 Prediction")
        pred_data = [d for d in st.session_state.profit_loss_data.to_dict('records') if d['Year'] == current_year and 'Prediction' in d]
        if pred_data:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                styled_pred_df = create_prediction_card(pred_data, unit)
                if styled_pred_df is not None:
                    st.dataframe(styled_pred_df, use_container_width=True)
                    csv = pd.DataFrame(pred_data).to_csv(index=False)
                    st.download_button(label="Download 2025 Prediction", data=csv, file_name="2025_prediction.csv", mime="text/csv")
                st.markdown('</div>', unsafe_allow_html=True)

    # Download all data
    all_df = st.session_state.profit_loss_data.fillna(0)
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
        - Set 'Compare Days' (number of trading days) and select an 'Analysis Mode'.
        - Click 'Run Analysis' to process data.
        - Toggle 'Show Profit/Loss as Percentage' to switch between percentage and absolute value.
        - Explore charts, tables, and download results.
        **Analysis Modes**:
        - **Raw Data**: Uses open/close prices for profit/loss.
        - **Open/Close/High/Low**: Uses high/low prices for max potential profit/loss.
        - **Technical Indicators**: Adjusts profit/loss based on RSI and MACD signals.
        **Table Display**:
        - Profit/loss values are color-coded: green for positive, red for negative.
        - Each year has its own table, with periods based on trading days, excluding weekends and holidays (e.g., Jan 1st).
        - 2025 shows historical data up to current date, with predictions in a separate section.
        **Troubleshooting**:
        - Ensure data spans 2010–2025 with valid dates (YYYY-MM-DD).
        - Verify required columns for the selected mode.
        - Check for sufficient trading days.
        - Install 'openpyxl' for Excel export: `pip install openpyxl`.
        - Install 'pandas_market_calendars' for trading day calculations: `pip install pandas_market_calendars`.
        """)

    # Footer
    st.markdown('<div style="text-align: center; padding: 10px; background-color: #F5F5F5; border-radius: 5px;">Version 3.3 | Developed with ❤️ by xAI</div>', unsafe_allow_html=True)

elif uploaded_file:
    st.info("Please click 'Run Analysis' to process the uploaded data.")
else:
    st.info("Please upload a CSV or Excel file to begin analysis.")
