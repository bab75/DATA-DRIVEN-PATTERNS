# ... (Previous imports and setup code remain unchanged)

# Sidebar for data source and settings
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Select Data Source", ["Upload CSV/XLSX", "Fetch Real-Time (Yahoo Finance)"], key="data_source")
symbol = st.sidebar.text_input("Stock Symbol (e.g., AAPL)", value=st.session_state.symbol, key="symbol_input")

# Single date range input with MM-DD-YYYY format
if 'date_range' not in st.session_state:
    st.session_state.date_range = (st.session_state.start_date, st.session_state.end_date)
date_range = st.sidebar.date_input("Select Date Range", value=st.session_state.date_range, key="date_range_input", format="MM-DD-YYYY")

# ... (Rest of sidebar settings remain unchanged)

# Load and validate data function
@st.cache_data
def load_data(primary_file, data_source, symbol, start_date, end_date):
    aapl_df = pd.DataFrame()
    pl_df = pd.DataFrame()
    
    if data_source == "Upload CSV/XLSX" and primary_file:
        try:
            if primary_file.name.endswith('.csv'):
                aapl_df = pd.read_csv(primary_file)
            elif primary_file.name.endswith('.xlsx'):
                aapl_df = pd.read_excel(primary_file)
            
            aapl_df.columns = aapl_df.columns.str.lower().str.strip()
            
            benchmark_cols = ['year', 'start date', 'end date', 'profit/loss (percentage)', 'profit/loss (value)']
            if any(col in aapl_df.columns for col in benchmark_cols):
                st.error(
                    "The uploaded file appears to be benchmark data. Please upload it as 'Benchmark Data' or upload a stock data file with columns: date, open, high, low, close, volume."
                )
                return pd.DataFrame(), pd.DataFrame()
            
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            actual_cols = aapl_df.columns.tolist()
            missing_cols = [col for col in required_cols if col not in actual_cols]
            if missing_cols:
                st.error(
                    f"Missing required columns in stock data: {', '.join(missing_cols)}. Please upload a file with columns: date, open, high, low, close, volume."
                )
                st.write("Available columns:", actual_cols)
                st.write("Sample data (first 5 rows):", aapl_df.head())
                st.write("Data types:", aapl_df.dtypes)
                return pd.DataFrame(), pd.DataFrame()
            
            aapl_df['date'] = pd.to_datetime(aapl_df['date'], errors='coerce', format='%m-%d-%Y')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                aapl_df[col] = pd.to_numeric(aapl_df[col], errors='coerce')
            
            aapl_df = aapl_df.interpolate(method='linear', limit_direction='both')
            
            if not aapl_df['date'].empty:
                min_date = aapl_df['date'].min()
                max_date = aapl_df['date'].max()
                st.sidebar.write(f"File date range: {min_date.strftime('%m-%d-%Y')} to {max_date.strftime('%m-%d-%Y')}")
                
                if start_date < min_date or end_date > max_date:
                    st.error(f"Selected data range ({start_date.strftime('%m-%d-%Y')} to {end_date.strftime('%m-%d-%Y')}) is outside the file's range ({min_date.strftime('%m-%d-%Y')} to {max_date.strftime('%m-%d-%Y')}).")
                    return pd.DataFrame(), pd.DataFrame()
                
                aapl_df = aapl_df[(aapl_df['date'] >= start_date) & (aapl_df['date'] <= end_date)]
                if aapl_df.empty:
                    st.error(f"No data available for the selected data range ({start_date.strftime('%m-%d-%Y')} to {end_date.strftime('%m-%d-%Y')}). Please adjust the date range.")
                    return pd.DataFrame(), pd.DataFrame()
                
                if len(aapl_df) < 52:
                    st.error(f"Insufficient data points ({len(aapl_df)}) in selected data range. Please select a range with at least 52 trading days.")
                    return pd.DataFrame(), pd.DataFrame()
            
            else:
                st.error("No valid dates found in the uploaded file. Please ensure the 'date' column contains valid dates.")
                return pd.DataFrame(), pd.DataFrame()
            
            if 'vwap' not in aapl_df.columns:
                st.warning("VWAP column is missing. VWAP plot will be skipped (optional).")
        
        except Exception as e:
            st.error(f"Error loading stock data: {str(e)}. Please check the file format and content.")
            st.write("Sample data (first 5 rows):", aapl_df.head() if not aapl_df.empty else "No data loaded")
            return pd.DataFrame(), pd.DataFrame()
    
    elif data_source == "Fetch Real-Time (Yahoo Finance)":
        try:
            symbol = symbol.strip()
            if not validate_symbol(symbol):
                st.error(f"Invalid symbol '{symbol}'. Please enter a single valid stock symbol (e.g., AAPL, MSFT, BRK.B).")
                return pd.DataFrame(), pd.DataFrame()
            
            aapl_df = yf.download(symbol, start=start_date, end=end_date + timedelta(days=1), progress=False)
            if aapl_df.empty:
                st.error(f"Failed to fetch {symbol} data from Yahoo Finance. Please check the symbol, date range, or internet connection.")
                return pd.DataFrame(), pd.DataFrame()
            
            if isinstance(aapl_df, pd.DataFrame) and aapl_df.columns.nlevels > 1:
                try:
                    aapl_df = aapl_df.xs(symbol, level=1, axis=1, drop_level=True)
                except KeyError:
                    st.error(f"Unexpected multi-index data for {symbol}. Please ensure a single valid symbol is entered (e.g., AAPL, not AAPL,MSFT).")
                    return pd.DataFrame(), pd.DataFrame()
            
            aapl_df = aapl_df.reset_index().rename(columns={
                'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
            })
            aapl_df['date'] = pd.to_datetime(aapl_df['date'], format='%m-%d-%Y')
            
            aapl_df = aapl_df.interpolate(method='linear', limit_direction='both')
            
            if len(aapl_df) < 52:
                st.error(f"Insufficient data points ({len(aapl_df)}) for {symbol}. Please select a wider date range (at least 52 trading days, e.g., 01-01-2025 to 06-24-2025).")
                return pd.DataFrame(), pd.DataFrame()
        
        except Exception as e:
            st.error(f"Error fetching {symbol} data from Yahoo Finance: {str(e)}. Please check the symbol, date range, or try uploading a file.")
            return pd.DataFrame(), pd.DataFrame()
    
    if secondary_file:
        try:
            if secondary_file.name.endswith('.csv'):
                pl_df = pd.read_csv(secondary_file)
            elif secondary_file.name.endswith('.xlsx'):
                pl_df = pd.read_excel(secondary_file)
            pl_df['Start Date'] = pd.to_datetime(pl_df['Start Date'], errors='coerce', format='%m-%d-%Y')
            pl_df['End Date'] = pd.to_datetime(pl_df['End Date'], errors='coerce', format='%m-%d-%Y')
            if pl_df[['Start Date', 'End Date', 'Profit/Loss (Percentage)']].isnull().any().any():
                st.warning("Benchmark data contains null values. Proceeding without benchmark.")
                pl_df = pd.DataFrame()
        except Exception as e:
            st.warning(f"Error loading benchmark data: {str(e)}. Proceeding without benchmark.")
    
    return aapl_df, pl_df

# Load data only if Submit is pressed and not already processed
if submit and not st.session_state.data_processed:
    st.session_state.data_loaded = True
    st.session_state.symbol = st.session_state.symbol_input
    st.session_state.start_date = pd.to_datetime(st.session_state.date_range_input[0], format='%m-%d-%Y')
    st.session_state.end_date = pd.to_datetime(st.session_state.date_range_input[1], format='%m-%d-%Y')
    st.session_state.date_range = (st.session_state.start_date, st.session_state.end_date)
    aapl_df, pl_df = load_data(primary_file, data_source, st.session_state.symbol, st.session_state.start_date, st.session_state.end_date)
    st.session_state.aapl_df = aapl_df
    st.session_state.pl_df = pl_df
    st.session_state.data_processed = True
elif not st.session_state.data_loaded:
    st.info("Please enter a symbol, select a data source, select a date range, and click 'Submit' to load data.")
    st.stop()

# ... (Rest of the script, including calculate_metrics, detect_consolidation_breakout, etc., remains unchanged but should use MM-DD-YYYY where applicable)
