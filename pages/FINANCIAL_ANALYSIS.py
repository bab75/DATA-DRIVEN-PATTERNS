import streamlit as st
import pandas as pd
import re

# Set page config
st.set_page_config(page_title="Financial Data Viewer", layout="wide")
st.title("📊 Financial Data Viewer")

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "report_name" not in st.session_state:
    st.session_state.report_name = None
if "error_message" not in st.session_state:
    st.session_state.error_message = None

# Function to load and process financial data
@st.cache_data
def load_financial_data(file):
    try:
        # Read Excel file
        raw = pd.read_excel(file, header=None)
        
        # Set report name to "Annual Analysis Report" as requested
        report_name = "Annual Analysis Report"
        raw_a1 = raw.iloc[0, 0]  # For debugging A1 value

        # Detect header row with years (look for FY 20XX or 20XX-12-31)
        header_row = None
        for idx, row in raw.iterrows():
            year_count = sum(bool(re.search(r'(FY\s*20\d{2}|20\d{2}-\d{2}-\d{2})', str(cell))) for cell in row)
            if year_count >= 3:
                header_row = idx
                break
        if header_row is None:
            return None, None, "No year headers (e.g., FY 20XX or 20XX-12-31) found in file."

        # Set up DataFrame
        df = raw.iloc[header_row+1:].copy()
        df.columns = raw.iloc[header_row]
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if df.empty or df.shape[1] < 2:
            return None, None, "DataFrame is empty or has insufficient columns."
        
        # Set first column as index (metrics)
        df = df.set_index(df.columns[0])
        df.columns = df.columns.astype(str).str.strip()
        # Replace 'nan' in index with 'Unknown'
        df.index = [str(idx).strip() if str(idx).strip().lower() != "nan" else "Unknown" for idx in df.index]
        
        # Convert to numeric
        df = df.apply(pd.to_numeric, errors="coerce")
        # Drop columns (metrics) with all NaN or all zero values
        df = df.dropna(axis=1, how="all")
        # Apply zero filtering before transposition
        df = df.loc[:, (df.abs() > 1e-10).any(axis=0)]  # Drop columns with all zeros
        # Transpose to have years as index
        df = df.T
        df.index.name = "Year"
        # Extract year from FY 20XX or 20XX-12-31
        df.index = [re.search(r'20\d{2}', str(x)).group(0) if re.search(r'20\d{2}', str(x)) else None for x in df.index]
        # Drop rows with None index (invalid years)
        df = df[df.index.notnull()]
        df = df.astype(float)
        # Drop rows (years) with all NaN or all zero values
        df = df.dropna(axis=0, how="all")
        df = df.loc[(df.abs() > 1e-10).any(axis=1)]  # Drop rows with all zeros
        # Ensure index and columns are strings to avoid JSON issues
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)

        if df.empty:
            return None, None, "No valid data after processing."
        
        return df, report_name, None, raw_a1
    except Exception as e:
        return None, None, f"Error loading file: {str(e)}", None

# Sidebar for file upload
st.sidebar.header("🔧 Upload File")
uploaded_file = st.sidebar.file_uploader("Upload a .xlsx file", type=["xlsx"])
if st.sidebar.button("📤 Submit") and uploaded_file:
    st.session_state.df, st.session_state.report_name, st.session_state.error_message, raw_a1 = load_financial_data(uploaded_file)
    if st.session_state.df is None:
        st.error(st.session_state.error_message or "❌ Failed to load file.")
    else:
        st.success("✅ File loaded successfully!")
        # Debug output
        with st.expander("Debug Info"):
            st.write(f"DataFrame shape: {st.session_state.df.shape}")
            st.write(f"Years: {sorted([int(y) for y in st.session_state.df.index if str(y).isdigit()])}")
            st.write(f"Metrics (first 5): {st.session_state.df.columns[:5].tolist()}")
            st.write(f"Report Name: {st.session_state.report_name}")
            st.write(f"Raw A1 Value: {raw_a1}")

# Display filtered DataFrame
if st.session_state.df is not None:
    st.subheader(f"📋 Financial Data: {st.session_state.report_name}")
    # Filter DataFrame to include only metrics with some non-null data
    valid_metrics = [col for col in st.session_state.df.columns if st.session_state.df[col].notna().any()]
    filtered_df = st.session_state.df[valid_metrics]
    # Filter years to include only those with some non-null, non-zero data
    filtered_df = filtered_df.loc[(filtered_df.notna().any(axis=1)) & (filtered_df.abs() > 1e-10).any(axis=1)]
    if not filtered_df.empty:
        # Replace NaN with empty string for display to avoid JSON issues
        display_df = filtered_df.fillna("")
        st.dataframe(display_df)
        # Add download button for filtered DataFrame
        csv = filtered_df.to_csv().encode("utf-8")
        st.download_button("📥 Download Data", csv, f"{st.session_state.report_name}_data.csv", "text/csv")
    else:
        st.warning("⚠️ No valid data to display after filtering.")
elif st.session_state.error_message:
    st.error(st.session_state.error_message)
else:
    st.info("📎 Upload a file and click Submit to begin.")
