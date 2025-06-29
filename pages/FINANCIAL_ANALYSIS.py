import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re

st.set_page_config(page_title="Financial Dashboard", layout="wide")
st.title("📊 Financial Analysis Dashboard")

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "selected_metrics" not in st.session_state:
    st.session_state.selected_metrics = []
if "selected_years" not in st.session_state:
    st.session_state.selected_years = []
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "analysis_generated" not in st.session_state:
    st.session_state.analysis_generated = False

# Sidebar upload
st.sidebar.header("🔧 Upload File")
uploaded_file = st.sidebar.file_uploader("Upload a .csv or .xls/.xlsx file", type=["csv", "xls", "xlsx"])
submitted = st.sidebar.button("📤 Submit")

@st.cache_data
def load_financial_data(file):
    try:
        if file.name.endswith(".csv"):
            raw = pd.read_csv(file, header=None)
        else:
            raw = pd.read_excel(file, header=None)

        # Debug: Save raw data
        raw.to_csv("raw_debug.csv", index=False)

        # Detect header row
        header_row = None
        for idx, row in raw.iterrows():
            count = sum(bool(re.search(r'(FY\s*20\d{2}|20\d{2})', str(cell))) for cell in row)
            if count >= 3:
                header_row = idx
                break
        if header_row is None:
            raise ValueError("Could not detect year headers (expected 'FY 20XX' or '20XX' formats).")

        # Extract and format
        df = raw.iloc[header_row+1:].copy()
        df.columns = raw.iloc[header_row]
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if df.empty or df.shape[1] < 2:
            raise ValueError("DataFrame is empty or has insufficient columns after initial cleaning.")

        # Set index to first column (metric names)
        if df.shape[1] > 0:
            df = df.set_index(df.columns[0])
        else:
            raise ValueError("No valid columns to set as index.")

        df.columns = df.columns.astype(str).str.strip()
        df.index = df.index.astype(str).str.strip()
        df = df.apply(pd.to_numeric, errors="coerce")

        # Transpose so years are index
        df = df.T
        df.index.name = "Year"
        # Extract year and validate
        df.index = df.index.astype(str).map(lambda x: re.search(r'20\d{2}', x).group(0) if re.search(r'20\d{2}', x) else np.nan)
        if df.index.isna().any():
            raise ValueError("Invalid year values in index after processing.")

        df = df.dropna(how="all")  # Drop rows with all NaN
        df = df.loc[:, ~df.isna().all()]  # Drop columns that are all NaN

        if df.empty:
            raise ValueError("No valid data after processing.")

        # Replace NaN with 0 early
        df = df.fillna(0)

        # Debug: Save cleaned DataFrame
        df.to_csv("cleaned_debug.csv")

        return df
    except Exception as e:
        st.session_state.error_message = f"Error loading file: {str(e)}"
        return None

# Main logic
if submitted and uploaded_file:
    with st.spinner("Processing file..."):
        st.session_state.error_message = None
        df = load_financial_data(uploaded_file)
        if df is None:
            st.error(st.session_state.error_message or "❌ Failed to load file.")
            st.stop()
        
        st.session_state.df = df
        st.success("✅ File loaded successfully!")

    # Debug: Show raw data and header
    raw = pd.read_excel(uploaded_file, header=None) if uploaded_file.name.endswith((".xls", ".xlsx")) else pd.read_csv(uploaded_file, header=None)
    st.write("Raw Data Preview:")
    st.dataframe(raw.head(10), use_container_width=True)
    header_row = next((idx for idx, row in raw.iterrows() if sum(bool(re.search(r'(FY\s*20\d{2}|20\d{2})', str(cell))) for cell in row) >= 3), None)
    if header_row is not None:
        st.write(f"Detected Header Row (index {header_row}):", raw.iloc[header_row].tolist())

    # Display cleaned DataFrame
    st.subheader("📋 Loaded Data")
    st.write("DataFrame Info:")
    st.write(df.info())
    st.dataframe(df, use_container_width=True)  # Ensure no NaN

    years = sorted([int(y) for y in df.index if str(y).isdigit()])
    all_metrics = [col for col in df.columns if df[col].abs().sum() > 0]  # Exclude zero-only columns
    preferred = ["Revenue", "Net Income", "Operating Income (Loss)", "Research & Development", "Restructuring Charges"]
    default_metrics = [m for m in preferred if m in all_metrics]

    with st.form("analysis_form"):
        st.subheader("📊 Analysis Configuration")
        metrics = st.multiselect("📌 Select metrics to analyze", all_metrics, default=default_metrics, key="metrics_select")
        if not years:
            st.warning("⚠️ No valid year labels found.")
            st.stop()
        yr_range = st.slider("📆 Select Year Range", min(years), max(years), (min(years), max(years)), key="year_range")
        analyze_button = st.form_submit_button("🔍 Generate Analysis")

    if analyze_button:
        try:
            with st.spinner("Generating analysis..."):
                # Persist selections
                st.session_state.selected_metrics = metrics
                st.session_state.selected_years = yr_range

                # Clear previous results
                st.session_state.analysis_results = None
                st.session_state.error_message = None
                
                selected_years = [str(y) for y in years if yr_range[0] <= y <= yr_range[1]]
                
                if not metrics:
                    st.warning("⚠️ Please select at least one metric to analyze.")
                    st.stop()
                
                if not selected_years:
                    st.warning("⚠️ No valid years in selected range.")
                    st.stop()

                results = {"charts": [], "table": None}
                
                # Generate charts
                for metric in metrics:
                    if metric not in df.columns:
                        st.warning(f"⚠️ Metric '{metric}' not found in data, skipping.")
                        continue
                        
                    series = df.loc[selected_years, metric].copy()
                    if series.isna().all() or len(series) == 0:
                        st.warning(f"⚠️ No valid data for '{metric}' in selected years.")
                        continue
                    
                    # Create chart
                    fig = px.line(series, markers=True, title=f"{metric} Over Time", 
                                 labels={"index": "Year", "value": metric})
                    
                    # Add trendline if enough data points
                    if len(series) > 1:
                        z = np.polyfit(range(len(series)), series.values, 1)
                        trend = np.poly1d(z)(range(len(series)))
                        fig.add_trace(go.Scatter(x=series.index, y=trend, name="Trendline",
                                               mode="lines", line=dict(dash="dot", color="gray")))
                    
                    # Add deviation flags
                    pct_change = series.pct_change().fillna(0) * 100
                    for i, (year, pct_val) in enumerate(pct_change.items()):
                        if abs(pct_val) > 30:
                            fig.add_vline(x=year, line=dict(color="red", dash="dot"),
                                         annotation_text="⚠️ Deviation", annotation_position="top right")
                    
                    results["charts"].append(fig)
                
                # Generate YOY table
                if metrics and selected_years:
                    table = df.loc[selected_years, metrics].copy()
                    for m in metrics:
                        if m in table.columns:
                            table[f"{m} YOY (%)"] = table[m].pct_change().round(4) * 100
                    table = table.fillna(0)  # Ensure no NaN in table
                    results["table"] = table
                
                # Store results
                st.session_state.analysis_results = results
                st.session_state.analysis_generated = True
                
        except Exception as e:
            st.session_state.error_message = f"Error generating analysis: {str(e)}"
            st.error(st.session_state.error_message)
            st.session_state.analysis_results = None
            st.session_state.analysis_generated = False

# Display results
if st.session_state.analysis_results is not None and st.session_state.analysis_generated:
    st.success("✅ Analysis generated successfully!")
    
    # Display charts
    if "charts" in st.session_state.analysis_results and st.session_state.analysis_results["charts"]:
        for fig in st.session_state.analysis_results["charts"]:
            st.plotly_chart(fig, use_container_width=True)
    
    # Display table
    if "table" in st.session_state.analysis_results and st.session_state.analysis_results["table"] is not None:
        st.subheader("📊 YOY Change Table")
        st.dataframe(st.session_state.analysis_results["table"], use_container_width=True)
        
        csv = st.session_state.analysis_results["table"].to_csv().encode("utf-8")
        st.download_button("📥 Download YOY Summary", csv, "yoy_summary.csv", "text/csv")
elif st.session_state.df is not None:
    st.info("ℹ️ Configure analysis settings and click 'Generate Analysis' to view results.")
else:
    st.info("📎 Upload a file and click Submit to begin.")
