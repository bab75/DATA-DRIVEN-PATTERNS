import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import numpy as np

# Set page config
st.set_page_config(page_title="Financial Analysis Dashboard", layout="wide")
st.title("📊 Financial Analysis Dashboard")

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "report_name" not in st.session_state:
    st.session_state.report_name = None
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "metrics" not in st.session_state:
    st.session_state.metrics = []
if "years" not in st.session_state:
    st.session_state.years = []
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# Function to load and process financial data
@st.cache_data
def load_financial_data(file):
    try:
        # Read Excel file
        raw = pd.read_excel(file, header=None)
        
        # Extract report name from cell A1 and combine with "Annual Analysis Report"
        a1_value = str(raw.iloc[0, 0]).strip() if not pd.isna(raw.iloc[0, 0]) and str(raw.iloc[0, 0]).strip() else "3M CO"
        report_name = f"{a1_value} Annual Analysis Report"
        raw_a1 = raw.iloc[0, 0]  # For debugging A1 value

        # Detect header row with years (look for FY 20XX or 20XX-12-31)
        header_row = None
        for idx, row in raw.iterrows():
            year_count = sum(bool(re.search(r'(FY\s*20\d{2}|20\d{2}-\d{2}-\d{2})', str(cell))) for cell in row)
            if year_count >= 3:
                header_row = idx
                break
        if header_row is None:
            return None, None, "No year headers (e.g., FY 20XX or 20XX-12-31) found in file.", None

        # Set up DataFrame
        df = raw.iloc[header_row+1:].copy()
        df.columns = raw.iloc[header_row]
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if df.empty or df.shape[1] < 2:
            return None, None, "DataFrame is empty or has insufficient columns.", None
        
        # Set first column as index (metrics)
        df = df.set_index(df.columns[0])
        df.columns = df.columns.astype(str).str.strip()
        # Replace 'nan' in index with 'Unknown'
        df.index = [str(idx).strip() if str(idx).strip().lower() != "nan" else "Unknown" for idx in df.index]
        
        # Convert to numeric
        df = df.apply(pd.to_numeric, errors="coerce")
        # Drop columns (metrics) with all NaN or all zero values
        df = df.dropna(axis=1, how="all")
        df = df.loc[:, (df.abs() > 1e-10).any(axis=0)]  # Drop columns with all zeros
        # Debug: Log columns before transposition
        with st.expander("Debug: Columns Before Filtering"):
            st.write(f"All columns: {df.columns.tolist()}")
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
            return None, None, "No valid data after processing.", None
        
        return df, report_name, None, raw_a1
    except Exception as e:
        return None, None, f"Error loading file: {str(e)}", None

# Sidebar for file upload and buttons
st.sidebar.header("🔧 Upload File")
uploaded_file = st.sidebar.file_uploader("Upload a .xlsx file", type=["xlsx"])

# Create two columns for buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    submit_button = st.button("📤 Submit")
with col2:
    clear_button = st.button("🗑️ Clear Analysis")

# Handle clear analysis
if clear_button:
    st.session_state.df = None
    st.session_state.report_name = None
    st.session_state.error_message = None
    st.session_state.metrics = []
    st.session_state.years = []
    st.session_state.analysis_results = None
    st.success("✅ Analysis cleared! Upload a new file to start again.")

# Handle file submission
if submit_button and uploaded_file:
    st.session_state.df, st.session_state.report_name, st.session_state.error_message, raw_a1 = load_financial_data(uploaded_file)
    if st.session_state.df is None:
        st.error(st.session_state.error_message or "❌ Failed to load file.")
    else:
        st.success("✅ File loaded successfully!")
        # Store available metrics and years
        st.session_state.metrics = [col for col in st.session_state.df.columns if st.session_state.df[col].notna().any()]
        st.session_state.years = sorted([int(y) for y in st.session_state.df.index if str(y).isdigit()])
        # Debug output
        with st.expander("Debug Info"):
            st.write(f"DataFrame shape: {st.session_state.df.shape}")
            st.write(f"Years: {st.session_state.years}")
            st.write(f"Metrics (first 5): {st.session_state.metrics[:5]}")
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

# Analysis form
if st.session_state.df is not None:
    with st.form("analysis_form"):
        st.subheader("📊 Analysis Configuration")
        metrics = st.multiselect(
            "📌 Select metrics to analyze",
            options=st.session_state.metrics,
            default=[m for m in ["Revenue", "Net Income", "Gross Profit"] if m in st.session_state.metrics]
        )
        if not st.session_state.years:
            st.warning("⚠️ No valid years found in data.")
            st.stop()
        year_range = st.slider(
            "📆 Select Year Range",
            min_value=min(st.session_state.years),
            max_value=max(st.session_state.years),
            value=(min(st.session_state.years), max(st.session_state.years))
        )
        analyze_button = st.form_submit_button("🔍 Generate Analysis")

    # Process analysis
    if analyze_button and metrics:
        try:
            st.session_state.error_message = None
            st.session_state.analysis_results = {"charts": [], "table": None}
            
            valid_years = [str(y) for y in st.session_state.years if year_range[0] <= y <= year_range[1]]
            if not valid_years:
                st.error(f"No data available for years {year_range[0]}-{year_range[1]}.")
                st.stop()

            # Generate charts
            for metric in metrics:
                try:
                    series = st.session_state.df.loc[valid_years, metric].dropna()
                    if series.empty or series.isna().all() or len(series) < 1 or (series.abs() < 1e-10).all():
                        st.warning(f"No valid data for '{metric}' in selected years.")
                        continue
                    
                    # Debug: Log data types and values
                    with st.expander(f"Debug: Data for {metric}"):
                        st.write(f"Index: {series.index.tolist()}")
                        st.write(f"Values: {series.values.tolist()}")
                        st.write(f"Index dtype: {series.index.dtype}")
                        st.write(f"Values dtype: {series.values.dtype}")
                    
                    # Ensure values are numeric
                    y_values = pd.to_numeric(series.values, errors="coerce")
                    if np.any(pd.isna(y_values)):
                        st.warning(f"Non-numeric data detected in '{metric}': {y_values.tolist()}")
                        continue
                    
                    # Create line chart
                    fig = px.line(
                        x=series.index,
                        y=y_values,
                        markers=True,
                        title=f"{metric} Over Time ({st.session_state.report_name})",
                        labels={"x": "Year", "y": metric}
                    )
                    
                    # Add trendline if 2+ non-NaN points
                    if len(y_values) > 1:
                        x_numeric = np.arange(len(y_values))
                        z = np.polyfit(x_numeric, y_values, 1)
                        trend = np.poly1d(z)(x_numeric)
                        fig.add_trace(go.Scatter(
                            x=series.index,
                            y=trend,
                            name="Trendline",
                            mode="lines",
                            line=dict(dash="dot", color="gray")
                        ))
                        
                        # Add deviation flags
                        pct_change = series.pct_change().fillna(0) * 100
                        for year, pct_val in pct_change.items():
                            if abs(pct_val) > 30:
                                fig.add_vline(
                                    x=year,
                                    line=dict(color="red", dash="dot"),
                                    annotation_text="⚠️ Deviation",
                                    annotation_position="top right"
                                )
                    
                    st.session_state.analysis_results["charts"].append(fig)
                
                except Exception as e:
                    st.warning(f"Failed to generate chart for {metric}: {str(e)}")
                    continue
            
            # Generate YOY table
            try:
                table = st.session_state.df.loc[valid_years, metrics].copy()
                if len(valid_years) > 1:
                    for m in metrics:
                        if m in table.columns:
                            table[f"{m} YOY (%)"] = table[m].pct_change().round(4) * 100
                st.session_state.analysis_results["table"] = table
            except Exception as e:
                st.warning(f"Failed to generate YOY table: {str(e)}")
            
            if not st.session_state.analysis_results["charts"] and st.session_state.analysis_results["table"] is None:
                st.error("No charts or table generated due to data issues.")
                st.session_state.analysis_results = None

        except Exception as e:
            st.session_state.error_message = f"Analysis failed: {str(e)}"
            st.session_state.analysis_results = None

# Display results
if st.session_state.error_message:
    st.error(st.session_state.error_message)
elif st.session_state.analysis_results and (st.session_state.analysis_results["charts"] or st.session_state.analysis_results["table"] is not None):
    st.success("✅ Analysis generated successfully!")
    
    # Display charts
    for fig in st.session_state.analysis_results["charts"]:
        try:
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying chart: {str(e)}")
    
    # Display table
    if st.session_state.analysis_results["table"] is not None:
        st.subheader(f"📊 Year-Over-Year Table: {st.session_state.report_name}")
        display_table = st.session_state.analysis_results["table"].fillna("")  # Avoid JSON issues
        st.dataframe(display_table)
        csv = st.session_state.analysis_results["table"].to_csv().encode("utf-8")
        st.download_button("📥 Download YOY Summary", csv, f"{st.session_state.report_name}_yoy_summary.csv", "text/csv")
elif st.session_state.df is not None:
    st.info("Configure analysis settings and click 'Generate Analysis' to view results.")
else:
    st.info("📎 Upload a file and click Submit to begin.")
