import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re

# Set page config
st.set_page_config(page_title="Financial Dashboard", layout="wide")
st.title("📊 Financial Analysis Dashboard")

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "metrics" not in st.session_state:
    st.session_state.metrics = []
if "years" not in st.session_state:
    st.session_state.years = []

# Function to load and process financial data
@st.cache_data
def load_financial_data(file):
    try:
        # Read file
        if file.name.endswith(".csv"):
            raw = pd.read_csv(file, header=None)
        else:
            raw = pd.read_excel(file, header=None)

        # Detect header row with years (look for FY 20XX or 20XX-12-31)
        header_row = None
        for idx, row in raw.iterrows():
            year_count = sum(bool(re.search(r'(FY\s*20\d{2}|20\d{2}-\d{2}-\d{2})', str(cell))) for cell in row)
            if year_count >= 3:
                header_row = idx
                break
        if header_row is None:
            return None, "No year headers (e.g., FY 20XX or 20XX-12-31) found in file."

        # Set up DataFrame
        df = raw.iloc[header_row+1:].copy()
        df.columns = raw.iloc[header_row]
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if df.empty or df.shape[1] < 2:
            return None, "DataFrame is empty or has insufficient columns."
        
        # Set first column as index (metrics)
        df = df.set_index(df.columns[0])
        df.columns = df.columns.astype(str).str.strip()
        df.index = df.index.astype(str).str.strip()
        
        # Convert to numeric and transpose
        df = df.apply(pd.to_numeric, errors="coerce")
        # Drop columns (metrics) with all NaN or all zero values
        df = df.dropna(axis=1, how="all")
        df = df.loc[:, (df != 0).any(axis=0)]  # Drop columns with all zeros
        df = df.T
        df.index.name = "Year"
        # Extract year from FY 20XX or 20XX-12-31
        df.index = [re.search(r'20\d{2}', str(x)).group(0) if re.search(r'20\d{2}', str(x)) else None for x in df.index]
        # Drop rows with None index (invalid years)
        df = df[df.index.notnull()]
        df = df.astype(float)
        # Drop rows (years) with all NaN or all zero values
        df = df.dropna(axis=0, how="all")
        df = df.loc[(df != 0).any(axis=1)]  # Drop rows with all zeros

        if df.empty:
            return None, "No valid data after processing."
        
        return df, None
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

# Sidebar for file upload
st.sidebar.header("🔧 Upload File")
uploaded_file = st.sidebar.file_uploader("Upload a .csv or .xls/.xlsx file", type=["csv", "xls", "xlsx"])
if st.sidebar.button("📤 Submit") and uploaded_file:
    st.session_state.df, st.session_state.error_message = load_financial_data(uploaded_file)
    if st.session_state.df is None:
        st.error(st.session_state.error_message or "❌ Failed to load file.")
    else:
        st.success("✅ File loaded successfully!")
        # Store available metrics and years
        st.session_state.metrics = [col for col in st.session_state.df.columns if st.session_state.df[col].notna().any()]
        st.session_state.years = sorted([int(y) for y in st.session_state.df.index if str(y).isdigit()])
        # Minimal debug output
        with st.expander("Debug Info"):
            st.write(f"DataFrame shape: {st.session_state.df.shape}")
            st.write(f"Years: {st.session_state.years}")
            st.write(f"Metrics (first 5): {st.session_state.metrics[:5]}")

# Display filtered DataFrame
if st.session_state.df is not None:
    st.subheader("📋 Loaded Financial Data")
    # Filter DataFrame to include only metrics with some non-null data
    valid_metrics = [col for col in st.session_state.df.columns if st.session_state.df[col].notna().any()]
    filtered_df = st.session_state.df[valid_metrics]
    # Filter years to include only those with some non-null, non-zero data
    filtered_df = filtered_df.loc[(filtered_df.notna().any(axis=1)) & (filtered_df != 0).any(axis=1)]
    if not filtered_df.empty:
        st.dataframe(filtered_df)
        # Add download button for filtered DataFrame
        csv = filtered_df.to_csv().encode("utf-8")
        st.download_button("📥 Download Filtered Data", csv, "filtered_financial_data.csv", "text/csv")
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
                    if series.empty or series.isna().all() or len(series) < 1 or (series == 0).all():
                        st.warning(f"No valid data for '{metric}' in selected years.")
                        continue
                    
                    # Create line chart
                    fig = px.line(
                        x=series.index,
                        y=series.values,
                        markers=True,
                        title=f"{metric} Over Time",
                        labels={"x": "Year", "y": metric}
                    )
                    
                    # Add trendline if 2+ points
                    if len(series) > 1:
                        x_numeric = range(len(series))
                        z = np.polyfit(x_numeric, series.values, 1)
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
        st.subheader("📊 Year-Over-Year Table")
        st.dataframe(st.session_state.analysis_results["table"])
        csv = st.session_state.analysis_results["table"].to_csv().encode("utf-8")
        st.download_button("📥 Download YOY Summary", csv, "yoy_summary.csv", "text/csv")
elif st.session_state.df is not None:
    st.info("Configure analysis settings and click 'Generate Analysis' to view results.")
else:
    st.info("📎 Upload a file and click Submit to begin.")
