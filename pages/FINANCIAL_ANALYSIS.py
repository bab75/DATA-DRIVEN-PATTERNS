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

        # Detect the header row
        header_row = None
        for idx, row in raw.iterrows():
            count = sum(bool(re.search(r'20\d{2}', str(cell))) for cell in row)
            if count >= 3:
                header_row = int(idx)
                break
        if header_row is None:
            raise ValueError("Could not detect year headers in file.")

        # Extract and format
        df = raw.iloc[header_row+1:].copy()
        df.columns = raw.iloc[header_row]
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
        if df.empty or df.shape[1] < 2:
            raise ValueError("DataFrame is empty or has insufficient columns after cleaning.")
        df = df.set_index(df.columns[0])
        df.columns = df.columns.astype(str).str.strip()
        df.index = df.index.astype(str).str.strip()
        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")

        df = df.T
        df.index.name = "Year"
        df.index = df.index.astype(str).map(lambda x: re.search(r"20\d{2}", x).group(0) if re.search(r"20\d{2}", x) else None)
        df = df.dropna().astype(float)

        if df.empty:
            raise ValueError("No valid data after processing.")
        return df
    except Exception as e:
        st.session_state.error_message = f"Error loading file: {str(e)}"
        return None

# Main logic
if submitted and uploaded_file:
    st.session_state.error_message = None
    df = load_financial_data(uploaded_file)
    if df is None:
        st.error(st.session_state.error_message or "❌ Failed to load file.")
        st.stop()
    
    st.session_state.df = df
    st.success("✅ File loaded successfully!")
    
    # Debug information
    st.write("**Debug Info:**")
    st.write(f"DataFrame shape: {df.shape}")
    st.write(f"DataFrame index: {df.index.tolist()}")
    st.write(f"DataFrame columns (first 5): {df.columns[:5].tolist()}")
    
    years = sorted([int(y) for y in df.index if str(y).isdigit()])
    st.write(f"Detected years: {years}")
    
    all_metrics = df.columns.tolist()
    st.write(f"Available metrics: {len(all_metrics)} total")

    preferred = ["Revenue", "Net Income", "Operating Income (Loss)", "Research & Development", "Restructuring Charges"]
    default_metrics = [m for m in preferred if m in all_metrics]

    with st.form("analysis_form"):
        st.subheader("📊 Analysis Configuration")
        metrics = st.multiselect("📌 Select metrics to analyze", all_metrics, default=st.session_state.selected_metrics or default_metrics)
        if not years:
            st.warning("⚠️ No valid year labels found.")
            st.stop()
        yr_range = st.slider("📆 Select Year Range", min(years), max(years), (min(years), max(years)), key="year_range")
        analyze_button = st.form_submit_button("🔍 Generate Analysis")

    if analyze_button and metrics:
        try:
            # Clear previous results
            st.session_state.analysis_results = None
            st.session_state.error_message = None
            st.session_state.selected_metrics = metrics
            st.session_state.selected_years = [str(y) for y in years if yr_range[0] <= y <= yr_range[1]]

            # Debug: Check what we're working with
            st.write("**Analysis Debug:**")
            st.write(f"Selected year range: {yr_range}")
            st.write(f"Selected metrics: {metrics}")
            
            # Get available years from DataFrame index
            available_years = [str(year) for year in df.index if str(year).isdigit()]
            valid_years = [year for year in st.session_state.selected_years if year in available_years]
            
            st.write(f"Available years in data: {available_years}")
            st.write(f"Requested years: {st.session_state.selected_years}")
            st.write(f"Valid years (intersection): {valid_years}")
            
            if not metrics:
                st.warning("Please select at least one metric to analyze.")
                st.stop()
            
            if not valid_years:
                st.warning(f"No valid years found. Data contains: {available_years}, requested: {st.session_state.selected_years}")
                st.stop()

            results = {"charts": [], "table": None}
            
            # Generate charts
            for metric in metrics:
                if metric not in df.columns:
                    st.warning(f"Metric '{metric}' not found in data. Available: {df.columns.tolist()}")
                    continue
                
                try:
                    series = df.loc[valid_years, metric]
                    st.write(f"Data for {metric}: {len(series)} points, values: {series.values.tolist()}")
                    
                    if series.empty or series.isna().all():
                        st.warning(f"No valid data for '{metric}' in selected years.")
                        continue
                    
                    # Remove NaN values for analysis
                    series_clean = series.dropna()
                    if len(series_clean) == 0:
                        st.warning(f"No valid data for '{metric}' after removing NaN values.")
                        continue
                    
                    # Create chart
                    st.write(f"Creating chart for {metric}...")
                    fig = px.line(x=series_clean.index, y=series_clean.values, 
                                 markers=True, title=f"{metric} Over Time",
                                 labels={"x": "Year", "y": metric})
                    
                    # Add trendline if we have enough data points
                    if len(series_clean) > 1:
                        x_numeric = range(len(series_clean))
                        z = np.polyfit(x_numeric, series_clean.values, 1)
                        trend = np.poly1d(z)(x_numeric)
                        fig.add_trace(go.Scatter(x=series_clean.index, y=trend, name="Trendline",
                                               mode="lines", line=dict(dash="dot", color="gray")))
                    
                    # Add deviation flags (handle single-year case)
                    if len(series_clean) > 1:
                        pct_change = series_clean.pct_change().fillna(0) * 100
                        for year, pct_val in pct_change.items():
                            if abs(pct_val) > 30:
                                fig.add_vline(x=year, line=dict(color="red", dash="dot"),
                                             annotation_text="⚠️ Deviation", annotation_position="top right")
                    
                    results["charts"].append(fig)
                    st.write(f"✅ Generated chart for {metric}")
                    
                except Exception as e:
                    st.error(f"Error processing {metric}: {str(e)}")
                    continue
            
            # Generate YOY table
            if metrics and valid_years:
                try:
                    table = df.loc[valid_years, metrics].copy()
                    st.write(f"Table shape: {table.shape}")
                    
                    if len(valid_years) > 1:  # Only calculate YOY if multiple years
                        for m in metrics:
                            if m in table.columns:
                                table[f"{m} YOY (%)"] = table[m].pct_change().round(4) * 100
                    else:
                        st.warning("Only one year selected, skipping YOY calculation.")
                    
                    results["table"] = table
                    st.write("✅ Generated YOY table")
                    
                except Exception as e:
                    st.error(f"Error generating table: {str(e)}")
                    results["table"] = None
            
            # Store results in session state
            st.session_state.analysis_results = results
            st.session_state.analysis_generated = True
            st.write(f"✅ Analysis complete. Charts: {len(results['charts'])}, Table: {'Yes' if results['table'] is not None else 'No'}")
            
        except Exception as e:
            st.session_state.error_message = f"Error generating analysis: {str(e)}"
            st.session_state.analysis_results = None
            st.session_state.analysis_generated = False
            st.error(f"Analysis failed: {str(e)}")

# Display results
if st.session_state.error_message:
    st.error(st.session_state.error_message)
elif st.session_state.analysis_results is not None:
    st.success("✅ Analysis generated successfully!")
    
    # Display charts
    if "charts" in st.session_state.analysis_results and st.session_state.analysis_results["charts"]:
        for fig in st.session_state.analysis_results["charts"]:
            try:
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying chart: {str(e)}")
    
    # Display table
    if "table" in st.session_state.analysis_results and st.session_state.analysis_results["table"] is not None:
        st.subheader("📊 YOY Change Table")
        st.dataframe(st.session_state.analysis_results["table"])
        
        csv = st.session_state.analysis_results["table"].to_csv().encode("utf-8")
        st.download_button("📥 Download YOY Summary", csv, "yoy_summary.csv", "text/csv")
elif st.session_state.df is not None:
    st.info("Configure analysis settings and click 'Generate Analysis' to view results.")
else:
    st.info("📎 Upload a file and click Submit to begin.")
