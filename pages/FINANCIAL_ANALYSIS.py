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
    st.session_state.analysis_results = {"charts": [], "table": None}
if "selected_metrics" not in st.session_state:
    st.session_state.selected_metrics = []
if "selected_years" not in st.session_state:
    st.session_state.selected_years = []
if "error_message" not in st.session_state:
    st.session_state.error_message = None

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

        header_row = None
        for idx, row in raw.iterrows():
            count = sum(bool(re.search(r'20\d{2}', str(cell))) for cell in row)
            if count >= 3:
                header_row = int(idx)
                break
        if header_row is None:
            raise ValueError("Could not detect year headers in file.")

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
        df.index = df.index.astype(str).map(lambda x: re.search(r'20\d{2}', x).group(0) if re.search(r'20\d{2}', x) else None)
        df = df.dropna().astype(float)

        if df.empty:
            raise ValueError("No valid data after processing.")
        return df
    except Exception as e:
        st.session_state.error_message = f"Error loading file: {str(e)}"
        return None

if submitted and uploaded_file:
    st.session_state.error_message = None
    df = load_financial_data(uploaded_file)
    if df is None:
        st.error(st.session_state.error_message or "❌ Failed to load file.")
        st.stop()
    
    st.session_state.df = df
    st.success("✅ File loaded successfully!")

    years = sorted([int(y) for y in df.index if str(y).isdigit()])
    all_metrics = df.columns.tolist()
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
            st.session_state.error_message = None
            selected_years = [str(y) for y in years if yr_range[0] <= y <= yr_range[1]]

            if not selected_years:
                st.warning("No valid years in selected range.")
                st.stop()

            results = {"charts": [], "table": None}
            
            for metric in metrics:
                if metric not in df.columns:
                    st.warning(f"Metric '{metric}' not found in data, skipping.")
                    continue
                series = df.loc[selected_years, metric].dropna()
                if len(series) < 1:
                    st.warning(f"No valid data for '{metric}' in selected years.")
                    continue
                
                fig = px.line(series, markers=True, title=f"{metric} Over Time", labels={"index": "Year", "value": metric})
                if len(series) > 1:
                    z = np.polyfit(range(len(series)), series.values, 1)
                    trend = np.poly1d(z)(range(len(series)))
                    fig.add_trace(go.Scatter(x=series.index, y=trend, name="Trendline", mode="lines", line=dict(dash="dot", color="gray")))
                
                pct_change = series.pct_change().fillna(0) * 100
                for year, pct_val in pct_change.items():
                    if abs(pct_val) > 30:
                        fig.add_vline(x=year, line=dict(color="red", dash="dot"), annotation_text="⚠️ Deviation", annotation_position="top right")
                
                results["charts"].append(fig)

            if metrics and selected_years:
                table = df.loc[selected_years, metrics].copy()
                for m in metrics:
                    if m in table.columns:
                        table[f"{m} YOY (%)"] = table[m].pct_change().round(4) * 100
                results["table"] = table

            st.session_state.analysis_results = results
            st.session_state.selected_metrics = metrics
            st.session_state.selected_years = selected_years

        except Exception as e:
            st.session_state.error_message = f"Error generating analysis: {str(e)}"
            st.session_state.analysis_results = {"charts": [], "table": None}

# Display results
if st.session_state.error_message:
    st.error(st.session_state.error_message)
elif st.session_state.analysis_results is not None and (st.session_state.analysis_results["charts"] or st.session_state.analysis_results["table"]):
    st.success("✅ Analysis generated successfully!")
    if st.session_state.analysis_results["charts"]:
        for fig in st.session_state.analysis_results["charts"]:
            st.plotly_chart(fig, use_container_width=True)
    if st.session_state.analysis_results["table"] is not None:
        st.subheader("📊 YOY Change Table")
        st.dataframe(st.session_state.analysis_results["table"])
        csv = st.session_state.analysis_results["table"].to_csv().encode("utf-8")
        st.download_button("📥 Download YOY Summary", csv, "yoy_summary.csv", "text/csv", key="download_button")
elif st.session_state.df is not None:
    st.info("Configure analysis settings and click 'Generate Analysis' to view results.")
else:
    st.info("📎 Upload a file and click Submit to begin.")
