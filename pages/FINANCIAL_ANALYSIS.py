import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re

st.set_page_config(page_title="Financial Dashboard", layout="wide")
st.title("📊 Financial Analysis Dashboard")

# Initialize session state for storing analysis results
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "selected_metrics" not in st.session_state:
    st.session_state.selected_metrics = []
if "selected_years" not in st.session_state:
    st.session_state.selected_years = []

# Sidebar upload
st.sidebar.header("🔧 Upload File")
uploaded_file = st.sidebar.file_uploader("Upload a .csv or .xls/.xlsx file", type=["csv", "xls", "xlsx"])
submitted = st.sidebar.button("📤 Submit")

@st.cache_data
def load_financial_data(file):
    try:
        # Load as raw DataFrame
        if file.name.endswith(".csv"):
            raw = pd.read_csv(file, header=None)
        else:
            raw = pd.read_excel(file, header=None)

        # Detect the header row (the one with year-like labels)
        header_row = None
        for idx, row in raw.iterrows():
            count = sum(bool(re.search(r'20\d{2}', str(cell))) for cell in row)
            if count >= 3:
                header_row = int(idx)  # Ensure it's an integer
                break
        if header_row is None:
            raise ValueError("❌ Could not detect year headers in file.")

        # Extract and format
        df = raw.iloc[header_row+1:].copy()
        df.columns = raw.iloc[header_row]
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
        df = df.set_index(df.columns[0])
        df.columns = df.columns.astype(str).str.strip()
        df.index = df.index.astype(str).str.strip()
        df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")

        df = df.T  # Transpose: years as index
        df.index.name = "Year"
        df.index = df.index.astype(str).map(lambda x: re.search(r"20\d{2}", x).group(0) if re.search(r"20\d{2}", x) else None)
        df = df.dropna().astype(float)

        return df
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None

if submitted and uploaded_file:
    try:
        df = load_financial_data(uploaded_file)
        if df is None:
            st.stop()

        st.success("✅ File loaded successfully!")
        years = sorted([int(y) for y in df.index if str(y).isdigit()])
        year_labels = [str(y) for y in years]
        all_metrics = df.columns.tolist()

        # Preferred metrics if available
        preferred = ["Revenue", "Net Income", "Operating Income (Loss)", "Research & Development", "Restructuring Charges"]
        default_metrics = [m for m in preferred if m in all_metrics]

        # Create a form for analysis controls
        with st.form("analysis_form"):
            st.subheader("📊 Analysis Configuration")
            metrics = st.multiselect("📌 Select metrics to analyze", all_metrics, default=default_metrics)
            if years:
                yr_range = st.slider("📆 Select Year Range", min(years), max(years), (min(years), max(years)))
            else:
                st.warning("⚠️ No valid year labels found.")
                st.stop()
            analyze_button = st.form_submit_button("🔍 Generate Analysis")

        # Process analysis when button is clicked
        if analyze_button:
            try:
                st.session_state.selected_metrics = metrics
                st.session_state.selected_years = [str(y) for y in years if yr_range[0] <= y <= yr_range[1]]
                
                # Store analysis results in session state
                results = {"charts": [], "table": None}
                for metric in st.session_state.selected_metrics:
                    if metric not in df.columns:
                        continue
                    series = df.loc[st.session_state.selected_years, metric]
                    fig = px.line(series, markers=True, title=f"{metric} Over Time", labels={"index": "Year", "value": metric})
                    
                    # Trendline
                    z = np.polyfit(range(len(series)), series.values, 1)
                    trend = np.poly1d(z)(range(len(series)))
                    fig.add_trace(go.Scatter(x=st.session_state.selected_years, y=trend, name="Trendline",
                                            mode="lines", line=dict(dash="dot", color="gray")))
                    
                    # Deviation flags
                    pct = series.pct_change().fillna(0) * 100
                    for i, v in enumerate(pct):
                        if abs(v) > 30:
                            fig.add_vline(x=st.session_state.selected_years[i], line=dict(color="red", dash="dot"),
                                        annotation_text="⚠️ Deviation", annotation_position="top right")
                    results["charts"].append(fig)

                # YOY Table
                table = df.loc[st.session_state.selected_years, st.session_state.selected_metrics].copy()
                for m in st.session_state.selected_metrics:
                    table[f"{m} YOY (%)"] = table[m].pct_change().round(4) * 100
                results["table"] = table

                st.session_state.analysis_results = results
            except Exception as e:
                st.error(f"❌ Error generating analysis: {e}")

        # Display stored results if available
        if st.session_state.analysis_results:
            for fig in st.session_state.analysis_results["charts"]:
                st.plotly_chart(fig, use_container_width=True)
            
            if st.session_state.analysis_results["table"] is not None:
                st.subheader("📊 YOY Change Table")
                st.dataframe(st.session_state.analysis_results["table"])
                
                # Export
                csv = st.session_state.analysis_results["table"].to_csv().encode("utf-8")
                st.download_button("📥 Download YOY Summary", csv, "yoy_summary.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("📎 Upload a file and click Submit to begin.")
