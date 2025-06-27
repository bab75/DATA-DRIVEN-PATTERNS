import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re

st.set_page_config(page_title="Financial Dashboard", layout="wide")
st.title("📊 Financial Analysis Dashboard")

# Sidebar Upload
st.sidebar.header("🔧 Upload File")
uploaded_file = st.sidebar.file_uploader("Upload .csv or .xls/.xlsx", type=["csv", "xls", "xlsx"])
submitted = st.sidebar.button("📤 Submit")

@st.cache_data
def load_financial_data(file):
    if file.name.endswith('.csv'):
        raw = pd.read_csv(file, header=None)
    else:
        raw = pd.read_excel(file, header=None)

    # Detect header row with years like 2018, 2019-12-31, FY 2020
    for idx, row in raw.iterrows():
        count_years = sum(bool(re.search(r'20\d{2}', str(cell))) for cell in row)
        if count_years >= 3:
            header_idx = idx
            break
    else:
        raise ValueError("No valid row with year headers found.")

    df = raw.iloc[header_idx + 1:].copy()
    df.columns = raw.iloc[header_idx]
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    df = df.set_index(df.columns[0])
    df.columns = df.columns.astype(str).str.strip()
    df.index = df.index.astype(str).str.strip()
    df = df.apply(pd.to_numeric, errors='coerce').dropna(how='all')

    df = df.T  # Transpose so years = index
    df.index.name = "Year"

    # Extract only valid 4-digit years from things like 'FY 2022' or '2020-12-31'
    df.index = df.index.astype(str).map(lambda x: re.search(r'20\d{2}', x).group(0) if re.search(r'20\d{2}', x) else None)
    df = df.dropna().astype(float)

    return df

if submitted and uploaded_file:
    try:
        df = load_financial_data(uploaded_file)
        st.success("✅ File processed successfully!")

        # Get available years & safe default metrics
        years = sorted([int(y) for y in df.index if y.isdigit()])
        year_labels = [str(y) for y in years]
        all_metrics = df.columns.tolist()

        preferred = [
            "Revenue", "Net Income", "Operating Income (Loss)",
            "Research & Development", "Restructuring Charges"
        ]
        default_metrics = [m for m in preferred if m in all_metrics]

        # Use session state to avoid dropdown reset
        if "selected_metrics" not in st.session_state:
            st.session_state.selected_metrics = default_metrics

        metrics = st.multiselect("📌 Select metrics to analyze", all_metrics, default=st.session_state.selected_metrics)
        st.session_state.selected_metrics = metrics

        # Year slider
        if years:
            yr_min, yr_max = min(years), max(years)
            yr_range = st.slider("📆 Select Year Range", yr_min, yr_max, (yr_min, yr_max))
            selected_years = [str(y) for y in years if yr_range[0] <= y <= yr_range[1]]
        else:
            st.error("No valid numeric years found.")
            st.stop()

        # Generate charts
        for metric in metrics:
            if metric not in df.columns:
                continue
            series = df.loc[selected_years, metric]
            fig = px.line(series, markers=True, title=f"{metric} Over Time",
                          labels={"index": "Year", "value": metric})

            # Trendline
            z = np.polyfit(range(len(series)), series.values, 1)
            trend = np.poly1d(z)(range(len(series)))
            fig.add_trace(go.Scatter(x=selected_years, y=trend, mode="lines", name="Trendline",
                                     line=dict(dash="dot", color="gray")))

            # Spike alerts
            pct = series.pct_change().fillna(0) * 100
            for i, change in enumerate(pct):
                if abs(change) > 30:
                    fig.add_vline(x=selected_years[i], line=dict(color="red", dash="dot"),
                                  annotation_text="⚠️ Deviation", annotation_position="top right")

            st.plotly_chart(fig, use_container_width=True)

        # YOY table
        st.subheader("📊 YOY Change Table")
        result = df.loc[selected_years, metrics].copy()
        for col in metrics:
            result[f"{col} YOY (%)"] = result[col].pct_change().round(4) * 100
        st.dataframe(result)

        # Export download
        csv = result.to_csv().encode("utf-8")
        st.download_button("📥 Download YOY Summary", csv, "yoy_summary.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("📎 Upload your file and click Submit to begin.")
