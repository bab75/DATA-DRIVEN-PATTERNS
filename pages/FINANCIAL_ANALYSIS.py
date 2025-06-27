import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re

st.set_page_config(page_title="Financial Dashboard", layout="wide")
st.title("📊 Financial Analysis Dashboard")

# Sidebar file upload
st.sidebar.header("🔧 Upload File")
uploaded_file = st.sidebar.file_uploader("Upload a .csv or .xls/.xlsx file", type=["csv", "xls", "xlsx"])
submitted = st.sidebar.button("📤 Submit")

@st.cache_data
def load_financial_data(file):
    if file.name.endswith('.csv'):
        raw = pd.read_csv(file, header=None)
    else:
        raw = pd.read_excel(file, header=None)

    # Auto-detect row with year headers
    for idx, row in raw.iterrows():
        year_row = [str(x) for x in row if re.match(r'^20\d{2}$', str(x).strip())]
        if len(year_row) >= 3:
            header_idx = idx
            break
    else:
        raise ValueError("No valid year header row found.")

    df = raw.iloc[header_idx + 1:].copy()
    df.columns = raw.iloc[header_idx]
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    df = df.set_index(df.columns[0])
    df.columns = df.columns.astype(str).str.strip()
    df.index = df.index.astype(str).str.strip()
    df = df.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    df = df.T  # Transpose so years become index
    df.index.name = "Year"
    df.index = df.index.astype(str).str.extract(r'(^20\d{2})')[0]
    df = df.dropna().astype(float)
    return df

# App logic
if submitted and uploaded_file:
    try:
        df = load_financial_data(uploaded_file)
        st.success("✅ File loaded successfully!")

        # Year list
        year_ints = sorted([int(y) for y in df.index if y.isdigit()])
        selected_years = [str(y) for y in year_ints]

        # Default metrics fallback
        all_metrics = df.columns.tolist()
        preferred = [
            "Revenue", "Net Income", "Operating Income (Loss)",
            "Research & Development", "Restructuring Charges"
        ]
        default_metrics = [m for m in preferred if m in all_metrics]

        if "selected_metrics" not in st.session_state:
            st.session_state.selected_metrics = default_metrics

        metrics = st.multiselect("📌 Select metrics to analyze", all_metrics, default=st.session_state.selected_metrics)
        st.session_state.selected_metrics = metrics

        # Year range slider
        if year_ints:
            yr_range = st.slider("📆 Select Year Range", min_value=year_ints[0], max_value=year_ints[-1],
                                 value=(year_ints[0], year_ints[-1]))
            filtered_years = [str(y) for y in year_ints if yr_range[0] <= y <= yr_range[1]]
        else:
            st.error("Could not extract valid years.")
            st.stop()

        # Charts and insights
        for metric in metrics:
            if metric not in df.columns:
                continue
            series = df.loc[filtered_years, metric]
            fig = px.line(series, markers=True, title=f"{metric} Over Time",
                          labels={"index": "Year", "value": metric})

            # Add trendline
            z = np.polyfit(range(len(series)), series.values, 1)
            trend = np.poly1d(z)(range(len(series)))
            fig.add_trace(go.Scatter(x=filtered_years, y=trend, mode="lines", name="Trendline",
                                     line=dict(dash="dot", color="gray")))

            # Auto-flag spikes
            pct = series.pct_change().fillna(0) * 100
            for i, v in enumerate(pct):
                if abs(v) > 30:
                    fig.add_vline(x=filtered_years[i], line=dict(color="red", dash="dot"),
                                  annotation_text="⚠️ Deviation", annotation_position="top right")

            st.plotly_chart(fig, use_container_width=True)

        # YOY table
        st.subheader("📊 YOY Change Table")
        summary = df.loc[filtered_years, metrics].copy()
        for m in metrics:
            summary[f"{m} YOY (%)"] = summary[m].pct_change().round(4) * 100
        st.dataframe(summary)

        # Export button
        csv = summary.to_csv().encode("utf-8")
        st.download_button("📥 Download YOY Summary", csv, "yoy_summary.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("📎 Upload a file and click Submit to begin.")
