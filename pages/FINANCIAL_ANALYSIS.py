import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Financial Dashboard", layout="wide")
st.title("📊 Financial Analysis Dashboard")

# Upload section
st.sidebar.header("🔧 Upload File")
uploaded_file = st.sidebar.file_uploader("Upload .csv or .xls/.xlsx", type=["csv", "xls", "xlsx"])
submitted = st.sidebar.button("📤 Submit")

@st.cache_data
def load_and_process(file):
    if file.name.endswith('.csv'):
        raw = pd.read_csv(file)
    else:
        raw = pd.read_excel(file)
    
    raw = raw.dropna(how='all')  # Drop entirely blank rows
    raw = raw.set_index(raw.columns[0])  # Set first column as index
    df = raw.T  # Transpose so years become index
    df.index.name = "Year"
    df.columns = df.columns.str.strip()
    df.index = df.index.astype(str).str.extract(r'(\d{4})')[0]  # Extract 4-digit year
    df = df.dropna().astype(float)  # Clean and ensure float
    return df

if submitted and uploaded_file:
    try:
        df = load_and_process(uploaded_file)
        years = df.index.tolist()

        st.success("✅ Data loaded successfully!")

        # Metrics selector
        metrics = st.multiselect("📌 Select Metrics", df.columns.tolist(),
                                 default=["Revenue", "Net Income", "Operating Income (Loss)",
                                          "Research & Development", "Restructuring Charges"])

        # Year slider
        year_range = st.slider("📆 Select Year Range", int(years[0]), int(years[-1]), (int(years[0]), int(years[-1])))
        filtered_years = [y for y in years if int(y) >= year_range[0] and int(y) <= year_range[1]]

        for metric in metrics:
            y = df.loc[filtered_years, metric]
            fig = px.line(y, title=f"{metric} Trend", markers=True, labels={"index": "Year", metric: "Value"})
            # Trendline overlay
            z = np.polyfit(range(len(y)), y, 1)
            trend = np.poly1d(z)(range(len(y)))
            fig.add_trace(go.Scatter(x=filtered_years, y=trend, mode='lines', name='Trendline',
                                     line=dict(dash="dot", color="gray")))

            # Insight flags
            pct = y.pct_change() * 100
            for i, v in enumerate(pct):
                if abs(v) > 30:
                    fig.add_vline(x=filtered_years[i], line=dict(color="red", dash="dot"),
                                  annotation_text="⚠️ Deviation", annotation_position="top right")
            st.plotly_chart(fig, use_container_width=True)

        # YOY Summary
        st.subheader("📊 YOY Summary Table")
        yoy_table = df.loc[filtered_years, metrics].copy()
        for col in metrics:
            yoy_table[f"{col} YOY (%)"] = yoy_table[col].pct_change().round(4) * 100
        st.dataframe(yoy_table)

        # Export
        csv = yoy_table.to_csv().encode("utf-8")
        st.download_button("📥 Download YOY Summary", csv, "yoy_summary.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.info("📎 Upload your file and hit Submit to start.")
