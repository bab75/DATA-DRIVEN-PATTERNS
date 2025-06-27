import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re

st.set_page_config(page_title="Financial Dashboard", layout="wide")
st.title("📊 Financial Analysis Dashboard")

# Upload section
st.sidebar.header("🔧 Upload File")
uploaded_file = st.sidebar.file_uploader("Upload .csv or .xls/.xlsx", type=["csv", "xls", "xlsx"])
submitted = st.sidebar.button("📤 Submit")

@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        raw = pd.read_csv(file, header=None)
    else:
        raw = pd.read_excel(file, header=None)
    
    # Find first row that contains actual years
    for idx, row in raw.iterrows():
        potential_years = [str(x) for x in row if re.match(r'20\d{2}', str(x))]
        if len(potential_years) >= 3:
            data_start_idx = idx
            break
    else:
        raise ValueError("Could not find a row with year headers.")

    df = raw.iloc[data_start_idx+1:, :]  # Data below header
    df.columns = raw.iloc[data_start_idx]
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')  # Clean
    df = df.set_index(df.columns[0])  # First col = metric name
    df.columns = df.columns.astype(str)
    df.index = df.index.astype(str).str.strip()
    df = df.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    df = df.T  # Transpose: years = index
    df.index.name = 'Year'
    return df

if submitted and uploaded_file:
    try:
        df = load_data(uploaded_file)
        years = df.index.dropna().astype(str).tolist()
        st.success("✅ File processed successfully!")

        # Metric selector
        metrics = st.multiselect("📌 Choose metrics to analyze",
                                 df.columns.tolist(),
                                 default=["Revenue", "Net Income", "Operating Income (Loss)", 
                                          "Research & Development", "Restructuring Charges"])

        # Year range
        year_ints = sorted([int(y) for y in years if y.isdigit()])
        if not year_ints:
            st.error("Could not extract valid year values. Please verify the data structure.")
        else:
            yr_range = st.slider("📆 Select Year Range", min_value=year_ints[0], max_value=year_ints[-1],
                                 value=(year_ints[0], year_ints[-1]))
            selected_years = [str(y) for y in year_ints if yr_range[0] <= y <= yr_range[1]]

            for metric in metrics:
                y = df.loc[selected_years, metric]
                fig = px.line(y, title=f"{metric} Trend", markers=True, labels={"index": "Year", metric: "Value"})

                # Trendline
                z = np.polyfit(range(len(y)), y, 1)
                trend = np.poly1d(z)(range(len(y)))
                fig.add_trace(go.Scatter(x=selected_years, y=trend, mode='lines', name='Trendline',
                                         line=dict(dash="dot", color="gray")))

                # Spike flag
                pct = y.pct_change().fillna(0) * 100
                for i, v in enumerate(pct):
                    if abs(v) > 30:
                        fig.add_vline(x=selected_years[i], line=dict(color="red", dash="dot"),
                                      annotation_text="⚠️ Deviation", annotation_position="top right")

                st.plotly_chart(fig, use_container_width=True)

            # YOY summary
            st.subheader("📊 YOY Change Table")
            table = df.loc[selected_years, metrics].copy()
            for col in metrics:
                table[f"{col} YOY (%)"] = table[col].pct_change().round(4) * 100
            st.dataframe(table)

            # Export
            csv = table.to_csv().encode("utf-8")
            st.download_button("📥 Download YOY Summary", csv, "yoy_summary.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Failed to load and process file: {e}")
else:
    st.info("📎 Upload a file and hit Submit to begin.")
