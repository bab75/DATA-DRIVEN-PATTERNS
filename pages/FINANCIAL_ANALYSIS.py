import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re

st.set_page_config(page_title="Financial Dashboard", layout="wide")
st.title("📊 Financial Analysis Dashboard")

# Sidebar File Upload
st.sidebar.header("🔧 Upload File")
uploaded_file = st.sidebar.file_uploader("Upload a .csv or .xls/.xlsx file", type=["csv", "xls", "xlsx"])
submitted = st.sidebar.button("📤 Submit")

@st.cache_data
def load_financial_data(file):
    if file.name.endswith('.csv'):
        df_raw = pd.read_csv(file, header=None)
    else:
        df_raw = pd.read_excel(file, header=None)

    # Detect where year headers begin
    for idx, row in df_raw.iterrows():
        potential_years = [str(x) for x in row if re.match(r'20\d{2}', str(x))]
        if len(potential_years) >= 3:
            header_row = idx
            break
    else:
        raise ValueError("No row with enough year headers found.")

    df_data = df_raw.iloc[header_row+1:, :].copy()
    df_data.columns = df_raw.iloc[header_row]
    df_data = df_data.dropna(axis=1, how='all').dropna(axis=0, how='all')
    df_data = df_data.set_index(df_data.columns[0])
    df_data.columns = df_data.columns.astype(str).str.strip()
    df_data.index = df_data.index.astype(str).str.strip()
    df_data = df_data.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    df_data = df_data.T  # Transpose so years = index
    df_data.index.name = 'Year'
    return df_data

if submitted and uploaded_file:
    try:
        df = load_financial_data(uploaded_file)
        years = df.index.dropna().astype(str).tolist()
        st.success("✅ File loaded and processed!")

        # Prepare metric selector
        all_metrics = df.columns.tolist()
        preferred_metrics = [
            "Revenue", "Net Income", "Operating Income (Loss)",
            "Research & Development", "Restructuring Charges"
        ]
        default_metrics = [m for m in preferred_metrics if m in all_metrics]
        metrics = st.multiselect("📌 Select metrics to analyze", all_metrics, default=default_metrics)

        # Year range slider
        year_ints = sorted([int(y) for y in years if y.isdigit()])
        if not year_ints:
            st.warning("⚠️ No numeric year values found.")
        else:
            year_range = st.slider("📆 Select Year Range", min_value=year_ints[0], max_value=year_ints[-1],
                                   value=(year_ints[0], year_ints[-1]))
            selected_years = [str(y) for y in year_ints if year_range[0] <= y <= year_range[1]]

            # Plot each metric
            for metric in metrics:
                y = df.loc[selected_years, metric]
                fig = px.line(y, title=f"{metric} Over Time", markers=True,
                              labels={"index": "Year", metric: "Value"})
                
                # Add trendline
                z = np.polyfit(range(len(y)), y, 1)
                trendline = np.poly1d(z)(range(len(y)))
                fig.add_trace(go.Scatter(x=selected_years, y=trendline, mode="lines", name="Trendline",
                                         line=dict(color="gray", dash="dot")))

                # Auto-insight flags
                pct = y.pct_change().fillna(0) * 100
                for i, change in enumerate(pct):
                    if abs(change) > 30:
                        fig.add_vline(x=selected_years[i], line=dict(color="red", dash="dot"),
                                      annotation_text="⚠️ Deviation", annotation_position="top right")

                st.plotly_chart(fig, use_container_width=True)

            # YOY Table
            st.subheader("📊 YOY Change Table")
            table = df.loc[selected_years, metrics].copy()
            for col in metrics:
                table[f"{col} YOY (%)"] = table[col].pct_change().round(4) * 100
            st.dataframe(table)

            # CSV Export
            csv = table.to_csv().encode("utf-8")
            st.download_button("📥 Download YOY Summary", csv, "yoy_summary.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Failed to process file: {e}")
else:
    st.info("📎 Please upload a file and click Submit to start.")
