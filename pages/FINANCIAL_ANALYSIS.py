import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Financial Dashboard", layout="wide")

st.title("📊 Financial Analysis Dashboard")
st.markdown("Upload a `.csv` or `.xls` file with financial data (e.g., Revenue, Net Income, R&D, etc.) to explore trends and generate insights.")

# Sidebar Upload
st.sidebar.header("🔧 Upload Your File")
uploaded_file = st.sidebar.file_uploader("Select .csv or .xls/.xlsx file", type=["csv", "xls", "xlsx"])
submitted = st.sidebar.button("📤 Submit")

@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file, index_col=0)
    else:
        return pd.read_excel(file, index_col=0)

if submitted and uploaded_file:
    try:
        df = load_data(uploaded_file)
        df.columns = df.columns.map(str).str.strip()
        df.index = df.index.map(str).str.strip()
        
        # Extract and validate year columns
        year_cols = [col for col in df.columns if col.isdigit() and col.startswith('20')]
        if not year_cols:
            st.error("❌ No valid year columns found. Ensure your data has columns labeled like '2020', '2021', etc.")
        else:
            years = sorted([int(y) for y in year_cols])
            selected_metrics = st.multiselect(
                "📌 Choose metrics to analyze",
                df.index.tolist(),
                default=["Revenue", "Net Income", "Operating Income (Loss)", 
                         "Research & Development", "Restructuring Charges"]
            )

            year_range = st.slider("📆 Select Year Range", min_value=years[0], max_value=years[-1], value=(years[0], years[-1]))
            filtered_years = [str(y) for y in years if year_range[0] <= y <= year_range[1]]

            # Plot for each metric
            for metric in selected_metrics:
                if metric in df.index:
                    values = df.loc[metric, filtered_years].astype(float).values
                    base_df = pd.DataFrame({'Year': filtered_years, metric: values})

                    fig = px.line(base_df, x="Year", y=metric, markers=True, title=f"{metric} Trend")

                    # Trendline
                    trend = np.poly1d(np.polyfit(range(len(values)), values, 1))
                    fig.add_trace(go.Scatter(
                        x=filtered_years,
                        y=trend(range(len(values))),
                        mode="lines",
                        name="Trendline",
                        line=dict(color="gray", dash="dot")
                    ))

                    # Insight Flags
                    pct_change = pd.Series(values).pct_change().fillna(0) * 100
                    flagged_years = [filtered_years[i] for i, pct in enumerate(pct_change) if abs(pct) > 30]
                    for fy in flagged_years:
                        fig.add_vline(x=fy, line=dict(color="red", dash="dot"),
                                      annotation_text="⚠️ Deviation", annotation_position="top right")

                    st.plotly_chart(fig, use_container_width=True)

            # YOY Summary Table
            st.subheader("📊 YOY % Change Summary")
            yoy_df = pd.DataFrame({'Year': filtered_years})
            for metric in selected_metrics:
                if metric in df.index:
                    values = df.loc[metric, filtered_years].astype(float).values
                    yoy = pd.Series(values).pct_change().fillna(0) * 100
                    yoy_df[metric] = values
                    yoy_df[f"{metric} YOY (%)"] = yoy.round(2)
            st.dataframe(yoy_df.set_index("Year"), use_container_width=True)

            # Export
            csv = yoy_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download YOY Summary", csv, "yoy_summary.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error while processing your file: {e}")

else:
    st.info("📎 Upload your file and click Submit to start.")
