import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="wide")
st.title("📊 Financial Analysis Dashboard – 3M & Beyond")
st.markdown("Upload a `.csv` or `.xls` file containing P&L data. Analyze trends, drill into R&D and restructuring, and export YOY summaries.")

# Sidebar: Upload + Submit
st.sidebar.header("🔧 Upload & Configure")
uploaded_file = st.sidebar.file_uploader("Choose .csv or .xls/.xlsx file", type=["csv", "xls", "xlsx"])
submitted = st.sidebar.button("Submit")

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file, index_col=0)
    return pd.read_excel(file, index_col=0)

if submitted and uploaded_file:
    df = load_data(uploaded_file)
    df.columns = df.columns.map(str)
    years = [col for col in df.columns if col.startswith("20")]

    st.success("✅ File uploaded and processed!")

    # --- Metric Selector & Year Range
    all_metrics = df.index.tolist()
    metrics = st.multiselect("📌 Select Metrics to Analyze", all_metrics,
                             default=["Revenue", "Net Income", "Operating Income (Loss)", 
                                      "Research & Development", "Restructuring Charges"])
    
    year_range = st.slider("📆 Select Year Range", min_value=int(years[0]), 
                           max_value=int(years[-1]), 
                           value=(int(years[0]), int(years[-1])))

    filter_years = [y for y in years if int(y) >= year_range[0] and int(y) <= year_range[1]]
    
    for metric in metrics:
        if metric in df.index:
            vals = df.loc[metric, filter_years].astype(float).values
            plot_df = pd.DataFrame({"Year": filter_years, metric: vals})
            
            # Add Trendline (optional)
            fig = px.line(plot_df, x="Year", y=metric, markers=True, title=f"{metric} Trend")
            z = np.polyfit(range(len(vals)), vals, 1)
            trendline = np.poly1d(z)(range(len(vals)))
            fig.add_trace(go.Scatter(x=filter_years, y=trendline, mode="lines",
                                     name="Trendline", line=dict(dash="dash", color="gray")))
            
            # Auto-Insight Flag
            change_pct = pd.Series(vals).pct_change().fillna(0) * 100
            flagged = [filter_years[i] for i, pct in enumerate(change_pct) if abs(pct) > 30]
            if flagged:
                for f in flagged:
                    fig.add_vline(x=f, line=dict(color="red", width=1, dash="dot"),
                                  annotation_text="⚠️ Spike", annotation_position="top right")

            st.plotly_chart(fig, use_container_width=True)

    # --- YOY Calculation Table
    st.subheader("📊 Year-over-Year (%) Change Summary")
    yoy_df = pd.DataFrame({"Year": filter_years})
    for metric in metrics:
        if metric in df.index:
            vals = df.loc[metric, filter_years].astype(float).values
            yoy_df[metric] = vals
            yoy_df[f"{metric} YOY (%)"] = pd.Series(vals).pct_change().fillna(0).round(3) * 100

    st.dataframe(yoy_df.set_index("Year"), use_container_width=True)

    # --- Export to CSV
    csv = yoy_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download YOY Data", csv, "YOY_Summary.csv", "text/csv")

else:
    st.info("📎 Upload a file and click Submit to begin analysis.")
