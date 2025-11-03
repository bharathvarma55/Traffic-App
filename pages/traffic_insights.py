# pages/1_Traffic_Insights.py
import os
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Traffic Insights", page_icon="📊", layout="wide")
st.title("📊 Traffic Insights & Patterns")
st.markdown("This page shows hourly patterns and historical summaries (simulated if logs are missing).")

# Attempt to read logs.csv
if os.path.exists("logs.csv"):
    try:
        df_logs = pd.read_csv("logs.csv", parse_dates=["Timestamp"])
        df_logs["Hour"] = pd.to_datetime(df_logs["Timestamp"]).dt.hour
        hourly = df_logs.groupby("Hour")["Predicted Volume"].mean().reset_index()
        source = "logs.csv"
    except Exception:
        df_logs = None
        source = "simulated"
else:
    df_logs = None
    source = "simulated"

if source == "simulated":
    hours = np.arange(0, 24)
    base = 60 + 60 * (np.sin((hours - 7) / 3) + np.sin((hours - 17) / 3))
    noise = np.random.randint(-10, 10, size=24)
    hourly = pd.DataFrame({"Hour": hours, "Predicted Volume": np.maximum(20, base + noise)})

st.markdown(f"Data source: **{source}**")
fig = px.line(hourly, x="Hour", y="Predicted Volume", markers=True, title="Hourly Average Predicted Traffic")
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
st.plotly_chart(fig, use_container_width=True)

st.markdown("**Tip:** Use this page to decide thresholds for alerts and to understand daily peaks.")
