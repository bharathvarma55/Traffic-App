import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Model Feature Impact", page_icon="📊")

st.title("📊 Feature Importance: Model Insights")

try:
    model = joblib.load("best_model.pkl")
    if hasattr(model, "feature_importances_"):
        features = ["CarCount", "BikeCount", "BusCount", "TruckCount", "Hour", "Weekday", "IsWeekend"]
        importance = model.feature_importances_
        imp_df = pd.DataFrame({"Feature": features, "Importance": importance})
        imp_df = imp_df.sort_values(by="Importance", ascending=False)

        fig = px.bar(imp_df, x="Feature", y="Importance", color="Feature",
                     title="Feature Impact on Traffic Volume", color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ This model type does not have feature_importances_.")
except Exception as e:
    st.error(f"Error loading model: {e}")
