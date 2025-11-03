import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import plotly.graph_objects as go
import pydeck as pdk

# ---------------------------
# 1️⃣ Load Model
# ---------------------------
MODEL_PATH = "best_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    st.error(f"⚠️ Error loading model {MODEL_PATH}: {e}")

# ---------------------------
# 2️⃣ Streamlit Config
# ---------------------------
st.set_page_config(page_title="Intelligent Traffic Flow Prediction", page_icon="🚗", layout="wide")

# Custom CSS - Neon dark theme
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #FAFAFA;
}
h2, h3, p, .stMarkdown {
    color: #FAFAFA !important;
}
.metric-card {
    background: linear-gradient(135deg, #1E1E2F, #2D2D44);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 10px #00FFFF33;
}
.high { background-color: #ff4b4b; animation: blink 1s infinite; }
.medium { background-color: #ffa534; }
.low { background-color: #28a745; }
@keyframes blink { 50% {opacity: 0.6;} }
.clock {
    font-size: 20px;
    color: #00FFAA;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 3️⃣ Header & Live Clock
# ---------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<h2>🚦 Intelligent Traffic Flow Prediction Dashboard</h2>", unsafe_allow_html=True)
with col2:
    current_time = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"<div class='clock'>🕒 {current_time}</div>", unsafe_allow_html=True)

st.markdown("Predicts **real-time traffic volume** based on vehicles and time using a trained ML model.")

# ---------------------------
# 4️⃣ Sidebar Input Controls
# ---------------------------
st.sidebar.header("🧾 Enter Traffic Details")

date_time = st.sidebar.time_input("Select Time", datetime.now().time())
hour = date_time.hour
weekday = datetime.now().weekday()
is_weekend = 1 if weekday >= 5 else 0

cars = st.sidebar.slider("Number of Cars", 0, 500, 120)
bikes = st.sidebar.slider("Number of Bikes", 0, 300, 80)
buses = st.sidebar.slider("Number of Buses", 0, 100, 10)
trucks = st.sidebar.slider("Number of Trucks", 0, 100, 15)

# ---------------------------
# 5️⃣ Prepare Input Data
# ---------------------------
input_data = pd.DataFrame({
    "CarCount": [cars],
    "BikeCount": [bikes],
    "BusCount": [buses],
    "TruckCount": [trucks],
    "Hour": [hour],
    "Weekday": [weekday],
    "IsWeekend": [is_weekend]
})

# ---------------------------
# 6️⃣ Prediction & Alerts
# ---------------------------
if model is not None:
    try:
        prediction = float(model.predict(input_data)[0])
        if prediction > 250:
            alert_text, alert_class = "🚨 High Congestion Expected!", "high"
            gauge_color = "red"
        elif prediction > 150:
            alert_text, alert_class = "⚠️ Moderate Traffic — Plan Accordingly.", "medium"
            gauge_color = "orange"
        else:
            alert_text, alert_class = "✅ Smooth Traffic Flow — Roads are Clear!", "low"
            gauge_color = "green"

        st.markdown(f"<div class='{alert_class}' style='padding:15px;border-radius:10px;text-align:center;font-weight:bold;'>{alert_text}</div>", unsafe_allow_html=True)

        # Gauge Meter
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': "Traffic Volume (vehicles/hour)"},
            gauge={
                'axis': {'range': [0, 300]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 150], 'color': "lightgreen"},
                    {'range': [150, 250], 'color': "gold"},
                    {'range': [250, 300], 'color': "crimson"}
                ],
            }
        ))
        fig_gauge.update_layout(paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig_gauge, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")
else:
    st.warning("⚠️ Model not loaded. Please check best_model.pkl")

# ---------------------------
# 7️⃣ NYC Simulated Congestion Map
# ---------------------------
st.subheader("🗺️ Simulated Congestion Zones - NYC")

zones = pd.DataFrame({
    'lat': [40.758, 40.730, 40.712, 40.706, 40.751],
    'lon': [-73.985, -73.997, -74.006, -74.009, -73.977],
    'zone': ['Midtown', 'Greenwich', 'Downtown', 'Financial District', 'Grand Central'],
    'intensity': np.random.uniform(0, 1, 5)
})

layer = pdk.Layer(
    'HeatmapLayer',
    data=zones,
    get_position='[lon, lat]',
    get_weight='intensity',
    radiusPixels=70,
    colorRange=[[0, 255, 0, 100], [255, 255, 0, 140], [255, 0, 0, 180]],
)

view_state = pdk.ViewState(latitude=40.75, longitude=-73.98, zoom=11, pitch=45)
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

# ---------------------------
# 8️⃣ Future Traffic Projection
# ---------------------------
st.subheader("📈 Predicted Trend (Next 6 Hours)")
future_hours = np.arange(hour, hour + 6)
predicted_trend = [prediction * (1 + np.random.uniform(-0.15, 0.15)) for _ in future_hours]
trend_df = pd.DataFrame({"Hour": future_hours % 24, "Predicted Volume": predicted_trend})

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=trend_df["Hour"], y=trend_df["Predicted Volume"],
    mode='lines+markers', line=dict(color="#00FFFF", width=3)
))
fig_trend.update_layout(
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    font_color="white", xaxis_title="Hour", yaxis_title="Predicted Volume"
)
st.plotly_chart(fig_trend, use_container_width=True)

# ---------------------------
# 9️⃣ Input Summary
# ---------------------------
with st.expander("🔍 View Input Details"):
    st.dataframe(input_data)
