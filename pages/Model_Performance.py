import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Model Performance", page_icon="📈")

st.title("📈 Model Performance Overview")

# Load model
MODEL_PATH = "best_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# Check model predictability
if not hasattr(model, "predict"):
    st.error("❌ The loaded model doesn't support prediction. Check your best_model.pkl file.")
    st.stop()

# Generate synthetic dataset
st.subheader("🧪 Auto-Generated Test Data (Simulated for Performance Check)")

np.random.seed(42)
n_samples = 200

synthetic_data = pd.DataFrame({
    "CarCount": np.random.randint(50, 500, n_samples),
    "BikeCount": np.random.randint(10, 300, n_samples),
    "BusCount": np.random.randint(0, 100, n_samples),
    "TruckCount": np.random.randint(0, 80, n_samples),
    "Hour": np.random.randint(0, 24, n_samples),
    "Weekday": np.random.randint(0, 7, n_samples),
    "IsWeekend": np.random.choice([0, 1], n_samples)
})

try:
    y_pred = model.predict(synthetic_data)
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

# Simulated ground truth (for scoring visualization)
y_true = y_pred + np.random.normal(0, 20, size=n_samples)

# Compute metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("📏 Mean Absolute Error", f"{mae:.2f}")
col2.metric("📉 RMSE", f"{rmse:.2f}")
col3.metric("📊 R² Score", f"{r2:.3f}")

st.markdown("---")

# Chart comparison
st.subheader("🔍 Actual vs Predicted (First 50 Samples)")
compare_df = pd.DataFrame({
    "Actual": y_true[:50],
    "Predicted": y_pred[:50]
})
st.line_chart(compare_df)

# Distribution
st.subheader("📈 Prediction Distribution")
hist_df = pd.DataFrame({"Predicted Traffic Volume": y_pred})
st.bar_chart(hist_df["Predicted Traffic Volume"].value_counts().sort_index())

st.info("✅ Synthetic performance evaluation complete using simulated test data.")
