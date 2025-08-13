import streamlit as st
import numpy as np
import os
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline


# Page configuration
st.set_page_config(page_title="Wine Quality Prediction", page_icon="🍷", layout="centered")

st.title("🍷 Wine Quality Prediction")
st.markdown("Use this app to train the model and predict wine quality.")


# Sidebar for training
st.sidebar.header("Model Management")
if st.sidebar.button("🔄 Train Model"):
    with st.spinner("Training model... Please wait."):
        exit_code = os.system("python main.py")
    if exit_code == 0:
        st.sidebar.success("✅ Training completed successfully!")
    else:
        st.sidebar.error("❌ Training failed. Check logs.")


st.header("🔍 Predict Wine Quality")

# Input form for prediction
with st.form("prediction_form"):
    fixed_acidity = st.number_input("Fixed Acidity", step=0.01)
    volatile_acidity = st.number_input("Volatile Acidity", step=0.01)
    citric_acid = st.number_input("Citric Acid", step=0.01)
    residual_sugar = st.number_input("Residual Sugar", step=0.01)
    chlorides = st.number_input("Chlorides", step=0.0001)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", step=0.1)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", step=0.1)
    density = st.number_input("Density", step=0.0001)
    pH = st.number_input("pH", step=0.01)
    sulphates = st.number_input("Sulphates", step=0.01)
    alcohol = st.number_input("Alcohol", step=0.01)

    submitted = st.form_submit_button("Predict Quality")

# Prediction logic
if submitted:
    try:
        features = np.array([
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            free_sulfur_dioxide,
            total_sulfur_dioxide,
            density,
            pH,
            sulphates,
            alcohol
        ]).reshape(1, -1)

        pipeline = PredictionPipeline()
        prediction = pipeline.predict(features)

        st.success(f"✅ Predicted Wine Quality: **{prediction[0]}**")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
