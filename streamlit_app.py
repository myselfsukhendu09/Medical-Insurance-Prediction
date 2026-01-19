
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(page_title="Medical Insurance Predictor", page_icon="🏥")

st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("---")

# Feature Mapping
region_mapping = {'Northeast': 0, 'Northwest': 1, 'Southeast': 2, 'Southwest': 3}
sex_mapping = {'Female': 0, 'Male': 1}
smoker_mapping = {'No': 0, 'Yes': 1}

# Load Model
MODEL_PATHS = [
    'models/xgboost.pkl',
    'models/random_forest.pkl',
    'models/gradient_boosting.pkl',
    'models/best_model.pkl'
]

model = None
for path in MODEL_PATHS:
    if os.path.exists(path):
        model = joblib.load(path)
        break

if model is None:
    st.error("No trained model found! Please run the training pipeline first.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    sex = st.selectbox("Gender", ['Female', 'Male'])
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

with col2:
    children = st.number_input("Number of Children", 0, 10, 0)
    smoker = st.selectbox("Smoker", ['No', 'Yes'])
    region = st.selectbox("Region", ['Northeast', 'Northwest', 'Southeast', 'Southwest'])

if st.button("Predict Insurance Charges"):
    # Encoding
    sex_encoded = sex_mapping[sex]
    smoker_encoded = smoker_mapping[smoker]
    region_encoded = region_mapping[region]
    
    # Interaction features (based on the model's expected features in app.py)
    bmi_smoker = bmi * smoker_encoded
    age_bmi = age * bmi
    
    # Construct feature array
    # Based on app.py: [age, sex, bmi, children, smoker, region, bmi_smoker, age_bmi]
    features = np.array([[
        age, sex_encoded, bmi, children, smoker_encoded, region_encoded, bmi_smoker, age_bmi
    ]])
    
    try:
        prediction = model.predict(features)[0]
        prediction = max(0, prediction)
        
        st.markdown("---")
        st.success(f"### Predicted Insurance Cost: ${prediction:,.2f}")
        
        # Insights
        st.subheader("Key Factors & Insights")
        if smoker == 'Yes':
            st.warning("⚠ Smoking status is the highest driver of your premium (typically 2-3x higher).")
        if bmi >= 30:
            st.warning("⚠ High BMI (obese range) contributed to a higher estimation.")
        if age > 50:
            st.info("ℹ Age-related risk factors are factored into this cost.")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Ensure the model was trained with the interaction features (bmi_smoker, age_bmi).")
