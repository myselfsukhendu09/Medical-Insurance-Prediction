import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Set page configuration
st.set_page_config(
    page_title="MedSure | Medical Insurance Predictor",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #2e7d32;
        color: white;
        height: 3em;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background-color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        text-align: center;
    }
    .insight-item {
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
    }
    .insight-warning {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    </style>
    """, unsafe_allow_stdio=True)

# Load model and preprocessors
@st.cache_resource
def load_assets():
    model_paths = ['models/xgboost.pkl', 'models/random_forest.pkl', 'models/gradient_boosting.pkl']
    model = None
    for path in model_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            break
    
    scalers = joblib.load('models/scalers.pkl') if os.path.exists('models/scalers.pkl') else None
    return model, scalers

model, scalers = load_assets()

def preprocess_input(age, sex, bmi, children, smoker, region):
    # Encoding
    sex_encoded = 1 if sex.lower() == 'male' else 0
    smoker_encoded = 1 if smoker.lower() == 'yes' else 0
    
    region_mapping = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
    region_encoded = region_mapping.get(region.lower(), 0)
    
    # Interaction features
    bmi_smoker = bmi * smoker_encoded
    age_bmi = age * bmi
    
    features = np.array([[
        age, sex_encoded, bmi, children, smoker_encoded, region_encoded, bmi_smoker, age_bmi
    ]])
    
    return features

# App UI
st.title("🏥 MedSure AI Insurance Predictor")
st.markdown("Estimate your annual medical insurance costs using high-precision machine learning.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Patient Demographics")
    sub_col1, sub_col2 = st.columns(2)
    
    with sub_col1:
        age = st.slider("Age", 18, 100, 30)
        sex = st.selectbox("Gender", ["Male", "Female"])
        bmi = st.number_input("BMI (Body Mass Index)", 10.0, 60.0, 25.0, step=0.1)
    
    with sub_col2:
        children = st.select_slider("Number of Children", options=[0, 1, 2, 3, 4, 5])
        smoker = st.radio("Do you smoke?", ["No", "Yes"])
        region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

    if st.button("Predict Insurance Cost"):
        if model is not None:
            features = preprocess_input(age, sex, bmi, children, smoker, region)
            prediction = model.predict(features)[0]
            prediction = max(0, prediction)
            
            st.session_state['prediction'] = prediction
            st.session_state['inputs'] = {
                'age': age, 'sex': sex, 'bmi': bmi, 
                'children': children, 'smoker': smoker, 'region': region
            }
        else:
            st.error("Model files not found. Please ensure the 'models' folder exists.")

with col2:
    if 'prediction' in st.session_state:
        pred = st.session_state['prediction']
        st.markdown(f"""
            <div class="prediction-card">
                <h3>Estimated Annual Cost</h3>
                <h1 style="color: #2e7d32;">${pred:,.2f}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        st.subheader("💡 AI Insights")
        
        # Insights logic
        insights = []
        if age > 50: insights.append(("⚠ Age over 50 increases risk factor", "warning"))
        if bmi >= 30: insights.append(("⚠ BMI in obese range may spike premiums", "warning"))
        if smoker == "Yes": insights.append(("⚠ Smoking status is the primary cost driver", "warning"))
        if bmi < 25: insights.append(("✓ Healthy BMI helps lower costs", "success"))
        if smoker == "No": insights.append(("✓ Non-smoker discount applied", "success"))
        
        for text, type in insights:
            div_class = "insight-item" + (" insight-warning" if type == "warning" else "")
            st.markdown(f'<div class="{div_class}">{text}</div>', unsafe_allow_html=True)
    else:
        st.info("Enter details and click 'Predict' to see the estimate.")

st.divider()
st.caption("Disclaimer: This tool provides estimates for educational purposes and is not a binding insurance quote.")
