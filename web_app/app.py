"""
Flask Web Application for Medical Insurance Cost Prediction
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)

# Global variables for model and encoders
model = None
encoders = None
scalers = None
feature_names = None


def load_model_and_preprocessors():
    """
    Load the best model and preprocessing objects
    """
    global model, encoders, scalers, feature_names
    
    try:
        # Try to load the best model (you can change this to any model)
        model_path = '../models/xgboost.pkl'
        if not os.path.exists(model_path):
            model_path = '../models/random_forest.pkl'
        if not os.path.exists(model_path):
            model_path = '../models/gradient_boosting.pkl'
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"✓ Loaded model from {model_path}")
        else:
            print("⚠ No trained model found. Please train models first.")
            return False
        
        # Load encoders if available
        encoders_path = '../models/encoders.pkl'
        if os.path.exists(encoders_path):
            encoders = joblib.load(encoders_path)
            print(f"✓ Loaded encoders from {encoders_path}")
        
        # Load scalers if available
        scalers_path = '../models/scalers.pkl'
        if os.path.exists(scalers_path):
            scalers = joblib.load(scalers_path)
            print(f"✓ Loaded scalers from {scalers_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return False


def preprocess_input(age, sex, bmi, children, smoker, region):
    """
    Preprocess user input to match training data format
    
    Parameters:
    -----------
    age : int
    sex : str
    bmi : float
    children : int
    smoker : str
    region : str
    
    Returns:
    --------
    np.array
        Preprocessed features ready for prediction
    """
    # Create a dictionary with input data
    data = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }
    
    # Encode categorical variables
    sex_encoded = 1 if sex.lower() == 'male' else 0
    smoker_encoded = 1 if smoker.lower() == 'yes' else 0
    
    # Encode region (simple encoding)
    region_mapping = {
        'northeast': 0,
        'northwest': 1,
        'southeast': 2,
        'southwest': 3
    }
    region_encoded = region_mapping.get(region.lower(), 0)
    
    # Create interaction features
    smoker_binary = smoker_encoded
    bmi_smoker = bmi * smoker_binary
    age_bmi = age * bmi
    
    # Create feature array (adjust based on your trained model's features)
    # This is a simplified version - adjust based on actual features used
    features = np.array([[
        age,
        sex_encoded,
        bmi,
        children,
        smoker_encoded,
        region_encoded,
        bmi_smoker,
        age_bmi
    ]])
    
    return features


@app.route('/')
def home():
    """
    Render home page
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction request
    """
    try:
        # Get form data
        data = request.get_json()
        
        age = int(data['age'])
        sex = data['sex']
        bmi = float(data['bmi'])
        children = int(data['children'])
        smoker = data['smoker']
        region = data['region']
        
        # Validate inputs
        if age < 18 or age > 100:
            return jsonify({'error': 'Age must be between 18 and 100'}), 400
        
        if bmi < 10 or bmi > 60:
            return jsonify({'error': 'BMI must be between 10 and 60'}), 400
        
        if children < 0 or children > 10:
            return jsonify({'error': 'Number of children must be between 0 and 10'}), 400
        
        # Preprocess input
        features = preprocess_input(age, sex, bmi, children, smoker, region)
        
        # Make prediction
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        prediction = model.predict(features)[0]
        
        # Ensure prediction is positive
        prediction = max(0, prediction)
        
        # Create response with additional insights
        response = {
            'prediction': round(prediction, 2),
            'formatted_prediction': f"${prediction:,.2f}",
            'inputs': {
                'age': age,
                'sex': sex,
                'bmi': bmi,
                'children': children,
                'smoker': smoker,
                'region': region
            },
            'insights': generate_insights(age, sex, bmi, children, smoker, region, prediction)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def generate_insights(age, sex, bmi, children, smoker, region, prediction):
    """
    Generate insights based on input parameters
    
    Returns:
    --------
    list
        List of insight strings
    """
    insights = []
    
    # Age insights
    if age < 30:
        insights.append("✓ Younger age typically results in lower insurance costs")
    elif age > 50:
        insights.append("⚠ Age over 50 may increase insurance premiums")
    
    # BMI insights
    if bmi < 25:
        insights.append("✓ Healthy BMI range (below 25)")
    elif bmi >= 30:
        insights.append("⚠ BMI in obese range (30+) may increase costs")
    elif bmi >= 25:
        insights.append("⚠ BMI in overweight range (25-30)")
    
    # Smoker insights
    if smoker.lower() == 'yes':
        insights.append("⚠ Smoking status significantly increases insurance costs (typically 2-3x higher)")
    else:
        insights.append("✓ Non-smoker status helps reduce insurance costs")
    
    # Children insights
    if children > 3:
        insights.append("⚠ Higher number of dependents may increase costs")
    
    # Overall cost assessment
    if prediction < 5000:
        insights.append("✓ Predicted cost is in the lower range")
    elif prediction > 20000:
        insights.append("⚠ Predicted cost is in the higher range")
    else:
        insights.append("ℹ Predicted cost is in the moderate range")
    
    return insights


@app.route('/health')
def health():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    print("\n" + "="*80)
    print("MEDICAL INSURANCE COST PREDICTION - WEB APPLICATION")
    print("="*80 + "\n")
    
    # Load model and preprocessors
    if load_model_and_preprocessors():
        print("\n✓ Application ready!")
        print("\nStarting Flask server...")
        print("Open your browser and navigate to: http://localhost:5000")
        print("\nPress Ctrl+C to stop the server\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n✗ Failed to load model. Please train the model first by running:")
        print("  python src/model_training.py")
