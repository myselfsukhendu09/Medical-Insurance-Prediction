# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Copy Your Data

Copy your CSV file to the project:
```bash
# Create data directory if it doesn't exist
mkdir -p data\raw

# Copy your data file
copy "C:\Users\mysel\Documents\python_final\Worksheet in Medical Insurance cost prediction.csv" data\raw\insurance.csv
```

### Step 3: Run the Complete Pipeline

```bash
# Run the main pipeline
python src\main.py
```

This will:
- ✅ Load and explore the data
- ✅ Perform EDA with visualizations
- ✅ Clean and preprocess data
- ✅ Engineer features
- ✅ Train 13 different models
- ✅ Evaluate and compare models
- ✅ Save all results and models

### Step 4: Launch the Web Application

```bash
# Navigate to web app directory
cd web_app

# Run the Flask app
python app.py
```

Then open your browser and go to: **http://localhost:5000**

---

## 📊 What You'll Get

### 1. **Trained Models** (in `models/` directory)
- Linear Regression
- Ridge & Lasso Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- SVR
- KNN

### 2. **Visualizations** (in `results/figures/` directory)
- Target distribution plots
- Feature distributions
- Correlation heatmap
- Model comparison charts
- Prediction vs actual plots
- Residual plots

### 3. **Results** (in `results/` directory)
- `model_comparison.csv` - Complete metrics for all models
- `model_comparison_simple.csv` - Simplified comparison table

### 4. **Processed Data** (in `data/processed/` directory)
- `cleaned_data.csv` - After preprocessing
- `engineered_data.csv` - After feature engineering

---

## 🎯 Running Individual Components

### Just Data Exploration
```bash
python src\data_loader.py
```

### Just EDA
```bash
python src\eda.py
```

### Just Preprocessing
```bash
python src\preprocessing.py
```

### Just Feature Engineering
```bash
python src\feature_engineering.py
```

### Just Model Training
```bash
python src\model_training.py
```

### Just Model Evaluation
```bash
python src\model_evaluation.py
```

---

## 🔧 Customization

### Change the Data Path

Edit `src/main.py` and update the `data_path` variable:
```python
data_path = r"YOUR_PATH_HERE\your_data.csv"
```

### Enable Hyperparameter Tuning

In `src/main.py`, change:
```python
run_tuning=False  # Change to True
```

**Note:** This will take significantly longer but may improve model performance.

### Skip EDA Visualizations

In `src/main.py`, change:
```python
run_eda=False  # Change to False to skip
```

---

## 📈 Expected Results

Based on typical insurance datasets, you should see:

- **Best Model**: Usually XGBoost or Gradient Boosting
- **R² Score**: 0.85 - 0.90 (85-90% variance explained)
- **RMSE**: $4,000 - $6,000 (depending on data)

### Key Insights:
1. **Smoking status** is typically the strongest predictor (2-3x cost increase)
2. **Age** shows positive correlation with costs
3. **BMI** has moderate impact, especially for smokers
4. **Region** usually has minimal impact

---

## 🐛 Troubleshooting

### Issue: "Module not found"
**Solution:** Make sure you're in the project root directory and have activated the virtual environment.

### Issue: "File not found"
**Solution:** Check that your data file path is correct in `src/main.py`.

### Issue: "Model not loaded" in web app
**Solution:** Run the training pipeline first: `python src\main.py`

### Issue: Port 5000 already in use
**Solution:** Change the port in `web_app/app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

---

## 📚 Next Steps

1. **Review Results**: Check `results/model_comparison.csv` for detailed metrics
2. **Analyze Visualizations**: Look at plots in `results/figures/`
3. **Test Web App**: Try different inputs in the web interface
4. **Experiment**: Try different preprocessing techniques or models
5. **Deploy**: Consider deploying the web app to a cloud platform

---

## 💡 Tips

- Start with `run_tuning=False` for faster initial results
- Review EDA visualizations to understand your data better
- The web app uses the best trained model automatically
- All models and preprocessors are saved for reuse
- Check the console output for detailed progress information

---

## 🆘 Need Help?

- Check the main README.md for detailed documentation
- Review individual module docstrings for function details
- Examine the example outputs in `results/`

---

**Happy Predicting! 🎉**
