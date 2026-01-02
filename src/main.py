"""
Main Pipeline Script
Runs the complete ML pipeline from data loading to model evaluation
"""

import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
from eda import EDAAnalyzer
from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer, prepare_data_for_training
from model_evaluation import ModelEvaluator


def create_directories():
    """Create necessary directories"""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'results/figures',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Created project directories")


def run_complete_pipeline(data_path, run_eda=True, run_tuning=False):
    """
    Run the complete ML pipeline
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV data file
    run_eda : bool
        Whether to run EDA (generates visualizations)
    run_tuning : bool
        Whether to run hyperparameter tuning (takes longer)
    """
    print("\n" + "="*80)
    print("MEDICAL INSURANCE COST PREDICTION - COMPLETE PIPELINE")
    print("="*80 + "\n")
    
    # Create directories
    create_directories()
    
    # Step 1: Load Data
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING")
    print("="*80)
    
    loader = DataLoader(data_path)
    df = loader.load_data()
    loader.get_basic_info()
    
    # Step 2: Exploratory Data Analysis
    if run_eda:
        print("\n" + "="*80)
        print("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        eda = EDAAnalyzer(df)
        eda.generate_eda_report()
    
    # Step 3: Data Preprocessing
    print("\n" + "="*80)
    print("STEP 3: DATA PREPROCESSING")
    print("="*80)
    
    preprocessor = DataPreprocessor(df)
    preprocessor.check_missing_values()
    preprocessor.handle_missing_values(strategy='mean')
    preprocessor.check_duplicates()
    preprocessor.remove_duplicates()
    preprocessor.detect_outliers_iqr()
    
    # Note: We're not removing outliers from charges as they're legitimate high costs
    # Only cap outliers in age and bmi if needed
    # preprocessor.handle_outliers(method='cap', columns=['age', 'bmi'])
    
    df_cleaned = preprocessor.get_preprocessing_summary()
    
    # Save cleaned data
    df_cleaned.to_csv('data/processed/cleaned_data.csv', index=False)
    print("\n✓ Saved cleaned data to data/processed/cleaned_data.csv")
    
    # Step 4: Feature Engineering
    print("\n" + "="*80)
    print("STEP 4: FEATURE ENGINEERING")
    print("="*80)
    
    fe = FeatureEngineer(df_cleaned)
    
    # Encode categorical variables
    categorical_cols = ['sex', 'smoker', 'region']
    fe.encode_categorical_label(categorical_cols)
    
    # Create interaction features
    fe.create_interaction_features()
    
    # Check and handle skewness
    skewness_df = fe.check_skewness(threshold=0.5)
    
    # Scale numerical features
    numerical_cols = ['age', 'bmi', 'children']
    fe.scale_features(columns=numerical_cols, method='standard')
    
    df_engineered = fe.get_feature_engineering_summary()
    
    # Save encoders and scalers
    fe.save_encoders_scalers()
    
    # Save engineered data
    df_engineered.to_csv('data/processed/engineered_data.csv', index=False)
    print("\n✓ Saved engineered data to data/processed/engineered_data.csv")
    
    # Step 5: Prepare Data for Training
    print("\n" + "="*80)
    print("STEP 5: DATA PREPARATION")
    print("="*80)
    
    X_train, X_test, y_train, y_test = prepare_data_for_training(
        df_engineered, 
        target_col='charges',
        test_size=0.2,
        random_state=42
    )
    
    # Step 6: Model Training
    print("\n" + "="*80)
    print("STEP 6: MODEL TRAINING")
    print("="*80)
    
    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    
    # Initialize and train all models
    trainer.initialize_models()
    trainer.train_all_models()
    
    # Cross-validation
    cv_results = trainer.cross_validate_models(cv=5)
    
    # Hyperparameter tuning (optional)
    if run_tuning:
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING (This may take a while...)")
        print("="*80)
        trainer.tune_all_models()
    
    # Save models
    trainer.save_models()
    
    # Step 7: Model Evaluation
    print("\n" + "="*80)
    print("STEP 7: MODEL EVALUATION")
    print("="*80)
    
    evaluator = ModelEvaluator(trainer.trained_models, X_train, X_test, y_train, y_test)
    
    # Evaluate all models
    comparison_df = evaluator.evaluate_all_models()
    
    # Print results
    evaluator.print_detailed_results()
    comparison_table = evaluator.print_comparison_table()
    
    # Create visualizations
    evaluator.plot_model_comparison()
    
    # Plot best model details
    best_model_name = comparison_df.iloc[0]['Model']
    evaluator.plot_predictions_vs_actual(best_model_name)
    evaluator.plot_residuals(best_model_name)
    
    # Save results
    evaluator.save_results()
    
    # Final Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    
    print("\n📊 Results Summary:")
    print(f"  - Best Model: {best_model_name}")
    print(f"  - Test R² Score: {comparison_df.iloc[0]['Test_R2']:.4f}")
    print(f"  - Test RMSE: ${comparison_df.iloc[0]['Test_RMSE']:,.2f}")
    
    print("\n📁 Output Files:")
    print("  - Cleaned Data: data/processed/cleaned_data.csv")
    print("  - Engineered Data: data/processed/engineered_data.csv")
    print("  - Models: models/*.pkl")
    print("  - Results: results/model_comparison.csv")
    print("  - Figures: results/figures/*.png")
    
    print("\n🚀 Next Steps:")
    print("  1. Review the model comparison results")
    print("  2. Check the visualizations in results/figures/")
    print("  3. Launch the web app: cd web_app && python app.py")
    
    print("\n" + "="*80 + "\n")
    
    return {
        'best_model': best_model_name,
        'comparison_df': comparison_df,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def main():
    """
    Main function
    """
    # Path to your data file
    data_path = r"C:\Users\mysel\Documents\python_final\Worksheet in Medical Insurance cost prediction.csv"
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please update the data_path variable in src/main.py")
        return
    
    # Run pipeline
    results = run_complete_pipeline(
        data_path=data_path,
        run_eda=True,      # Set to False to skip EDA visualizations
        run_tuning=False   # Set to True to run hyperparameter tuning (takes longer)
    )
    
    print("✓ Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
