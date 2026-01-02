"""
Model Evaluation Module
Comprehensive evaluation of trained models with multiple metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                            mean_absolute_percentage_error, explained_variance_score)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Class to evaluate and compare regression models
    """
    
    def __init__(self, models_dict, X_train, X_test, y_train, y_test):
        """
        Initialize ModelEvaluator
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary of trained models
        X_train, X_test : pd.DataFrame or np.array
            Training and testing features
        y_train, y_test : pd.Series or np.array
            Training and testing target
        """
        self.models = models_dict
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.results = {}
        self.comparison_df = None
        
    def calculate_adjusted_r2(self, r2, n_samples, n_features):
        """
        Calculate Adjusted R² Score
        
        Parameters:
        -----------
        r2 : float
            R² score
        n_samples : int
            Number of samples
        n_features : int
            Number of features
            
        Returns:
        --------
        float
            Adjusted R² score
        """
        adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        return adjusted_r2
    
    def evaluate_model(self, model, model_name):
        """
        Evaluate a single model on both train and test sets
        
        Parameters:
        -----------
        model : sklearn model
            Trained model
        model_name : str
            Name of the model
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        # Make predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Calculate metrics for training set
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(self.y_train, y_train_pred)
        train_adj_r2 = self.calculate_adjusted_r2(train_r2, len(self.y_train), 
                                                   self.X_train.shape[1])
        train_mape = mean_absolute_percentage_error(self.y_train, y_train_pred) * 100
        train_evs = explained_variance_score(self.y_train, y_train_pred)
        
        # Calculate metrics for testing set
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_r2 = r2_score(self.y_test, y_test_pred)
        test_adj_r2 = self.calculate_adjusted_r2(test_r2, len(self.y_test), 
                                                  self.X_test.shape[1])
        test_mape = mean_absolute_percentage_error(self.y_test, y_test_pred) * 100
        test_evs = explained_variance_score(self.y_test, y_test_pred)
        
        # Check for overfitting
        r2_diff = train_r2 - test_r2
        if r2_diff > 0.1:
            overfitting = "Yes"
        elif r2_diff > 0.05:
            overfitting = "Slight"
        else:
            overfitting = "No"
        
        results = {
            'Model': model_name,
            'Train_MAE': train_mae,
            'Test_MAE': test_mae,
            'Train_MSE': train_mse,
            'Test_MSE': test_mse,
            'Train_RMSE': train_rmse,
            'Test_RMSE': test_rmse,
            'Train_R2': train_r2,
            'Test_R2': test_r2,
            'Train_Adj_R2': train_adj_r2,
            'Test_Adj_R2': test_adj_r2,
            'Train_MAPE': train_mape,
            'Test_MAPE': test_mape,
            'Train_EVS': train_evs,
            'Test_EVS': test_evs,
            'Overfitting': overfitting,
            'R2_Difference': r2_diff
        }
        
        return results
    
    def evaluate_all_models(self):
        """
        Evaluate all models and create comparison dataframe
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with all evaluation metrics
        """
        print(f"\n{'='*80}")
        print("EVALUATING ALL MODELS")
        print(f"{'='*80}\n")
        
        all_results = []
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...", end=' ')
            
            try:
                results = self.evaluate_model(model, name)
                all_results.append(results)
                self.results[name] = results
                print("✓ Done")
            except Exception as e:
                print(f"✗ Failed: {str(e)}")
        
        self.comparison_df = pd.DataFrame(all_results)
        
        # Sort by Test R2 score
        self.comparison_df = self.comparison_df.sort_values('Test_R2', ascending=False)
        
        print(f"\n✓ Evaluated {len(all_results)} models")
        
        return self.comparison_df
    
    def print_detailed_results(self):
        """
        Print detailed results for all models
        """
        if self.comparison_df is None:
            self.evaluate_all_models()
        
        print(f"\n{'='*80}")
        print("DETAILED MODEL EVALUATION RESULTS")
        print(f"{'='*80}\n")
        
        for _, row in self.comparison_df.iterrows():
            print(f"\n{'-'*80}")
            print(f"MODEL: {row['Model']}")
            print(f"{'-'*80}")
            
            print(f"\nTraining Set Metrics:")
            print(f"  MAE:         ${row['Train_MAE']:,.2f}")
            print(f"  MSE:         {row['Train_MSE']:,.2f}")
            print(f"  RMSE:        ${row['Train_RMSE']:,.2f}")
            print(f"  R² Score:    {row['Train_R2']:.4f}")
            print(f"  Adj R²:      {row['Train_Adj_R2']:.4f}")
            print(f"  MAPE:        {row['Train_MAPE']:.2f}%")
            print(f"  Expl Var:    {row['Train_EVS']:.4f}")
            
            print(f"\nTesting Set Metrics:")
            print(f"  MAE:         ${row['Test_MAE']:,.2f}")
            print(f"  MSE:         {row['Test_MSE']:,.2f}")
            print(f"  RMSE:        ${row['Test_RMSE']:,.2f}")
            print(f"  R² Score:    {row['Test_R2']:.4f}")
            print(f"  Adj R²:      {row['Test_Adj_R2']:.4f}")
            print(f"  MAPE:        {row['Test_MAPE']:.2f}%")
            print(f"  Expl Var:    {row['Test_EVS']:.4f}")
            
            print(f"\nOverfitting Analysis:")
            print(f"  R² Difference: {row['R2_Difference']:.4f}")
            print(f"  Overfitting:   {row['Overfitting']}")
    
    def print_comparison_table(self):
        """
        Print simplified comparison table
        """
        if self.comparison_df is None:
            self.evaluate_all_models()
        
        print(f"\n{'='*80}")
        print("MODEL COMPARISON TABLE")
        print(f"{'='*80}\n")
        
        # Create simplified table
        comparison_table = self.comparison_df[[
            'Model', 'Train_RMSE', 'Test_RMSE', 'Train_R2', 'Test_R2', 'Overfitting'
        ]].copy()
        
        # Round numerical values
        comparison_table['Train_RMSE'] = comparison_table['Train_RMSE'].round(2)
        comparison_table['Test_RMSE'] = comparison_table['Test_RMSE'].round(2)
        comparison_table['Train_R2'] = comparison_table['Train_R2'].round(4)
        comparison_table['Test_R2'] = comparison_table['Test_R2'].round(4)
        
        print(comparison_table.to_string(index=False))
        
        # Highlight best model
        best_model = comparison_table.iloc[0]['Model']
        best_r2 = comparison_table.iloc[0]['Test_R2']
        
        print(f"\n{'='*80}")
        print(f"🏆 BEST MODEL: {best_model}")
        print(f"   Test R² Score: {best_r2:.4f}")
        print(f"{'='*80}")
        
        return comparison_table
    
    def plot_model_comparison(self):
        """
        Create visualization comparing all models
        """
        if self.comparison_df is None:
            self.evaluate_all_models()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. R² Score Comparison
        ax1 = axes[0, 0]
        x = np.arange(len(self.comparison_df))
        width = 0.35
        
        ax1.bar(x - width/2, self.comparison_df['Train_R2'], width, 
               label='Train R²', alpha=0.8, color='skyblue')
        ax1.bar(x + width/2, self.comparison_df['Test_R2'], width, 
               label='Test R²', alpha=0.8, color='coral')
        
        ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax1.set_title('R² Score Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.comparison_df['Model'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. RMSE Comparison
        ax2 = axes[0, 1]
        ax2.bar(x - width/2, self.comparison_df['Train_RMSE'], width, 
               label='Train RMSE', alpha=0.8, color='lightgreen')
        ax2.bar(x + width/2, self.comparison_df['Test_RMSE'], width, 
               label='Test RMSE', alpha=0.8, color='salmon')
        
        ax2.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax2.set_ylabel('RMSE', fontsize=12, fontweight='bold')
        ax2.set_title('RMSE Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.comparison_df['Model'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Test R² Ranking
        ax3 = axes[1, 0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.comparison_df)))
        ax3.barh(self.comparison_df['Model'], self.comparison_df['Test_R2'], 
                color=colors, alpha=0.8)
        ax3.set_xlabel('Test R² Score', fontsize=12, fontweight='bold')
        ax3.set_title('Model Ranking by Test R²', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Overfitting Analysis
        ax4 = axes[1, 1]
        overfitting_counts = self.comparison_df['Overfitting'].value_counts()
        colors_pie = ['#90EE90', '#FFD700', '#FF6B6B']
        ax4.pie(overfitting_counts.values, labels=overfitting_counts.index, 
               autopct='%1.1f%%', startangle=90, colors=colors_pie)
        ax4.set_title('Overfitting Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs('results/figures', exist_ok=True)
        plt.savefig('results/figures/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n✓ Model comparison plot saved to results/figures/model_comparison.png")
    
    def plot_predictions_vs_actual(self, model_name):
        """
        Plot predictions vs actual values for a specific model
        
        Parameters:
        -----------
        model_name : str
            Name of the model to plot
        """
        if model_name not in self.models:
            print(f"Model '{model_name}' not found!")
            return
        
        model = self.models[model_name]
        
        # Make predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Training set
        axes[0].scatter(self.y_train, y_train_pred, alpha=0.5, color='blue')
        axes[0].plot([self.y_train.min(), self.y_train.max()], 
                    [self.y_train.min(), self.y_train.max()], 
                    'r--', lw=2)
        axes[0].set_xlabel('Actual Charges', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Predicted Charges', fontsize=12, fontweight='bold')
        axes[0].set_title(f'{model_name} - Training Set', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Testing set
        axes[1].scatter(self.y_test, y_test_pred, alpha=0.5, color='green')
        axes[1].plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 
                    'r--', lw=2)
        axes[1].set_xlabel('Actual Charges', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Predicted Charges', fontsize=12, fontweight='bold')
        axes[1].set_title(f'{model_name} - Testing Set', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"results/figures/{model_name.lower().replace(' ', '_')}_predictions.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✓ Predictions plot saved to {filename}")
    
    def plot_residuals(self, model_name):
        """
        Plot residuals for a specific model
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        """
        if model_name not in self.models:
            print(f"Model '{model_name}' not found!")
            return
        
        model = self.models[model_name]
        
        # Calculate residuals
        y_test_pred = model.predict(self.X_test)
        residuals = self.y_test - y_test_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Residual plot
        axes[0].scatter(y_test_pred, residuals, alpha=0.5, color='purple')
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Residuals', fontsize=12, fontweight='bold')
        axes[0].set_title(f'{model_name} - Residual Plot', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residual distribution
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1].set_xlabel('Residuals', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title(f'{model_name} - Residual Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        filename = f"results/figures/{model_name.lower().replace(' ', '_')}_residuals.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✓ Residuals plot saved to {filename}")
    
    def save_results(self, output_dir='results'):
        """
        Save evaluation results to CSV
        
        Parameters:
        -----------
        output_dir : str
            Directory to save results
        """
        if self.comparison_df is None:
            self.evaluate_all_models()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full results
        filepath = os.path.join(output_dir, 'model_comparison.csv')
        self.comparison_df.to_csv(filepath, index=False)
        print(f"\n✓ Results saved to {filepath}")
        
        # Save simplified comparison table
        comparison_table = self.comparison_df[[
            'Model', 'Train_RMSE', 'Test_RMSE', 'Train_R2', 'Test_R2', 'Overfitting'
        ]]
        filepath_simple = os.path.join(output_dir, 'model_comparison_simple.csv')
        comparison_table.to_csv(filepath_simple, index=False)
        print(f"✓ Simplified results saved to {filepath_simple}")


def main():
    """
    Main function to run model evaluation
    """
    from model_training import prepare_data_for_training
    import pandas as pd
    
    # Load data
    if os.path.exists('data/processed/engineered_data.csv'):
        df = pd.read_csv('data/processed/engineered_data.csv')
    else:
        print("Error: Engineered data not found.")
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data_for_training(df)
    
    # Load trained models
    models_dir = 'models'
    models = {}
    
    model_files = [
        'linear_regression.pkl',
        'decision_tree.pkl',
        'random_forest.pkl',
        'gradient_boosting.pkl',
        'svr.pkl',
        'knn.pkl',
        'xgboost.pkl'
    ]
    
    for model_file in model_files:
        filepath = os.path.join(models_dir, model_file)
        if os.path.exists(filepath):
            model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
            models[model_name] = joblib.load(filepath)
    
    if not models:
        print("No trained models found. Run model_training.py first.")
        return
    
    # Initialize evaluator
    evaluator = ModelEvaluator(models, X_train, X_test, y_train, y_test)
    
    # Evaluate all models
    evaluator.evaluate_all_models()
    
    # Print results
    evaluator.print_detailed_results()
    evaluator.print_comparison_table()
    
    # Create visualizations
    evaluator.plot_model_comparison()
    
    # Plot best model details
    best_model_name = evaluator.comparison_df.iloc[0]['Model']
    evaluator.plot_predictions_vs_actual(best_model_name)
    evaluator.plot_residuals(best_model_name)
    
    # Save results
    evaluator.save_results()
    
    print("\n" + "="*80)
    print("MODEL EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
