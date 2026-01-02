"""
Model Training Module
Trains multiple regression models and performs hyperparameter tuning
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Class to handle model training and hyperparameter tuning
    """
    
    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize ModelTrainer
        
        Parameters:
        -----------
        X_train, X_test : pd.DataFrame or np.array
            Training and testing features
        y_train, y_test : pd.Series or np.array
            Training and testing target
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.models = {}
        self.trained_models = {}
        self.best_models = {}
        
    def initialize_models(self):
        """
        Initialize all regression models
        """
        print(f"\n{'='*80}")
        print("INITIALIZING REGRESSION MODELS")
        print(f"{'='*80}\n")
        
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'ElasticNet': ElasticNet(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'AdaBoost': AdaBoostRegressor(random_state=42),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor(),
            'XGBoost': XGBRegressor(random_state=42, n_jobs=-1),
            'LightGBM': LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
            'CatBoost': CatBoostRegressor(random_state=42, verbose=0)
        }
        
        print(f"✓ Initialized {len(self.models)} regression models:")
        for i, model_name in enumerate(self.models.keys(), 1):
            print(f"  {i}. {model_name}")
        
        return self.models
    
    def train_all_models(self):
        """
        Train all initialized models
        
        Returns:
        --------
        dict
            Dictionary of trained models
        """
        print(f"\n{'='*80}")
        print("TRAINING ALL MODELS")
        print(f"{'='*80}\n")
        
        if not self.models:
            self.initialize_models()
        
        for name, model in self.models.items():
            print(f"Training {name}...", end=' ')
            
            try:
                model.fit(self.X_train, self.y_train)
                self.trained_models[name] = model
                print("✓ Done")
            except Exception as e:
                print(f"✗ Failed: {str(e)}")
        
        print(f"\n✓ Successfully trained {len(self.trained_models)} models")
        
        return self.trained_models
    
    def cross_validate_models(self, cv=5):
        """
        Perform cross-validation on all models
        
        Parameters:
        -----------
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with cross-validation scores
        """
        print(f"\n{'='*80}")
        print(f"CROSS-VALIDATION ({cv}-Fold)")
        print(f"{'='*80}\n")
        
        if not self.trained_models:
            self.train_all_models()
        
        cv_results = []
        
        for name, model in self.trained_models.items():
            print(f"Cross-validating {name}...", end=' ')
            
            try:
                # Perform cross-validation
                scores = cross_val_score(model, self.X_train, self.y_train, 
                                       cv=cv, scoring='r2', n_jobs=-1)
                
                cv_results.append({
                    'Model': name,
                    'CV_Mean_R2': scores.mean(),
                    'CV_Std_R2': scores.std(),
                    'CV_Min_R2': scores.min(),
                    'CV_Max_R2': scores.max()
                })
                
                print(f"✓ R² = {scores.mean():.4f} (±{scores.std():.4f})")
            except Exception as e:
                print(f"✗ Failed: {str(e)}")
        
        cv_df = pd.DataFrame(cv_results)
        cv_df = cv_df.sort_values('CV_Mean_R2', ascending=False)
        
        print(f"\n{'-'*80}")
        print("CROSS-VALIDATION RESULTS")
        print(f"{'-'*80}")
        print(cv_df.to_string(index=False))
        
        return cv_df
    
    def tune_random_forest(self, cv=3):
        """
        Hyperparameter tuning for Random Forest
        
        Parameters:
        -----------
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        RandomForestRegressor
            Best model after tuning
        """
        print(f"\n{'='*80}")
        print("HYPERPARAMETER TUNING: Random Forest")
        print(f"{'='*80}\n")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt']
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        print("Performing GridSearchCV...")
        grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='r2', 
                                   n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\n✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV R² score: {grid_search.best_score_:.4f}")
        
        self.best_models['Random Forest'] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def tune_gradient_boosting(self, cv=3):
        """
        Hyperparameter tuning for Gradient Boosting
        
        Parameters:
        -----------
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        GradientBoostingRegressor
            Best model after tuning
        """
        print(f"\n{'='*80}")
        print("HYPERPARAMETER TUNING: Gradient Boosting")
        print(f"{'='*80}\n")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        gb = GradientBoostingRegressor(random_state=42)
        
        print("Performing GridSearchCV...")
        grid_search = GridSearchCV(gb, param_grid, cv=cv, scoring='r2', 
                                   n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\n✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV R² score: {grid_search.best_score_:.4f}")
        
        self.best_models['Gradient Boosting'] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def tune_xgboost(self, cv=3):
        """
        Hyperparameter tuning for XGBoost
        
        Parameters:
        -----------
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        XGBRegressor
            Best model after tuning
        """
        print(f"\n{'='*80}")
        print("HYPERPARAMETER TUNING: XGBoost")
        print(f"{'='*80}\n")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb = XGBRegressor(random_state=42, n_jobs=-1)
        
        print("Performing RandomizedSearchCV...")
        random_search = RandomizedSearchCV(xgb, param_grid, n_iter=20, cv=cv, 
                                          scoring='r2', n_jobs=-1, verbose=1, 
                                          random_state=42)
        random_search.fit(self.X_train, self.y_train)
        
        print(f"\n✓ Best parameters: {random_search.best_params_}")
        print(f"✓ Best CV R² score: {random_search.best_score_:.4f}")
        
        self.best_models['XGBoost'] = random_search.best_estimator_
        
        return random_search.best_estimator_
    
    def tune_svr(self, cv=3):
        """
        Hyperparameter tuning for SVR
        
        Parameters:
        -----------
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        SVR
            Best model after tuning
        """
        print(f"\n{'='*80}")
        print("HYPERPARAMETER TUNING: SVR")
        print(f"{'='*80}\n")
        
        param_grid = {
            'kernel': ['rbf', 'linear', 'poly'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'epsilon': [0.01, 0.1, 0.2]
        }
        
        svr = SVR()
        
        print("Performing RandomizedSearchCV...")
        random_search = RandomizedSearchCV(svr, param_grid, n_iter=20, cv=cv, 
                                          scoring='r2', n_jobs=-1, verbose=1, 
                                          random_state=42)
        random_search.fit(self.X_train, self.y_train)
        
        print(f"\n✓ Best parameters: {random_search.best_params_}")
        print(f"✓ Best CV R² score: {random_search.best_score_:.4f}")
        
        self.best_models['SVR'] = random_search.best_estimator_
        
        return random_search.best_estimator_
    
    def tune_knn(self, cv=3):
        """
        Hyperparameter tuning for KNN
        
        Parameters:
        -----------
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        KNeighborsRegressor
            Best model after tuning
        """
        print(f"\n{'='*80}")
        print("HYPERPARAMETER TUNING: KNN")
        print(f"{'='*80}\n")
        
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        
        knn = KNeighborsRegressor()
        
        print("Performing GridSearchCV...")
        grid_search = GridSearchCV(knn, param_grid, cv=cv, scoring='r2', 
                                   n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\n✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV R² score: {grid_search.best_score_:.4f}")
        
        self.best_models['KNN'] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def tune_all_models(self):
        """
        Perform hyperparameter tuning on selected models
        """
        print(f"\n{'='*80}")
        print("HYPERPARAMETER TUNING - ALL MODELS")
        print(f"{'='*80}\n")
        
        # Tune ensemble models (most important)
        self.tune_random_forest(cv=3)
        self.tune_gradient_boosting(cv=3)
        self.tune_xgboost(cv=3)
        
        # Tune other models
        self.tune_svr(cv=3)
        self.tune_knn(cv=3)
        
        print(f"\n✓ Hyperparameter tuning complete for {len(self.best_models)} models")
        
        return self.best_models
    
    def save_models(self, output_dir='models'):
        """
        Save all trained models
        
        Parameters:
        -----------
        output_dir : str
            Directory to save models
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("SAVING MODELS")
        print(f"{'='*80}\n")
        
        # Save base models
        for name, model in self.trained_models.items():
            filename = name.lower().replace(' ', '_') + '.pkl'
            filepath = os.path.join(output_dir, filename)
            joblib.dump(model, filepath)
            print(f"✓ Saved {name} to {filepath}")
        
        # Save best models (tuned)
        for name, model in self.best_models.items():
            filename = name.lower().replace(' ', '_') + '_best.pkl'
            filepath = os.path.join(output_dir, filename)
            joblib.dump(model, filepath)
            print(f"✓ Saved tuned {name} to {filepath}")
        
        print(f"\n✓ All models saved to {output_dir}/")
    
    def load_model(self, model_path):
        """
        Load a saved model
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
            
        Returns:
        --------
        model
            Loaded model
        """
        return joblib.load(model_path)


def prepare_data_for_training(df, target_col='charges', test_size=0.2, random_state=42):
    """
    Prepare data for model training
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    test_size : float
        Proportion of test set
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test
    """
    print(f"\n{'='*80}")
    print("PREPARING DATA FOR TRAINING")
    print(f"{'='*80}\n")
    
    # Separate features and target
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Remove non-numeric columns
    X = X.select_dtypes(include=['int64', 'float64'])
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeatures used ({X.shape[1]}):")
    for col in X.columns:
        print(f"  - {col}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"\nTrain set size: {X_train.shape[0]} ({(1-test_size)*100:.0f}%)")
    print(f"Test set size: {X_test.shape[0]} ({test_size*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test


def main():
    """
    Main function to run model training
    """
    import pandas as pd
    
    # Load engineered data
    if os.path.exists('data/processed/engineered_data.csv'):
        df = pd.read_csv('data/processed/engineered_data.csv')
    else:
        print("Error: Engineered data not found. Run feature_engineering.py first.")
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data_for_training(df)
    
    # Initialize trainer
    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    
    # Train all models
    trainer.train_all_models()
    
    # Cross-validate models
    trainer.cross_validate_models(cv=5)
    
    # Hyperparameter tuning (optional - takes time)
    # trainer.tune_all_models()
    
    # Save models
    trainer.save_models()
    
    print("\n" + "="*80)
    print("MODEL TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
