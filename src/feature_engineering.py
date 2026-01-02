"""
Feature Engineering Module
Handles encoding, scaling, and feature transformations
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import PowerTransformer
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


class FeatureEngineer:
    """
    Class to handle feature engineering tasks
    """
    
    def __init__(self, df):
        """
        Initialize FeatureEngineer
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        """
        self.df = df.copy()
        self.encoders = {}
        self.scalers = {}
        self.feature_log = []
        
    def encode_categorical_label(self, columns=None):
        """
        Encode categorical variables using Label Encoding
        
        Parameters:
        -----------
        columns : list, optional
            List of columns to encode. If None, encodes all object columns
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with encoded columns
        """
        print(f"\n{'='*80}")
        print("LABEL ENCODING CATEGORICAL VARIABLES")
        print(f"{'='*80}\n")
        
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col])
                self.encoders[col] = le
                
                print(f"✓ Encoded {col}:")
                print(f"  Original values: {self.df[col].unique()}")
                print(f"  Encoded values: {self.df[f'{col}_encoded'].unique()}")
                print(f"  Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
                print()
                
        self.feature_log.append(f"Label encoded {len(columns)} categorical columns")
        
        return self.df
    
    def encode_categorical_onehot(self, columns=None, drop_first=True):
        """
        Encode categorical variables using One-Hot Encoding
        
        Parameters:
        -----------
        columns : list, optional
            List of columns to encode
        drop_first : bool
            Whether to drop first category to avoid multicollinearity
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with one-hot encoded columns
        """
        print(f"\n{'='*80}")
        print("ONE-HOT ENCODING CATEGORICAL VARIABLES")
        print(f"{'='*80}\n")
        
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col in self.df.columns:
                # Create dummy variables
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=drop_first)
                
                # Add to dataframe
                self.df = pd.concat([self.df, dummies], axis=1)
                
                print(f"✓ One-hot encoded {col}:")
                print(f"  Created columns: {list(dummies.columns)}")
                print()
                
        self.feature_log.append(f"One-hot encoded {len(columns)} categorical columns")
        
        return self.df
    
    def check_skewness(self, threshold=0.5):
        """
        Check skewness of numerical features
        
        Parameters:
        -----------
        threshold : float
            Threshold for considering a feature as skewed
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with skewness information
        """
        print(f"\n{'='*80}")
        print(f"SKEWNESS ANALYSIS (threshold={threshold})")
        print(f"{'='*80}\n")
        
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        skewness_data = []
        
        for col in numerical_cols:
            skew_value = self.df[col].skew()
            kurt_value = self.df[col].kurtosis()
            is_skewed = abs(skew_value) > threshold
            
            skewness_data.append({
                'Feature': col,
                'Skewness': skew_value,
                'Kurtosis': kurt_value,
                'Is_Skewed': is_skewed
            })
            
        skewness_df = pd.DataFrame(skewness_data)
        skewness_df = skewness_df.sort_values('Skewness', key=abs, ascending=False)
        
        print(skewness_df.to_string(index=False))
        
        print(f"\nSkewed features (|skewness| > {threshold}):")
        skewed_features = skewness_df[skewness_df['Is_Skewed']]['Feature'].tolist()
        print(skewed_features)
        
        return skewness_df
    
    def handle_skewness(self, columns=None, method='log'):
        """
        Handle skewness in numerical features
        
        Parameters:
        -----------
        columns : list, optional
            Columns to transform. If None, transforms all skewed columns
        method : str
            Transformation method ('log', 'sqrt', 'boxcox', 'yeojohnson')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with transformed features
        """
        print(f"\n{'='*80}")
        print(f"HANDLING SKEWNESS - Method: {method}")
        print(f"{'='*80}\n")
        
        if columns is None:
            # Auto-detect skewed columns
            skewness_df = self.check_skewness(threshold=0.5)
            columns = skewness_df[skewness_df['Is_Skewed']]['Feature'].tolist()
        
        for col in columns:
            if col not in self.df.columns:
                continue
                
            original_skew = self.df[col].skew()
            
            if method == 'log':
                # Log transformation (only for positive values)
                if (self.df[col] > 0).all():
                    self.df[f'{col}_log'] = np.log1p(self.df[col])
                    new_skew = self.df[f'{col}_log'].skew()
                    print(f"✓ Log transformed {col}: skewness {original_skew:.3f} → {new_skew:.3f}")
                else:
                    print(f"⚠ Skipped {col} (contains non-positive values)")
                    
            elif method == 'sqrt':
                # Square root transformation
                if (self.df[col] >= 0).all():
                    self.df[f'{col}_sqrt'] = np.sqrt(self.df[col])
                    new_skew = self.df[f'{col}_sqrt'].skew()
                    print(f"✓ Sqrt transformed {col}: skewness {original_skew:.3f} → {new_skew:.3f}")
                else:
                    print(f"⚠ Skipped {col} (contains negative values)")
                    
            elif method == 'boxcox':
                # Box-Cox transformation (only for positive values)
                if (self.df[col] > 0).all():
                    self.df[f'{col}_boxcox'], _ = stats.boxcox(self.df[col])
                    new_skew = self.df[f'{col}_boxcox'].skew()
                    print(f"✓ Box-Cox transformed {col}: skewness {original_skew:.3f} → {new_skew:.3f}")
                else:
                    print(f"⚠ Skipped {col} (contains non-positive values)")
                    
            elif method == 'yeojohnson':
                # Yeo-Johnson transformation (works with negative values)
                pt = PowerTransformer(method='yeo-johnson')
                self.df[f'{col}_yj'] = pt.fit_transform(self.df[[col]])
                new_skew = self.df[f'{col}_yj'].skew()
                print(f"✓ Yeo-Johnson transformed {col}: skewness {original_skew:.3f} → {new_skew:.3f}")
        
        self.feature_log.append(f"Applied {method} transformation to {len(columns)} columns")
        
        return self.df
    
    def scale_features(self, columns=None, method='standard'):
        """
        Scale numerical features
        
        Parameters:
        -----------
        columns : list, optional
            Columns to scale. If None, scales all numerical columns
        method : str
            Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with scaled features
        """
        print(f"\n{'='*80}")
        print(f"FEATURE SCALING - Method: {method}")
        print(f"{'='*80}\n")
        
        if columns is None:
            columns = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            # Remove target variable if present
            if 'charges' in columns:
                columns.remove('charges')
        
        if method == 'standard':
            scaler = StandardScaler()
            scaler_name = 'StandardScaler'
        elif method == 'minmax':
            scaler = MinMaxScaler()
            scaler_name = 'MinMaxScaler'
        elif method == 'robust':
            scaler = RobustScaler()
            scaler_name = 'RobustScaler'
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform
        scaled_data = scaler.fit_transform(self.df[columns])
        
        # Create new column names
        scaled_columns = [f'{col}_scaled' for col in columns]
        
        # Add scaled columns to dataframe
        scaled_df = pd.DataFrame(scaled_data, columns=scaled_columns, index=self.df.index)
        self.df = pd.concat([self.df, scaled_df], axis=1)
        
        # Store scaler
        self.scalers[method] = scaler
        
        print(f"✓ Scaled {len(columns)} features using {scaler_name}")
        print(f"  Columns: {columns}")
        print(f"  New columns: {scaled_columns}")
        
        self.feature_log.append(f"Scaled {len(columns)} features using {method}")
        
        return self.df
    
    def create_interaction_features(self):
        """
        Create interaction features
        """
        print(f"\n{'='*80}")
        print("CREATING INTERACTION FEATURES")
        print(f"{'='*80}\n")
        
        # BMI * Smoker interaction (important for insurance)
        if 'bmi' in self.df.columns and 'smoker' in self.df.columns:
            # Create binary smoker column if not exists
            if self.df['smoker'].dtype == 'object':
                self.df['smoker_binary'] = (self.df['smoker'] == 'yes').astype(int)
            else:
                self.df['smoker_binary'] = self.df['smoker']
                
            self.df['bmi_smoker'] = self.df['bmi'] * self.df['smoker_binary']
            print("✓ Created bmi_smoker interaction")
        
        # Age * BMI interaction
        if 'age' in self.df.columns and 'bmi' in self.df.columns:
            self.df['age_bmi'] = self.df['age'] * self.df['bmi']
            print("✓ Created age_bmi interaction")
        
        # Age groups
        if 'age' in self.df.columns:
            self.df['age_group'] = pd.cut(self.df['age'], 
                                         bins=[0, 25, 40, 60, 100],
                                         labels=['young', 'middle', 'senior', 'elderly'])
            print("✓ Created age_group feature")
        
        # BMI categories
        if 'bmi' in self.df.columns:
            self.df['bmi_category'] = pd.cut(self.df['bmi'],
                                            bins=[0, 18.5, 25, 30, 100],
                                            labels=['underweight', 'normal', 'overweight', 'obese'])
            print("✓ Created bmi_category feature")
        
        self.feature_log.append("Created interaction and derived features")
        
        return self.df
    
    def save_encoders_scalers(self, output_dir='models'):
        """
        Save encoders and scalers for future use
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the objects
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save encoders
        if self.encoders:
            joblib.dump(self.encoders, os.path.join(output_dir, 'encoders.pkl'))
            print(f"✓ Saved encoders to {output_dir}/encoders.pkl")
        
        # Save scalers
        if self.scalers:
            joblib.dump(self.scalers, os.path.join(output_dir, 'scalers.pkl'))
            print(f"✓ Saved scalers to {output_dir}/scalers.pkl")
    
    def get_feature_engineering_summary(self):
        """
        Get summary of feature engineering steps
        """
        print(f"\n{'='*80}")
        print("FEATURE ENGINEERING SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"Final dataset shape: {self.df.shape}")
        print(f"\nFeature engineering steps performed:")
        for i, step in enumerate(self.feature_log, 1):
            print(f"{i}. {step}")
        
        print(f"\nFinal columns ({len(self.df.columns)}):")
        for col in self.df.columns:
            print(f"  - {col}")
            
        return self.df


def main():
    """
    Main function to run feature engineering
    """
    # Load preprocessed data
    import os
    
    if os.path.exists('data/processed/cleaned_data.csv'):
        df = pd.read_csv('data/processed/cleaned_data.csv')
    else:
        from data_loader import DataLoader
        data_path = r"C:\Users\mysel\Documents\python_final\Worksheet in Medical Insurance cost prediction.csv"
        loader = DataLoader(data_path)
        df = loader.load_data()
    
    # Initialize feature engineer
    fe = FeatureEngineer(df)
    
    # Check skewness
    fe.check_skewness()
    
    # Encode categorical variables (using label encoding for simplicity)
    categorical_cols = ['sex', 'smoker', 'region']
    fe.encode_categorical_label(categorical_cols)
    
    # Create interaction features
    fe.create_interaction_features()
    
    # Handle skewness (if needed)
    # fe.handle_skewness(columns=['charges'], method='log')
    
    # Scale features
    numerical_cols = ['age', 'bmi', 'children']
    fe.scale_features(columns=numerical_cols, method='standard')
    
    # Get summary
    fe.get_feature_engineering_summary()
    
    # Save encoders and scalers
    os.makedirs('models', exist_ok=True)
    fe.save_encoders_scalers()
    
    # Save engineered data
    os.makedirs('data/processed', exist_ok=True)
    fe.df.to_csv('data/processed/engineered_data.csv', index=False)
    print("\n✓ Engineered data saved to data/processed/engineered_data.csv")


if __name__ == "__main__":
    main()
