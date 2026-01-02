"""
Data Preprocessing Module
Handles missing values, outliers, and data cleaning
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class DataPreprocessor:
    """
    Class to handle data preprocessing tasks
    """
    
    def __init__(self, df):
        """
        Initialize DataPreprocessor
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        """
        self.df = df.copy()
        self.original_shape = df.shape
        self.preprocessing_log = []
        
    def check_missing_values(self):
        """
        Check for missing values in the dataset
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with missing value statistics
        """
        print(f"\n{'='*80}")
        print("MISSING VALUES ANALYSIS")
        print(f"{'='*80}\n")
        
        missing_count = self.df.isnull().sum()
        missing_percent = (missing_count / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': missing_count.values,
            'Missing_Percentage': missing_percent.values
        })
        
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
            'Missing_Count', ascending=False
        )
        
        if len(missing_df) == 0:
            print("✓ No missing values found!")
        else:
            print(missing_df.to_string(index=False))
            
        return missing_df
    
    def handle_missing_values(self, strategy='mean'):
        """
        Handle missing values based on strategy
        
        Parameters:
        -----------
        strategy : str
            Strategy to handle missing values ('mean', 'median', 'mode', 'drop')
        """
        print(f"\n{'='*80}")
        print(f"HANDLING MISSING VALUES - Strategy: {strategy}")
        print(f"{'='*80}\n")
        
        missing_before = self.df.isnull().sum().sum()
        
        if missing_before == 0:
            print("No missing values to handle.")
            return self.df
        
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        if strategy == 'mean':
            for col in numerical_cols:
                if self.df[col].isnull().sum() > 0:
                    mean_val = self.df[col].mean()
                    self.df[col].fillna(mean_val, inplace=True)
                    print(f"✓ Filled {col} with mean: {mean_val:.2f}")
                    
        elif strategy == 'median':
            for col in numerical_cols:
                if self.df[col].isnull().sum() > 0:
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    print(f"✓ Filled {col} with median: {median_val:.2f}")
                    
        elif strategy == 'mode':
            for col in self.df.columns:
                if self.df[col].isnull().sum() > 0:
                    mode_val = self.df[col].mode()[0]
                    self.df[col].fillna(mode_val, inplace=True)
                    print(f"✓ Filled {col} with mode: {mode_val}")
                    
        elif strategy == 'drop':
            rows_before = len(self.df)
            self.df.dropna(inplace=True)
            rows_after = len(self.df)
            print(f"✓ Dropped {rows_before - rows_after} rows with missing values")
        
        # Handle categorical separately with mode
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                mode_val = self.df[col].mode()[0]
                self.df[col].fillna(mode_val, inplace=True)
                print(f"✓ Filled categorical {col} with mode: {mode_val}")
        
        missing_after = self.df.isnull().sum().sum()
        print(f"\nMissing values before: {missing_before}")
        print(f"Missing values after: {missing_after}")
        
        self.preprocessing_log.append(f"Handled missing values using {strategy} strategy")
        
        return self.df
    
    def detect_outliers_iqr(self, columns=None):
        """
        Detect outliers using IQR method
        
        Parameters:
        -----------
        columns : list, optional
            List of columns to check. If None, checks all numerical columns
            
        Returns:
        --------
        dict
            Dictionary with outlier information for each column
        """
        print(f"\n{'='*80}")
        print("OUTLIER DETECTION (IQR Method)")
        print(f"{'='*80}\n")
        
        if columns is None:
            columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        outlier_info = {}
        
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            outlier_percent = (outlier_count / len(self.df)) * 100
            
            outlier_info[col] = {
                'count': outlier_count,
                'percentage': outlier_percent,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            print(f"{col}:")
            print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
            print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"  Outliers: {outlier_count} ({outlier_percent:.2f}%)")
            print()
            
        return outlier_info
    
    def detect_outliers_zscore(self, columns=None, threshold=3):
        """
        Detect outliers using Z-score method
        
        Parameters:
        -----------
        columns : list, optional
            List of columns to check
        threshold : float
            Z-score threshold (default: 3)
            
        Returns:
        --------
        dict
            Dictionary with outlier information
        """
        print(f"\n{'='*80}")
        print(f"OUTLIER DETECTION (Z-Score Method, threshold={threshold})")
        print(f"{'='*80}\n")
        
        if columns is None:
            columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        outlier_info = {}
        
        for col in columns:
            z_scores = np.abs(stats.zscore(self.df[col]))
            outliers = self.df[z_scores > threshold]
            outlier_count = len(outliers)
            outlier_percent = (outlier_count / len(self.df)) * 100
            
            outlier_info[col] = {
                'count': outlier_count,
                'percentage': outlier_percent
            }
            
            print(f"{col}:")
            print(f"  Outliers: {outlier_count} ({outlier_percent:.2f}%)")
            print()
            
        return outlier_info
    
    def handle_outliers(self, method='cap', columns=None):
        """
        Handle outliers in the dataset
        
        Parameters:
        -----------
        method : str
            Method to handle outliers ('cap', 'remove', 'log_transform')
        columns : list, optional
            Columns to process. If None, processes all numerical columns
        """
        print(f"\n{'='*80}")
        print(f"HANDLING OUTLIERS - Method: {method}")
        print(f"{'='*80}\n")
        
        if columns is None:
            columns = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        rows_before = len(self.df)
        
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            if method == 'cap':
                # Cap outliers at bounds
                outliers_before = len(self.df[(self.df[col] < lower_bound) | 
                                             (self.df[col] > upper_bound)])
                
                self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
                self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
                
                print(f"✓ Capped {outliers_before} outliers in {col}")
                
            elif method == 'remove':
                # Remove outlier rows
                outliers_before = len(self.df[(self.df[col] < lower_bound) | 
                                             (self.df[col] > upper_bound)])
                
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                
                print(f"✓ Removed {outliers_before} outliers from {col}")
                
            elif method == 'log_transform':
                # Apply log transformation (only for positive values)
                if (self.df[col] > 0).all():
                    self.df[col] = np.log1p(self.df[col])
                    print(f"✓ Applied log transformation to {col}")
                else:
                    print(f"⚠ Skipped {col} (contains non-positive values)")
        
        rows_after = len(self.df)
        
        if method == 'remove':
            print(f"\nRows before: {rows_before}")
            print(f"Rows after: {rows_after}")
            print(f"Rows removed: {rows_before - rows_after}")
        
        self.preprocessing_log.append(f"Handled outliers using {method} method")
        
        return self.df
    
    def check_duplicates(self):
        """
        Check for duplicate rows
        """
        print(f"\n{'='*80}")
        print("DUPLICATE ROWS CHECK")
        print(f"{'='*80}\n")
        
        duplicates = self.df.duplicated().sum()
        duplicate_percent = (duplicates / len(self.df)) * 100
        
        print(f"Duplicate rows: {duplicates} ({duplicate_percent:.2f}%)")
        
        if duplicates > 0:
            print("\nSample duplicate rows:")
            print(self.df[self.df.duplicated()].head())
            
        return duplicates
    
    def remove_duplicates(self):
        """
        Remove duplicate rows
        """
        print(f"\n{'='*80}")
        print("REMOVING DUPLICATE ROWS")
        print(f"{'='*80}\n")
        
        rows_before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        rows_after = len(self.df)
        
        removed = rows_before - rows_after
        
        print(f"Rows before: {rows_before}")
        print(f"Rows after: {rows_after}")
        print(f"Duplicates removed: {removed}")
        
        if removed > 0:
            self.preprocessing_log.append(f"Removed {removed} duplicate rows")
        
        return self.df
    
    def get_preprocessing_summary(self):
        """
        Get summary of all preprocessing steps
        """
        print(f"\n{'='*80}")
        print("PREPROCESSING SUMMARY")
        print(f"{'='*80}\n")
        
        print(f"Original shape: {self.original_shape}")
        print(f"Current shape: {self.df.shape}")
        print(f"Rows changed: {self.original_shape[0] - self.df.shape[0]}")
        print(f"Columns changed: {self.original_shape[1] - self.df.shape[1]}")
        
        print(f"\nPreprocessing steps performed:")
        for i, step in enumerate(self.preprocessing_log, 1):
            print(f"{i}. {step}")
            
        return self.df


def main():
    """
    Main function to run preprocessing
    """
    from data_loader import DataLoader
    
    # Load data
    data_path = r"C:\Users\mysel\Documents\python_final\Worksheet in Medical Insurance cost prediction.csv"
    loader = DataLoader(data_path)
    df = loader.load_data()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(df)
    
    # Check missing values
    preprocessor.check_missing_values()
    
    # Handle missing values if any
    preprocessor.handle_missing_values(strategy='mean')
    
    # Check duplicates
    preprocessor.check_duplicates()
    preprocessor.remove_duplicates()
    
    # Detect outliers
    preprocessor.detect_outliers_iqr()
    preprocessor.detect_outliers_zscore()
    
    # Handle outliers (cap method is safer than remove)
    # Note: For 'charges' we might want to keep outliers as they're legitimate high costs
    # preprocessor.handle_outliers(method='cap', columns=['age', 'bmi'])
    
    # Get summary
    preprocessor.get_preprocessing_summary()
    
    # Save processed data
    import os
    os.makedirs('data/processed', exist_ok=True)
    preprocessor.df.to_csv('data/processed/cleaned_data.csv', index=False)
    print("\n✓ Cleaned data saved to data/processed/cleaned_data.csv")


if __name__ == "__main__":
    main()
