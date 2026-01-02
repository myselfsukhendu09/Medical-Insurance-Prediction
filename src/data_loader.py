"""
Data Loading Module
Handles loading and initial exploration of the insurance dataset
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


class DataLoader:
    """
    Class to handle data loading and initial exploration
    """
    
    def __init__(self, data_path=None):
        """
        Initialize DataLoader
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the CSV file
        """
        self.data_path = data_path
        self.df = None
        
    def load_data(self, file_path=None):
        """
        Load data from CSV file
        
        Parameters:
        -----------
        file_path : str, optional
            Path to CSV file. If None, uses self.data_path
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataframe
        """
        path = file_path or self.data_path
        
        if path is None:
            raise ValueError("No data path provided")
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
            
        print(f"Loading data from: {path}")
        self.df = pd.read_csv(path)
        print(f"Data loaded successfully! Shape: {self.df.shape}")
        
        return self.df
    
    def get_basic_info(self):
        """
        Display basic information about the dataset
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        print("\n" + "="*80)
        print("DATASET BASIC INFORMATION")
        print("="*80)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Number of Rows: {self.df.shape[0]}")
        print(f"Number of Columns: {self.df.shape[1]}")
        
        print("\n" + "-"*80)
        print("FIRST FEW ROWS")
        print("-"*80)
        print(self.df.head())
        
        print("\n" + "-"*80)
        print("DATASET INFO")
        print("-"*80)
        print(self.df.info())
        
        print("\n" + "-"*80)
        print("STATISTICAL SUMMARY")
        print("-"*80)
        print(self.df.describe())
        
        print("\n" + "-"*80)
        print("DATA TYPES")
        print("-"*80)
        print(self.df.dtypes)
        
        print("\n" + "-"*80)
        print("MISSING VALUES")
        print("-"*80)
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_percent
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        if missing.sum() == 0:
            print("No missing values found!")
            
        print("\n" + "-"*80)
        print("DUPLICATE ROWS")
        print("-"*80)
        duplicates = self.df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicates}")
        
        print("\n" + "-"*80)
        print("UNIQUE VALUES PER COLUMN")
        print("-"*80)
        for col in self.df.columns:
            print(f"{col}: {self.df[col].nunique()} unique values")
            
        return self.df
    
    def get_column_info(self, column_name):
        """
        Get detailed information about a specific column
        
        Parameters:
        -----------
        column_name : str
            Name of the column
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in dataset")
            
        print(f"\n{'='*80}")
        print(f"COLUMN: {column_name}")
        print(f"{'='*80}")
        
        print(f"\nData Type: {self.df[column_name].dtype}")
        print(f"Unique Values: {self.df[column_name].nunique()}")
        print(f"Missing Values: {self.df[column_name].isnull().sum()}")
        
        if self.df[column_name].dtype in ['int64', 'float64']:
            print(f"\nStatistical Summary:")
            print(self.df[column_name].describe())
        else:
            print(f"\nValue Counts:")
            print(self.df[column_name].value_counts())
            
    def save_processed_data(self, output_path):
        """
        Save the dataframe to a CSV file
        
        Parameters:
        -----------
        output_path : str
            Path where to save the CSV file
        """
        if self.df is None:
            raise ValueError("No data to save. Load data first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        print(f"Data saved successfully to: {output_path}")


def main():
    """
    Main function to demonstrate data loading
    """
    # Update this path to your actual data file
    data_path = r"C:\Users\mysel\Documents\python_final\Worksheet in Medical Insurance cost prediction.csv"
    
    # Initialize loader
    loader = DataLoader(data_path)
    
    # Load data
    df = loader.load_data()
    
    # Get basic information
    loader.get_basic_info()
    
    # Get info for specific columns
    print("\n" + "="*80)
    print("DETAILED COLUMN ANALYSIS")
    print("="*80)
    
    for col in df.columns:
        loader.get_column_info(col)
        print("\n")


if __name__ == "__main__":
    main()
