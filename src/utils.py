"""
Utility Functions
Helper functions used across the project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import joblib


def save_dict_to_json(data_dict, filepath):
    """
    Save dictionary to JSON file
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary to save
    filepath : str
        Path to save JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(data_dict, f, indent=4)
    print(f"✓ Saved dictionary to {filepath}")


def load_json_to_dict(filepath):
    """
    Load JSON file to dictionary
    
    Parameters:
    -----------
    filepath : str
        Path to JSON file
        
    Returns:
    --------
    dict
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        data_dict = json.load(f)
    return data_dict


def save_model(model, filepath):
    """
    Save model using joblib
    
    Parameters:
    -----------
    model : sklearn model
        Model to save
    filepath : str
        Path to save model
    """
    joblib.dump(model, filepath)
    print(f"✓ Saved model to {filepath}")


def load_model(filepath):
    """
    Load model using joblib
    
    Parameters:
    -----------
    filepath : str
        Path to model file
        
    Returns:
    --------
    model
        Loaded model
    """
    model = joblib.load(filepath)
    print(f"✓ Loaded model from {filepath}")
    return model


def create_timestamp():
    """
    Create timestamp string
    
    Returns:
    --------
    str
        Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def plot_feature_importance(model, feature_names, top_n=10, figsize=(10, 6)):
    """
    Plot feature importance for tree-based models
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display
    figsize : tuple
        Figure size
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create dataframe
    feature_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feature_imp_df = feature_imp_df.sort_values('Importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.barh(feature_imp_df['Feature'], feature_imp_df['Importance'], color='skyblue')
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return feature_imp_df


def calculate_percentage_change(old_value, new_value):
    """
    Calculate percentage change
    
    Parameters:
    -----------
    old_value : float
        Old value
    new_value : float
        New value
        
    Returns:
    --------
    float
        Percentage change
    """
    if old_value == 0:
        return 0
    return ((new_value - old_value) / old_value) * 100


def format_currency(amount):
    """
    Format number as currency
    
    Parameters:
    -----------
    amount : float
        Amount to format
        
    Returns:
    --------
    str
        Formatted currency string
    """
    return f"${amount:,.2f}"


def print_section_header(title, char='=', width=80):
    """
    Print formatted section header
    
    Parameters:
    -----------
    title : str
        Section title
    char : str
        Character to use for border
    width : int
        Width of header
    """
    print(f"\n{char * width}")
    print(title.center(width))
    print(f"{char * width}\n")


def get_memory_usage(df):
    """
    Get memory usage of dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    str
        Memory usage string
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / (1024 ** 2)
    return f"{memory_mb:.2f} MB"


def reduce_memory_usage(df):
    """
    Reduce memory usage of dataframe by optimizing dtypes
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Optimized dataframe
    """
    start_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
    
    print(f"Memory usage before: {start_mem:.2f} MB")
    print(f"Memory usage after: {end_mem:.2f} MB")
    print(f"Reduced by: {100 * (start_mem - end_mem) / start_mem:.1f}%")
    
    return df


def create_correlation_matrix(df, figsize=(12, 10)):
    """
    Create and plot correlation matrix
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    figsize : tuple
        Figure size
    """
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    
    # Calculate correlation
    corr_matrix = numerical_df.corr()
    
    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
               center=0, square=True, linewidths=1)
    plt.title('Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    return corr_matrix


def get_dataset_summary(df):
    """
    Get comprehensive dataset summary
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Summary statistics
    """
    summary = {
        'shape': df.shape,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'memory_usage': get_memory_usage(df),
        'n_duplicates': df.duplicated().sum(),
        'n_missing': df.isnull().sum().sum(),
        'numerical_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    return summary


def print_dataset_summary(df):
    """
    Print dataset summary
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    summary = get_dataset_summary(df)
    
    print_section_header("DATASET SUMMARY")
    
    print(f"Shape: {summary['shape']}")
    print(f"Rows: {summary['n_rows']:,}")
    print(f"Columns: {summary['n_columns']}")
    print(f"Memory Usage: {summary['memory_usage']}")
    print(f"Duplicates: {summary['n_duplicates']:,}")
    print(f"Missing Values: {summary['n_missing']:,}")
    print(f"\nNumerical Columns ({len(summary['numerical_columns'])}):")
    for col in summary['numerical_columns']:
        print(f"  - {col}")
    print(f"\nCategorical Columns ({len(summary['categorical_columns'])}):")
    for col in summary['categorical_columns']:
        print(f"  - {col}")


if __name__ == "__main__":
    print("Utility functions module")
    print("Import this module to use utility functions")
