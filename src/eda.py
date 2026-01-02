"""
Exploratory Data Analysis Module
Comprehensive EDA with visualizations and statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class EDAAnalyzer:
    """
    Class to perform comprehensive Exploratory Data Analysis
    """
    
    def __init__(self, df):
        """
        Initialize EDA Analyzer
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        """
        self.df = df.copy()
        self.numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        
    def plot_target_distribution(self, target_col='charges'):
        """
        Plot distribution of target variable
        
        Parameters:
        -----------
        target_col : str
            Name of target column
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Histogram
        axes[0].hist(self.df[target_col], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0].set_title(f'Distribution of {target_col}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel(target_col, fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(self.df[target_col], vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7))
        axes[1].set_title(f'Box Plot of {target_col}', fontsize=14, fontweight='bold')
        axes[1].set_ylabel(target_col, fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(self.df[target_col], dist="norm", plot=axes[2])
        axes[2].set_title(f'Q-Q Plot of {target_col}', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/figures/target_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        print(f"\n{'='*80}")
        print(f"TARGET VARIABLE STATISTICS: {target_col}")
        print(f"{'='*80}")
        print(f"Mean: ${self.df[target_col].mean():,.2f}")
        print(f"Median: ${self.df[target_col].median():,.2f}")
        print(f"Std Dev: ${self.df[target_col].std():,.2f}")
        print(f"Min: ${self.df[target_col].min():,.2f}")
        print(f"Max: ${self.df[target_col].max():,.2f}")
        print(f"Skewness: {self.df[target_col].skew():.4f}")
        print(f"Kurtosis: {self.df[target_col].kurtosis():.4f}")
        
    def plot_numerical_distributions(self):
        """
        Plot distributions of all numerical features
        """
        num_cols = [col for col in self.numerical_cols if col != 'charges']
        n_cols = len(num_cols)
        
        fig, axes = plt.subplots(n_cols, 2, figsize=(15, 5*n_cols))
        
        for idx, col in enumerate(num_cols):
            # Histogram
            axes[idx, 0].hist(self.df[col], bins=30, edgecolor='black', 
                            alpha=0.7, color='coral')
            axes[idx, 0].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            axes[idx, 0].set_xlabel(col, fontsize=10)
            axes[idx, 0].set_ylabel('Frequency', fontsize=10)
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Box plot
            axes[idx, 1].boxplot(self.df[col], vert=False, patch_artist=True,
                               boxprops=dict(facecolor='lightblue', alpha=0.7))
            axes[idx, 1].set_title(f'Box Plot of {col}', fontsize=12, fontweight='bold')
            axes[idx, 1].set_xlabel(col, fontsize=10)
            axes[idx, 1].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig('results/figures/numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_categorical_distributions(self):
        """
        Plot distributions of categorical features
        """
        if not self.categorical_cols:
            print("No categorical columns found.")
            return
            
        n_cols = len(self.categorical_cols)
        fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 5))
        
        if n_cols == 1:
            axes = [axes]
            
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for idx, col in enumerate(self.categorical_cols):
            value_counts = self.df[col].value_counts()
            axes[idx].bar(value_counts.index, value_counts.values, 
                         color=colors[idx % len(colors)], alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(col, fontsize=10)
            axes[idx].set_ylabel('Count', fontsize=10)
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, v in enumerate(value_counts.values):
                axes[idx].text(i, v + max(value_counts.values)*0.01, str(v), 
                             ha='center', va='bottom', fontweight='bold')
                
        plt.tight_layout()
        plt.savefig('results/figures/categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_correlation_heatmap(self):
        """
        Plot correlation heatmap for numerical features
        """
        # Create a copy with encoded categorical variables for correlation
        df_encoded = self.df.copy()
        
        for col in self.categorical_cols:
            df_encoded[col] = pd.Categorical(df_encoded[col]).codes
            
        correlation_matrix = df_encoded.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('results/figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top correlations with target
        if 'charges' in correlation_matrix.columns:
            print(f"\n{'='*80}")
            print("CORRELATIONS WITH TARGET (charges)")
            print(f"{'='*80}")
            correlations = correlation_matrix['charges'].sort_values(ascending=False)
            print(correlations)
            
    def plot_feature_vs_target(self, target_col='charges'):
        """
        Plot each feature against target variable
        
        Parameters:
        -----------
        target_col : str
            Name of target column
        """
        # Numerical features vs target
        num_cols = [col for col in self.numerical_cols if col != target_col]
        
        if num_cols:
            fig, axes = plt.subplots(1, len(num_cols), figsize=(6*len(num_cols), 5))
            if len(num_cols) == 1:
                axes = [axes]
                
            for idx, col in enumerate(num_cols):
                axes[idx].scatter(self.df[col], self.df[target_col], alpha=0.5, color='purple')
                axes[idx].set_xlabel(col, fontsize=12)
                axes[idx].set_ylabel(target_col, fontsize=12)
                axes[idx].set_title(f'{col} vs {target_col}', fontsize=12, fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(self.df[col], self.df[target_col], 1)
                p = np.poly1d(z)
                axes[idx].plot(self.df[col], p(self.df[col]), "r--", alpha=0.8, linewidth=2)
                
            plt.tight_layout()
            plt.savefig('results/figures/numerical_vs_target.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Categorical features vs target
        if self.categorical_cols:
            fig, axes = plt.subplots(1, len(self.categorical_cols), 
                                   figsize=(6*len(self.categorical_cols), 5))
            if len(self.categorical_cols) == 1:
                axes = [axes]
                
            for idx, col in enumerate(self.categorical_cols):
                self.df.boxplot(column=target_col, by=col, ax=axes[idx], 
                              patch_artist=True, grid=True)
                axes[idx].set_xlabel(col, fontsize=12)
                axes[idx].set_ylabel(target_col, fontsize=12)
                axes[idx].set_title(f'{target_col} by {col}', fontsize=12, fontweight='bold')
                plt.sca(axes[idx])
                plt.xticks(rotation=45)
                
            plt.tight_layout()
            plt.savefig('results/figures/categorical_vs_target.png', dpi=300, bbox_inches='tight')
            plt.show()
            
    def analyze_outliers(self):
        """
        Detect and visualize outliers in numerical features
        """
        print(f"\n{'='*80}")
        print("OUTLIER ANALYSIS")
        print(f"{'='*80}\n")
        
        for col in self.numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            
            print(f"{col}:")
            print(f"  Lower Bound: {lower_bound:.2f}")
            print(f"  Upper Bound: {upper_bound:.2f}")
            print(f"  Number of Outliers: {len(outliers)} ({len(outliers)/len(self.df)*100:.2f}%)")
            print()
            
    def generate_eda_report(self):
        """
        Generate comprehensive EDA report
        """
        print(f"\n{'='*80}")
        print("COMPREHENSIVE EDA REPORT")
        print(f"{'='*80}\n")
        
        # Dataset overview
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Numerical Features: {len(self.numerical_cols)}")
        print(f"Categorical Features: {len(self.categorical_cols)}")
        print()
        
        # Generate all visualizations
        print("Generating visualizations...")
        
        self.plot_target_distribution()
        self.plot_numerical_distributions()
        self.plot_categorical_distributions()
        self.plot_correlation_heatmap()
        self.plot_feature_vs_target()
        self.analyze_outliers()
        
        print("\n" + "="*80)
        print("EDA COMPLETE! All visualizations saved to results/figures/")
        print("="*80)


def main():
    """
    Main function to run EDA
    """
    # Load data
    from data_loader import DataLoader
    
    data_path = r"C:\Users\mysel\Documents\python_final\Worksheet in Medical Insurance cost prediction.csv"
    loader = DataLoader(data_path)
    df = loader.load_data()
    
    # Create results directory
    import os
    os.makedirs('results/figures', exist_ok=True)
    
    # Perform EDA
    eda = EDAAnalyzer(df)
    eda.generate_eda_report()


if __name__ == "__main__":
    main()
