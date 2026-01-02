"""
Medical Insurance Cost Prediction Package
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .data_loader import DataLoader
from .preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer
from .model_evaluation import ModelEvaluator

__all__ = [
    'DataLoader',
    'DataPreprocessor',
    'FeatureEngineer',
    'ModelTrainer',
    'ModelEvaluator'
]
