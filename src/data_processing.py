"""
Data Processing Module
Handles loading, cleaning, and preprocessing of telco customer data
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DataProcessor:
    """Class to handle all data processing operations"""
    
    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize DataProcessor with configuration
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info("DataProcessor initialized")
    
    def load_raw_data(self):
        """
        Load raw telco customer churn dataset
        
        Returns:
            pd.DataFrame: Raw customer data
        """
        data_path = self.config['data']['raw_data_path']
        
        try:
            df = pd.read_csv(data_path)
            logger.info(f"Successfully loaded data from {data_path}")
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            return df
        
        except FileNotFoundError:
            logger.error(f"File not found: {data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_data(self, df):
        """
        Clean the dataset: handle missing values, fix data types, remove duplicates
        
        Args:
            df (pd.DataFrame): Raw dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info("Starting data cleaning...")
        
        df_clean = df.copy()
        

        initial_rows = len(df_clean)
        

        missing_summary = df_clean.isnull().sum()
        if missing_summary.sum() > 0:
            logger.info(f"Missing values found:\n{missing_summary[missing_summary > 0]}")
        
        df_clean = df_clean.dropna()
        logger.info(f"Dropped {initial_rows - len(df_clean)} rows with missing values")
        

        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
        

        if 'TotalCharges' in df_clean.columns:
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

            df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)
        

        df_clean.columns = df_clean.columns.str.strip()
        

        if 'Churn' in df_clean.columns:
            df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
            logger.info(f"Churn distribution:\n{df_clean['Churn'].value_counts(normalize=True)}")
        
        logger.info(f"Data cleaning complete. Final shape: {df_clean.shape}")
        
        return df_clean
    
    def get_data_summary(self, df):
        """
        Generate summary statistics for the dataset
        
        Args:
            df (pd.DataFrame): Dataframe to summarize
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_customers': len(df),
            'churned_customers': df['Churn'].sum() if 'Churn' in df.columns else 0,
            'churn_rate': df['Churn'].mean() if 'Churn' in df.columns else 0,
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object','str']).columns.tolist(),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        return summary
    
    def save_processed_data(self, df, output_path=None):
        """
        Save processed dataframe to CSV
        
        Args:
            df (pd.DataFrame): Processed dataframe
            output_path (str, optional): Custom output path
        """
        if output_path is None:
            output_path = self.config['data']['processed_data_path']
        
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")



def load_and_clean_data(config_path='config/config.yaml'):
    """
    Load raw data and perform cleaning in one step
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    processor = DataProcessor(config_path)
    df = processor.load_raw_data()
    df_clean = processor.clean_data(df)
    return df_clean


def get_feature_types(df):
    """
    Categorize features into numeric and categorical
    
    Args:
        df (pd.DataFrame): Dataframe to analyze
        
    Returns:
        dict: Dictionary with 'numeric' and 'categorical' feature lists
    """
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object','str']).columns.tolist()
    
    # Remove target variable if present
    if 'Churn' in numeric_features:
        numeric_features.remove('Churn')
    if 'customerID' in categorical_features:
        categorical_features.remove('customerID')
    
    return {
        'numeric': numeric_features,
        'categorical': categorical_features
    }



if __name__ == "__main__":
    
    processor = DataProcessor()
    
    df = processor.load_raw_data()
    print(f"\nRaw data loaded: {df.shape}")
    
    df_clean = processor.clean_data(df)
    print(f"\nCleaned data: {df_clean.shape}")
    
    summary = processor.get_data_summary(df_clean)
    print(f"\nData Summary:")
    print(f"Total customers: {summary['total_customers']}")
    print(f"Churned customers: {summary['churned_customers']}")
    print(f"Churn rate: {summary['churn_rate']:.2%}")
    
    feature_types = get_feature_types(df_clean)
    print(f"\nNumeric features: {feature_types['numeric']}")
    print(f"\nCategorical features: {feature_types['categorical']}")