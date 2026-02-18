"""
Feature Engineering Module
Creates Kenya-specific features to enhance churn prediction model
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class KenyanFeatureEngineer:
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.counties = self.config['features']['kenyan_counties']
        self.rural_counties = self.config['features']['rural_counties']
        logger.info("Feature engineer initialized")
    
    def add_kenyan_features(self, df):
        logger.info(f"Engineering features for {len(df):,} customers...")
        
        df_enhanced = df.copy()
        np.random.seed(42)
        
        df_enhanced = self.county_features(df_enhanced)
        df_enhanced = self.mpesa_features(df_enhanced)
        df_enhanced = self.bonga_features(df_enhanced)
        df_enhanced = self.safaricom_home_features(df_enhanced)
        df_enhanced = self.competitor_features(df_enhanced)
        df_enhanced = self.network_quality_features(df_enhanced)
        df_enhanced = self.data_bundle_features(df_enhanced)
        
        logger.info(f"Added {df_enhanced.shape[1] - df.shape[1]} new features")
        return df_enhanced
    
    def county_features(self, df):
        county_weights = [0.25, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05, 0.08, 0.06, 0.06]
        df['county'] = np.random.choice(self.counties, size=len(df), p=county_weights)
        df['is_rural'] = df['county'].isin(self.rural_counties).astype(int)
        df['location_type'] = df['is_rural'].map({0: 'urban', 1: 'rural'})
        return df
    
    def mpesa_features(self, df):

        df['mpesa_usage_score'] = np.random.randint(0, 100, len(df)).astype(float)
    
        if 'PaymentMethod' in df.columns:
            electronic_mask = df['PaymentMethod'].str.contains('Electronic|Bank', na=False)
            df.loc[electronic_mask, 'mpesa_usage_score'] = (
                df.loc[electronic_mask, 'mpesa_usage_score'] * 1.3
            ).clip(0, 100)
    
        if 'tenure' in df.columns:
            df['mpesa_usage_score'] = (
                df['mpesa_usage_score'] + (df['tenure'] * 0.5)
            ).clip(0, 100)
    
        df['mpesa_engagement'] = pd.cut(
            df['mpesa_usage_score'],
            bins=[0, 30, 60, 100],
            labels=['low', 'medium', 'high']
        )
    
        df['mpesa_monthly_transactions'] = np.random.poisson(
            lam=df['mpesa_usage_score'] / 10, 
            size=len(df)
        )
        return df
    
    def bonga_features(self, df):
        if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
            base_points = df['tenure'] * 50
            spending_bonus = df['MonthlyCharges'] * 2
            noise = np.random.randint(-500, 500, len(df))
            df['bonga_points'] = (base_points + spending_bonus + noise).clip(0, 15000)
        else:
            df['bonga_points'] = np.random.randint(0, 10000, len(df))
        
        df['days_since_bonga_redemption'] = np.random.randint(0, 365, len(df))
        
        low_points_mask = df['bonga_points'] < 500
        df.loc[low_points_mask, 'days_since_bonga_redemption'] = np.random.randint(
            180, 365, low_points_mask.sum()
        )
        
        df['bonga_active'] = (df['days_since_bonga_redemption'] < 90).astype(int)
        return df
    
    def safaricom_home_features(self, df):
        df['has_safaricom_home'] = 0
        
        if 'InternetService' in df.columns:
            fiber_mask = df['InternetService'] == 'Fiber optic'
            df.loc[fiber_mask, 'has_safaricom_home'] = np.random.choice(
                [0, 1], size=fiber_mask.sum(), p=[0.3, 0.7]
            )
        
        urban_mask = df['is_rural'] == 0
        df.loc[urban_mask & (df['has_safaricom_home'] == 0), 'has_safaricom_home'] = np.random.choice(
            [0, 1], size=((urban_mask) & (df['has_safaricom_home'] == 0)).sum(), p=[0.85, 0.15]
        )
        return df
    
    def competitor_features(self, df):
        df['competitor_exposure'] = np.random.randint(1, 6, len(df))
        
        if 'Contract' in df.columns:
            monthly_mask = df['Contract'] == 'Month-to-month'
            df.loc[monthly_mask, 'competitor_exposure'] = (
                df.loc[monthly_mask, 'competitor_exposure'] + 1
            ).clip(1, 5)
        
        urban_mask = df['is_rural'] == 0
        df.loc[urban_mask, 'competitor_exposure'] = (
            df.loc[urban_mask, 'competitor_exposure'] + 1
        ).clip(1, 5)
        
        df['high_competitor_risk'] = (df['competitor_exposure'] >= 4).astype(int)
        return df
    
    def network_quality_features(self, df):
        df['network_quality_score'] = np.random.randint(5, 11, len(df))
        
        rural_mask = df['is_rural'] == 1
        df.loc[rural_mask, 'network_quality_score'] = np.random.randint(3, 8, rural_mask.sum())
        
        df['network_satisfaction'] = pd.cut(
            df['network_quality_score'],
            bins=[0, 5, 7, 10],
            labels=['dissatisfied', 'neutral', 'satisfied']
        )
        
        if 'TechSupport' in df.columns:
            no_support_mask = df['TechSupport'] == 'No'
            df.loc[no_support_mask, 'network_quality_score'] = (
                df.loc[no_support_mask, 'network_quality_score'] - 1
            ).clip(1, 10)
        return df
    
    def data_bundle_features(self, df):
        df['uses_data_rollover'] = np.random.choice([0, 1], len(df), p=[0.6, 0.4])
        
        if 'InternetService' in df.columns:
            has_internet = df['InternetService'] != 'No'
            df.loc[has_internet, 'uses_data_rollover'] = np.random.choice(
                [0, 1], size=has_internet.sum(), p=[0.3, 0.7]
            )
        
        if 'MonthlyCharges' in df.columns:
            df['data_bundle_tier'] = pd.cut(
                df['MonthlyCharges'],
                bins=[0, 30, 60, 100, 200],
                labels=['basic', 'standard', 'premium', 'unlimited']
            )
        
        df['avg_monthly_data_gb'] = np.random.lognormal(2.5, 1.2, len(df)).clip(0.5, 100)
        return df
    
    def create_interaction_features(self, df):
        df_interact = df.copy()
        
        df_interact['digital_loyalty_score'] = (
            df_interact['mpesa_usage_score'] * 0.6 + 
            (df_interact['bonga_points'] / 150) * 0.4
        )
        
        df_interact['rural_network_risk'] = (
            (df_interact['is_rural'] == 1) & 
            (df_interact['network_quality_score'] < 6)
        ).astype(int)
        
        if 'MonthlyCharges' in df_interact.columns:
            df_interact['price_sensitive_risk'] = (
                (df_interact['MonthlyCharges'] > df_interact['MonthlyCharges'].median()) &
                (df_interact['competitor_exposure'] >= 4)
            ).astype(int)
        
        engagement_components = []
        if 'has_safaricom_home' in df_interact.columns:
            engagement_components.append(df_interact['has_safaricom_home'] * 25)
        
        engagement_components.append((df_interact['bonga_active']) * 20)
        engagement_components.append(df_interact['mpesa_usage_score'] * 0.3)
        
        if 'uses_data_rollover' in df_interact.columns:
            engagement_components.append(df_interact['uses_data_rollover'] * 15)
        
        df_interact['customer_engagement_score'] = sum(engagement_components)
        return df_interact
    
    def save_enhanced_data(self, df, output_path=None):
        if output_path is None:
            output_path = self.config['data']['processed_data_path']
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Enhanced data saved: {output_path}")


def engineer_kenyan_features(df, config_path='config/config.yaml'):
    engineer = KenyanFeatureEngineer(config_path)
    df_enhanced = engineer.add_kenyan_features(df)
    df_final = engineer.create_interaction_features(df_enhanced)
    return df_final


if __name__ == "__main__":
    from data_processing import load_and_clean_data
    
    df = load_and_clean_data()
    engineer = KenyanFeatureEngineer()
    df_enhanced = engineer.add_kenyan_features(df)
    df_final = engineer.create_interaction_features(df_enhanced)
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"Original features: {df.shape[1]}")
    print(f"Final features: {df_final.shape[1]}")
    print(f"New features: {df_final.shape[1] - df.shape[1]}")
    
    kenyan_cols = ['county', 'mpesa_usage_score', 'bonga_points', 
                   'has_safaricom_home', 'network_quality_score']
    print(f"\nSample Kenyan features:")
    print(df_final[kenyan_cols].head())
    
    engineer.save_enhanced_data(df_final)