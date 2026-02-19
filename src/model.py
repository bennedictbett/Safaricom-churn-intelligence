"""
Model Training Module
Handles ML model training, evaluation, and prediction for churn
"""

import pandas as pd
import numpy as np
import yaml
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ChurnPredictor:
    
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        logger.info("Churn predictor initialized")
    
    def prepare_features(self, df):
        df_model = df.copy()
        
        if 'customerID' in df_model.columns:
            df_model = df_model.drop('customerID', axis=1)
        
        if 'Churn' not in df_model.columns:
            raise ValueError("Target variable 'Churn' not found")
        
        X = df_model.drop('Churn', axis=1)
        y = df_model['Churn']
        
        categorical_cols = X.select_dtypes(include=['object', 'str', 'category']).columns.tolist()
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        logger.info(f"Prepared {X.shape[1]} features, {len(categorical_cols)} categorical")
        return X, y
    
    def train_model(self, X, y):
        logger.info("Training model...")
        
        test_size = self.config['model']['test_size']
        random_state = self.config['model']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        rf_params = self.config['model']['random_forest']
        self.model = RandomForestClassifier(
            n_estimators=rf_params['n_estimators'],
            max_depth=rf_params['max_depth'],
            min_samples_split=rf_params['min_samples_split'],
            class_weight=rf_params['class_weight'],
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        results = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled
        }
        
        logger.info("Training complete")
        return results
    
    def evaluate_model(self, results):
        y_test = results['y_test']
        y_pred = results['y_pred']
        y_pred_proba = results['y_pred_proba']
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE")
        print("="*60)
        print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"TN: {cm[0][0]:>4} | FP: {cm[0][1]:>4}")
        print(f"FN: {cm[1][0]:>4} | TP: {cm[1][1]:>4}")
        
        return {'roc_auc': roc_auc, 'confusion_matrix': cm}
    
    def get_feature_importance(self, X, top_n=15):
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Features:")
        for idx, row in feature_importance.head(top_n).iterrows():
            print(f"  {row['feature']:<30} {row['importance']:.4f}")
        
        return feature_importance
    
    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.config['model']['model_path']
        
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        model_artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        
        joblib.dump(model_artifacts, model_path)
        logger.info(f"Model saved: {model_path}")
    
    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.config['model']['model_path']
        
        model_artifacts = joblib.load(model_path)
        self.model = model_artifacts['model']
        self.scaler = model_artifacts['scaler']
        self.label_encoders = model_artifacts['label_encoders']
        logger.info(f"Model loaded: {model_path}")
    
    def predict_churn(self, X):
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities


def train_and_evaluate(df, config_path='config/config.yaml'):
    predictor = ChurnPredictor(config_path)
    X, y = predictor.prepare_features(df)
    results = predictor.train_model(X, y)
    metrics = predictor.evaluate_model(results)
    feature_importance = predictor.get_feature_importance(X)
    predictor.save_model()
    
    return predictor, results, metrics, feature_importance


if __name__ == "__main__":
    from data_processing import load_and_clean_data
    from feature_engineering import engineer_kenyan_features
    
    df = load_and_clean_data()
    df_enhanced = engineer_kenyan_features(df)
    
    predictor = ChurnPredictor()
    X, y = predictor.prepare_features(df_enhanced)
    results = predictor.train_model(X, y)
    metrics = predictor.evaluate_model(results)
    feature_importance = predictor.get_feature_importance(X, top_n=15)
    predictor.save_model()
    
    print("\n Training complete")