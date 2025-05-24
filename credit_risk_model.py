import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
import json
from datetime import datetime
import logging
from sklearn.impute import SimpleImputer

class CreditRiskModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.version = "1.0.0"
        self.metrics = {}
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            filename='models/model_metrics.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def preprocess_data(self, data):
        """Preprocess the input data for model training or prediction."""
        # Define categorical and numerical features
        categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
        numerical_features = ['person_age', 'person_income', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'person_emp_length', 'cb_person_cred_hist_length']
        
        # Create preprocessing pipelines
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return self.preprocessor.fit_transform(data)
    
    def train(self, data_path):
        """Train the credit risk model using the provided data."""
        # Load data
        df = pd.read_csv(data_path)
        
        # Separate features and target
        X = df.drop('loan_status', axis=1)
        y = df['loan_status']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Preprocess data
        X_train_processed = self.preprocess_data(X_train)
        X_test_processed = self.preprocess_data(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.model.fit(X_train_processed, y_train)
        
        # Evaluate model and store metrics
        self.evaluate_model(X_test_processed, y_test)
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance and store metrics"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'feature_importance': dict(zip(self.feature_names, 
                                        self.model.feature_importances_)),
            'timestamp': datetime.now().isoformat(),
            'version': self.version
        }
        
        # Log metrics
        logging.info(f"Model Evaluation Metrics: {json.dumps(self.metrics)}")
        
        # Print evaluation results
        print("\nModel Evaluation:")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nROC AUC Score:", self.metrics['roc_auc'])
        
    def predict(self, data):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Preprocess input data
        processed_data = self.preprocessor.transform(data)
        
        # Make predictions
        predictions = self.model.predict(processed_data)
        probabilities = self.model.predict_proba(processed_data)
        
        # Log prediction
        logging.info(f"Prediction made for data: {data.to_dict()}")
        
        return predictions, probabilities
    
    def save_model(self, model_path):
        """Save the trained model, preprocessor, and metrics."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model, preprocessor, and metrics
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'version': self.version
        }
        joblib.dump(model_data, model_path)
        
        # Save metrics separately for easy access
        metrics_path = os.path.join(os.path.dirname(model_path), 'model_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    @classmethod
    def load_model(cls, model_path):
        """Load a trained model, preprocessor, and metrics."""
        model_data = joblib.load(model_path)
        
        model = cls()
        model.model = model_data['model']
        model.preprocessor = model_data['preprocessor']
        model.feature_names = model_data['feature_names']
        model.metrics = model_data.get('metrics', {})
        model.version = model_data.get('version', '1.0.0')
        
        return model
    
    def get_feature_importance(self):
        """Return feature importance as a dictionary"""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def get_model_metrics(self):
        """Return current model metrics"""
        return self.metrics

if __name__ == "__main__":
    # Example usage
    model = CreditRiskModel()
    
    # Train model (uncomment when data is available)
    # model.train('data/credit_data.csv')
    # model.save_model('models/credit_risk_model.joblib')
    
    # Load model
    # loaded_model = CreditRiskModel.load_model('models/credit_risk_model.joblib') 