import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

class CreditRiskModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        
    def preprocess_data(self, data):
        """Preprocess the input data for model training or prediction."""
        # Define categorical and numerical features
        categorical_features = ['EmploymentStatus', 'Education', 'MaritalStatus']
        numerical_features = ['Age', 'Income', 'LoanAmount', 'LoanTerm', 'CreditScore']
        
        # Create preprocessing pipelines
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
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
        X = df.drop('Default', axis=1)  # Assuming 'Default' is the target column
        y = df['Default']
        
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
        
        # Evaluate model
        y_pred = self.model.predict(X_test_processed)
        print("\nModel Evaluation:")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return self.model
    
    def predict(self, data):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        # Preprocess input data
        processed_data = self.preprocessor.transform(data)
        
        # Make predictions
        predictions = self.model.predict(processed_data)
        probabilities = self.model.predict_proba(processed_data)
        
        return predictions, probabilities
    
    def save_model(self, model_path):
        """Save the trained model and preprocessor."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and preprocessor
        model_data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, model_path)
    
    @classmethod
    def load_model(cls, model_path):
        """Load a trained model and preprocessor."""
        model_data = joblib.load(model_path)
        
        model = cls()
        model.model = model_data['model']
        model.preprocessor = model_data['preprocessor']
        model.feature_names = model_data['feature_names']
        
        return model

if __name__ == "__main__":
    # Example usage
    model = CreditRiskModel()
    
    # Train model (uncomment when data is available)
    # model.train('data/credit_data.csv')
    # model.save_model('models/credit_risk_model.joblib')
    
    # Load model
    # loaded_model = CreditRiskModel.load_model('models/credit_risk_model.joblib') 