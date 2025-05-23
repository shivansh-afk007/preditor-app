import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

class CreditRiskModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """Load and preprocess the data"""
        df = pd.read_csv(file_path)
        
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Convert categorical variables to numerical
        categorical_cols = df.select_dtypes(include=['object']).columns
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        return df
    
    def prepare_data(self, df):
        """Prepare data for training"""
        X = df.drop('Risk', axis=1)
        y = df['Risk']
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    def train_model(self, X_train, y_train):
        """Train the model"""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model"""
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    def save_model(self, model_path='model', scaler_path='scaler'):
        """Save the trained model and scaler"""
        if not os.path.exists('models'):
            os.makedirs('models')
        joblib.dump(self.model, f'models/{model_path}.joblib')
        joblib.dump(self.scaler, f'models/{scaler_path}.joblib')
    
    def load_saved_model(self, model_path='models/model.joblib', scaler_path='models/scaler.joblib'):
        """Load the saved model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
    
    def predict(self, input_data):
        """Make predictions on new data"""
        # Scale the input data
        input_scaled = self.scaler.transform(input_data)
        # Make prediction
        prediction = self.model.predict(input_scaled)
        probability = self.model.predict_proba(input_scaled)
        return prediction, probability

def main():
    # Initialize the model
    credit_model = CreditRiskModel()
    
    # Load and prepare data
    df = credit_model.load_data('data/credit_risk.csv')
    X_train, X_test, y_train, y_test = credit_model.prepare_data(df)
    
    # Train the model
    credit_model.train_model(X_train, y_train)
    
    # Evaluate the model
    credit_model.evaluate_model(X_test, y_test)
    
    # Save the model
    credit_model.save_model()

if __name__ == "__main__":
    main() 