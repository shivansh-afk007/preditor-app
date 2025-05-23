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

def load_and_preprocess_data():
    # Load the data
    df = pd.read_csv('data/credit_risk.csv')
    
    # Separate features and target
    X = df.drop('Risk', axis=1)
    y = df['Risk']
    
    # Define numerical and categorical columns
    numerical_cols = ['Age', 'Income', 'LoanAmount', 'LoanTerm', 'CreditScore']
    categorical_cols = ['EmploymentStatus', 'Education', 'MaritalStatus']
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return X, y, preprocessor

def train_model():
    # Load and preprocess data
    X, y, preprocessor = load_and_preprocess_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    print("Training the model...")
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("\nModel Evaluation:")
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(model, 'models/credit_risk_model.joblib')
    print("\nModel saved to 'models/credit_risk_model.joblib'")
    
    return model

if __name__ == "__main__":
    train_model() 