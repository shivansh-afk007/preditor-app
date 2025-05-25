import pandas as pd
from credit_scoring_model import CreditScoringModel
from sklearn.model_selection import train_test_split
import joblib

def train_model():
    # Load the dummy data
    print("Loading dummy data...")
    df = pd.read_csv('dummy_credit_data.csv')
    
    # Select only the required features for the model
    selected_features = [
        'loan_amnt',
        'emp_length',
        'annual_inc',
        'verification_status',
        'delinq_2yrs',
        'pub_rec',
        'revol_util',
        'home_ownership',
        'mort_acc',
        'dti',
        'open_acc',
        'total_acc',
        'inq_last_6mths'
    ]
    X = df[selected_features].copy()
    y = df['default']
    
    # Encode categorical features as integers
    X['verification_status'] = X['verification_status'].map({'Not Verified': 0, 'Verified': 1, 'Source Verified': 2})
    X['home_ownership'] = X['home_ownership'].map({'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'OTHER': 3})
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    print("Training model...")
    model = CreditScoringModel()
    model.train(X_train, y_train)
    
    # Evaluate the model
    print("\nEvaluating model...")
    train_score = model.model.score(X_train, y_train)
    test_score = model.model.score(X_test, y_test)
    
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Save the model
    print("\nSaving model...")
    joblib.dump(model.model, 'xgboost.pkl')
    print("Model saved as xgboost.pkl")
    
    return model

if __name__ == "__main__":
    train_model() 