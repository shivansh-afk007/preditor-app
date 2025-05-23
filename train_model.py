from credit_risk_model import CreditRiskModel
from download_data import generate_sample_data
import os

def train_credit_risk_model():
    """Train the credit risk model using sample data."""
    # Generate sample data if it doesn't exist
    data_path = 'data/credit_data.csv'
    if not os.path.exists(data_path):
        print("Generating sample data...")
        generate_sample_data()
    
    # Initialize and train model
    print("\nTraining credit risk model...")
    model = CreditRiskModel()
    model.train(data_path)
    
    # Save trained model
    model_path = 'models/credit_risk_model.joblib'
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train_credit_risk_model() 