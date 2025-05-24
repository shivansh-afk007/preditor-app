from credit_risk_model import CreditRiskModel
from download_data import generate_sample_data
import os

# Define the path for the data file
data_file_path = 'data/credit_risk.csv'
model_path = 'models/credit_risk_model.joblib'

def train_credit_risk_model():
    # Check if the specified data file exists
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at {data_file_path}. Please ensure the data file is in the 'data' folder.")
        return

    print(f"Training model using data from {data_file_path}...")

    # Initialize the model
    model = CreditRiskModel()

    # Train the model
    model.train(data_file_path)

    # Save the trained model
    model.save_model(model_path)

    print(f"Model trained and saved to {model_path}")

if __name__ == "__main__":
    train_credit_risk_model() 