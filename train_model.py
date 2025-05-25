from credit_scoring_model import CreditScoringModel
import os

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize and train the model
    print("Initializing credit scoring model...")
    model = CreditScoringModel()
    
    print("Training model...")
    model.train_model()
    
    print("Model training completed and saved to models/credit_scoring_model.pkl")

if __name__ == "__main__":
    main() 