import pickle
import os
from clean_alternative_credit_scoring import AlternativeCreditScorer

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Initialize the alternative credit scorer
model = AlternativeCreditScorer()

# Save the model
with open('models/alternative_credit_scorer.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully to models/alternative_credit_scorer.pkl") 