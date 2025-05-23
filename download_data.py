import pandas as pd
import numpy as np
import os
from sklearn.datasets import make_classification

def generate_sample_data(n_samples=1000, output_path='data/credit_data.csv'):
    """Generate synthetic credit risk data for demonstration purposes."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate synthetic features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=5,  # Age, Income, LoanAmount, LoanTerm, CreditScore
        n_informative=4,
        n_redundant=1,
        random_state=42
    )
    
    # Create DataFrame with meaningful feature names
    df = pd.DataFrame(X, columns=['Age', 'Income', 'LoanAmount', 'LoanTerm', 'CreditScore'])
    
    # Scale and transform features to meaningful ranges
    df['Age'] = (df['Age'] * 10 + 50).astype(int)  # Ages between 20-80
    df['Income'] = (df['Income'] * 20000 + 50000).astype(int)  # Income between 10k-90k
    df['LoanAmount'] = (df['LoanAmount'] * 20000 + 30000).astype(int)  # Loan amounts between 10k-50k
    df['LoanTerm'] = (df['LoanTerm'] * 24 + 36).astype(int)  # Loan terms between 12-60 months
    df['CreditScore'] = (df['CreditScore'] * 100 + 650).astype(int)  # Credit scores between 550-750
    
    # Add categorical features
    df['EmploymentStatus'] = np.random.choice(['Employed', 'Unemployed', 'Self-employed'], size=n_samples)
    df['Education'] = np.random.choice(['High School', 'Bachelor\'s', 'Master\'s', 'PhD'], size=n_samples)
    df['MaritalStatus'] = np.random.choice(['Single', 'Married', 'Divorced'], size=n_samples)
    
    # Add target variable (Default: 1 for high risk, 0 for low risk)
    df['Default'] = y
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Generated sample data saved to {output_path}")
    
    return df

if __name__ == "__main__":
    # Generate sample data
    df = generate_sample_data()
    
    # Print data summary
    print("\nData Summary:")
    print(df.describe())
    print("\nSample of generated data:")
    print(df.head()) 