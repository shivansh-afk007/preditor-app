import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

def generate_dummy_data(n_samples=1000):
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate core features
    data = {
        'loan_amnt': np.random.uniform(1000, 35000, n_samples),
        'emp_length': np.random.uniform(0, 10, n_samples),
        'annual_inc': np.random.uniform(20000, 200000, n_samples),
        'verification_status': np.random.choice(['Verified', 'Source Verified', 'Not Verified'], n_samples),
        'dti': np.random.uniform(0, 40, n_samples),
        'delinq_2yrs': np.random.randint(0, 10, n_samples),
        'inq_last_6mths': np.random.randint(0, 10, n_samples),
        'mths_since_last_delinq': np.random.uniform(0, 100, n_samples),
        'mths_since_last_record': np.random.uniform(0, 100, n_samples),
        'open_acc': np.random.randint(0, 30, n_samples),
        'pub_rec': np.random.randint(0, 10, n_samples),
        'revol_bal': np.random.uniform(0, 100000, n_samples),
        'revol_util': np.random.uniform(0, 100, n_samples),
        'total_acc': np.random.randint(0, 50, n_samples),
        'mort_acc': np.random.randint(0, 10, n_samples),
        'pub_rec_bankruptcies': np.random.randint(0, 5, n_samples),
        'term': np.random.choice(['36 months', '60 months'], n_samples),
        'int_rate': np.random.uniform(5, 30, n_samples),
        'grade': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n_samples),
        'sub_grade': np.random.choice(['A1', 'A2', 'A3', 'A4', 'A5', 
                                     'B1', 'B2', 'B3', 'B4', 'B5',
                                     'C1', 'C2', 'C3', 'C4', 'C5',
                                     'D1', 'D2', 'D3', 'D4', 'D5',
                                     'E1', 'E2', 'E3', 'E4', 'E5',
                                     'F1', 'F2', 'F3', 'F4', 'F5',
                                     'G1', 'G2', 'G3', 'G4', 'G5'], n_samples),
        'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE', 'OTHER'], n_samples),
        'purpose': np.random.choice(['credit_card', 'debt_consolidation', 'home_improvement', 
                                   'major_purchase', 'small_business', 'other'], n_samples),
        'addr_state': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'], n_samples),
        'initial_list_status': np.random.choice(['f', 'w'], n_samples),
        'application_type': np.random.choice(['Individual', 'Joint App'], n_samples),
        'policy_code': np.random.choice([1, 2], n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate derived features
    df['loan_amnt_norm'] = df['loan_amnt'] / df['annual_inc']
    df['emp_length_norm'] = df['emp_length'] / 10
    df['dti_norm'] = df['dti'] / 100
    df['revol_util_norm'] = df['revol_util'] / 100
    df['int_rate_norm'] = df['int_rate'] / 100
    df['delinq_ratio'] = df['delinq_2yrs'] / (df['total_acc'] + 1)
    df['inq_ratio'] = df['inq_last_6mths'] / (df['total_acc'] + 1)
    df['revol_bal_norm'] = df['revol_bal'] / df['annual_inc']
    df['credit_history_strength'] = (df['total_acc'] * 0.5) + (df['emp_length'] * 0.5)
    
    # Generate target variable (default)
    # Higher probability of default for:
    # - High DTI
    # - Low income
    # - High interest rate
    # - High delinquency
    # - Low employment length
    default_prob = (
        df['dti_norm'] * 0.3 +
        (1 - df['emp_length_norm']) * 0.2 +
        df['int_rate_norm'] * 0.2 +
        df['delinq_ratio'] * 0.2 +
        df['inq_ratio'] * 0.1
    )
    
    # Add some noise
    default_prob += np.random.normal(0, 0.1, n_samples)
    default_prob = np.clip(default_prob, 0, 1)
    
    # Convert to binary
    df['default'] = (default_prob > 0.5).astype(int)
    
    return df

if __name__ == "__main__":
    # Generate data
    df = generate_dummy_data(n_samples=5000)
    
    # Save to CSV
    df.to_csv('dummy_credit_data.csv', index=False)
    print("Generated dummy data with shape:", df.shape)
    print("\nSample of generated data:")
    print(df.head())
    print("\nDefault rate:", df['default'].mean()) 