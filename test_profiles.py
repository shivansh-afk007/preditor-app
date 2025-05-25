from credit_scoring_model import CreditScoringModel

def test_profiles():
    # Initialize the model
    model = CreditScoringModel()
    
    # Define test profiles
    good_profile = {
        'loan_amnt': 15000,
        'emp_length': 10,
        'annual_inc': 150000,
        'verification_status': 2,  # Source Verified
        'delinq_2yrs': 0,
        'pub_rec': 0,
        'revol_util': 5,
        'home_ownership': 'OWN',
        'mort_acc': 3,
        'dti': 5,
        'open_acc': 5,
        'total_acc': 10,
        'inq_last_6mths': 0
    }

    average_profile = {
        'loan_amnt': 20,
        'emp_length': 40,
        'annual_inc': 650000,
        'verification_status': 1,  # Verified
        'delinq_2yrs': 0,
        'pub_rec': 0,
        'revol_util': 10,
        'home_ownership': 'OWN',
        'mort_acc': 1,
        'dti': 22,
        'open_acc': 8,
        'total_acc': 11,
        'inq_last_6mths': 1
    }

    poor_profile = {
        'loan_amnt': 30000,
        'emp_length': 1,
        'annual_inc': 45000,
        'verification_status': 0,  # Not Verified
        'delinq_2yrs': 3,
        'pub_rec': 1,
        'revol_util': 85,
        'home_ownership': 'RENT',
        'mort_acc': 0,
        'dti': 35,
        'open_acc': 12,
        'total_acc': 12,
        'inq_last_6mths': 5
    }

    # Test each profile
    profiles = {
        "Good Credit Profile": good_profile,
        "Average Credit Profile": average_profile,
        "Poor Credit Profile": poor_profile
    }

    for profile_name, profile in profiles.items():
        print(f"\n{profile_name} - Model Comparison:")
        print("-" * 80)
        print(f"{'Model':<25} {'Score':<10} {'Grade':<10} {'Default Prob':<15} {'Recommendation':<20}")
        print("-" * 80)
        
        result = model.predict_credit_score(profile)
        print(f"{'Logistic Regression':<25} {result['score']:<10} {result['grade']:<10} {result['default_probability']:.4f} {result['recommendation']:<20}")
        print("-" * 80)

if __name__ == "__main__":
    test_profiles() 