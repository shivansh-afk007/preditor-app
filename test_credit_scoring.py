import unittest
import pandas as pd
from credit_scoring_model import CreditScoringModel

class TestCreditScoringModel(unittest.TestCase):
    def setUp(self):
        self.model = CreditScoringModel()
        
    def test_preprocessing(self):
        # Create test data with various employment length and verification status formats
        test_data = pd.DataFrame({
            'emp_length': [
                '< 1 year',
                '1 year',
                '2 years',
                '6 years',
                '10+ years',
                None,  # Test missing value
                'invalid'  # Test invalid format
            ],
            'annual_inc': [50000] * 7,
            'verification_status': [
                'Not Verified',
                'Source Verified',
                'Verified',
                'Not Verified',
                'Source Verified',
                'Verified',
                'Invalid Status'  # Test invalid status
            ],
            'delinq_2yrs': [0] * 7,
            'pub_rec': [0] * 7,
            'revol_util': [50] * 7,
            'home_ownership': ['RENT'] * 7,
            'mort_acc': [0] * 7,
            'dti': [20] * 7,
            'open_acc': [5] * 7,
            'total_acc': [10] * 7,
            'inq_last_6mths': [0] * 7,
            'loan_amnt': [10000] * 7,
            'is_default': [0] * 7
        })
        
        # Preprocess the data
        X, y = self.model.preprocess_data(test_data)
        
        # Verify the employment length conversions
        expected_emp_length = [0, 1, 2, 6, 10, 0, 0]
        actual_emp_length = X['emp_length'].tolist()
        self.assertEqual(actual_emp_length, expected_emp_length, 
                        "Employment length preprocessing failed")
        
        # Verify the verification status conversions
        expected_verification = [0, 1, 2, 0, 1, 2, 0]
        actual_verification = X['verification_status'].tolist()
        self.assertEqual(actual_verification, expected_verification,
                        "Verification status preprocessing failed")

if __name__ == '__main__':
    unittest.main() 