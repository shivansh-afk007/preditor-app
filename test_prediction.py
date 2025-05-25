from app import app, db, User, Prediction
from credit_scoring_model import CreditScoringModel
from datetime import datetime

def test_prediction_flow():
    with app.app_context():
        try:
            # Create test user
            test_user = User(
                username='test_user',
                name='Test User',
                email='test@example.com',
                phone='1234567890',
                role='consumer',
                created_at=datetime.utcnow()
            )
            test_user.set_password('testpass123')
            db.session.add(test_user)
            db.session.commit()
            print("Created test user:", test_user.username)

            # Initialize credit scoring model
            credit_model = CreditScoringModel()

            # Create test input data
            input_data = {
                'loan_amnt': 5000,
                'emp_length': '5',
                'annual_inc': 100000,
                'verification_status': '1',
                'delinq_2yrs': 0,
                'pub_rec': 0,
                'revol_util': 20,
                'home_ownership': 'MORTGAGE',
                'mort_acc': 1,
                'dti': 15,
                'open_acc': 3,
                'total_acc': 5,
                'inq_last_6mths': 1,
                'loan_amnt_norm': 0.5,
                'emp_length_norm': 0.5,
                'annual_inc_norm': 0.5,
                'income_stability_score': 0.5,
                'delinq_2yrs_norm': 0.5,
                'pub_rec_norm': 0.5,
                'payment_consistency_score': 0.5,
                'home_ownership_score': 0.5,
                'mort_acc_norm': 0.5,
                'asset_value_score': 0.5,
                'dti_norm': 0.5,
                'inq_norm': 0.5,
                'behavioral_score': 0.5,
                'alternative_credit_score': 0.5
            }

            # Get prediction
            result = credit_model.predict_credit_score(input_data)
            print("\nPrediction Results:")
            print("Score:", result['score'])
            print("Grade:", result['grade'])
            print("Default Probability:", result['default_probability'])
            print("Recommendation:", result['recommendation'])
            print("Rate Range:", result['rate_range'])

            # Create prediction record
            prediction = Prediction(
                user_id=test_user.id,
                score=float(result['score']),
                grade=result['grade'],
                default_probability=float(result['default_probability']),
                recommendation=result['recommendation'],
                rate_range=result['rate_range'],
                loan_amount=input_data['loan_amnt'],
                employment_length=int(input_data['emp_length']),
                annual_income=input_data['annual_inc'],
                verification_status=int(input_data['verification_status']),
                delinquencies_2yrs=input_data['delinq_2yrs'],
                public_records=input_data['pub_rec'],
                revolving_utilization=input_data['revol_util'],
                home_ownership=input_data['home_ownership'],
                mortgage_accounts=input_data['mort_acc'],
                debt_to_income=input_data['dti'],
                open_accounts=input_data['open_acc'],
                total_accounts=input_data['total_acc'],
                inquiries_6mths=input_data['inq_last_6mths'],
                income_stability=str(result['breakdown']['income_stability']),
                payment_consistency=str(result['breakdown']['payment_consistency']),
                asset_profile=str(result['breakdown']['asset_profile']),
                behavioral_factors=float(result['breakdown']['behavioral_factors']) if isinstance(result['breakdown']['behavioral_factors'], (int, float)) else 0.0
            )
            db.session.add(prediction)
            db.session.commit()
            print("\nCreated prediction record for test user")

            # Verify prediction was created
            saved_prediction = Prediction.query.filter_by(user_id=test_user.id).first()
            if saved_prediction:
                print("\nVerified prediction record exists in database")
                print("Prediction ID:", saved_prediction.id)
                print("Score:", saved_prediction.score)
                print("Grade:", saved_prediction.grade)
                print("Default Probability:", saved_prediction.default_probability)
                print("Recommendation:", saved_prediction.recommendation)

        except Exception as e:
            print("Error during test:", str(e))
            db.session.rollback()
            raise e

        finally:
            # Clean up test data
            try:
                # Delete prediction first (due to foreign key constraint)
                Prediction.query.filter_by(user_id=test_user.id).delete()
                # Delete test user
                User.query.filter_by(username='test_user').delete()
                db.session.commit()
                print("\nCleaned up test data successfully")
            except Exception as e:
                print("Error during cleanup:", str(e))
                db.session.rollback()

if __name__ == '__main__':
    test_prediction_flow() 