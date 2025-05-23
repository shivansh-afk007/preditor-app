import pandas as pd
import joblib

def get_user_input():
    """Get input parameters from user"""
    print("\nPlease enter the following details:")
    
    # Get numerical inputs
    age = float(input("Age: "))
    income = float(input("Income: "))
    loan_amount = float(input("Loan Amount: "))
    loan_term = float(input("Loan Term (in months): "))
    credit_score = float(input("Credit Score: "))
    
    # Get categorical inputs
    print("\nEmployment Status options: Employed, Unemployed, Self-employed")
    employment_status = input("Employment Status: ")
    
    print("\nEducation options: High School, Bachelor's, Master's, PhD")
    education = input("Education: ")
    
    print("\nMarital Status options: Single, Married, Divorced")
    marital_status = input("Marital Status: ")
    
    # Create a dictionary of inputs
    input_data = {
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'LoanTerm': loan_term,
        'CreditScore': credit_score,
        'EmploymentStatus': employment_status,
        'Education': education,
        'MaritalStatus': marital_status
    }
    
    return pd.DataFrame([input_data])

def main():
    try:
        # Load the trained model
        model = joblib.load('models/credit_risk_model.joblib')
        
        while True:
            # Get user input
            user_input = get_user_input()
            
            # Make prediction
            prediction = model.predict(user_input)
            probability = model.predict_proba(user_input)
            
            # Display results
            print("\nPrediction Results:")
            print(f"Credit Risk Prediction: {'High Risk' if prediction[0] == 1 else 'Low Risk'}")
            print(f"Probability of High Risk: {probability[0][1]:.2%}")
            print(f"Probability of Low Risk: {probability[0][0]:.2%}")
            
            # Ask if user wants to make another prediction
            another = input("\nWould you like to make another prediction? (yes/no): ")
            if another.lower() != 'yes':
                break
                
    except FileNotFoundError:
        print("Error: Model file not found. Please train the model first by running train_model.py")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 