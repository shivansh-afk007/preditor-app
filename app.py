from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Load the model
model = joblib.load('models/credit_risk_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        data = {
            'Age': float(request.form['age']),
            'Income': float(request.form['income']),
            'LoanAmount': float(request.form['loan_amount']),
            'LoanTerm': float(request.form['loan_term']),
            'CreditScore': float(request.form['credit_score']),
            'EmploymentStatus': request.form['employment_status'],
            'Education': request.form['education'],
            'MaritalStatus': request.form['marital_status']
        }
        
        # Convert to DataFrame
        input_data = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        # Prepare response
        result = {
            'prediction': 'High Risk' if prediction[0] == 1 else 'Low Risk',
            'high_risk_probability': f"{probability[0][1]:.2%}",
            'low_risk_probability': f"{probability[0][0]:.2%}"
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 