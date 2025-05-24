from flask import Flask, render_template, redirect, url_for, request, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import joblib
import pandas as pd
import os
import json # Using json for simple user data storage for now
from datetime import datetime, timedelta
import random  # For demo purposes, replace with actual data in production

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here' # Replace with a real secret key

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- Jinja2 Filters ---
@app.template_filter('format_currency')
def format_currency_filter(value):
    if value is None:
        return "N/A"
    # Format as currency (e.g., $1,234)
    return f"{value:,.0f}"

# --- User Management (Simple JSON based for demo) ---
USERS_FILE = 'users.json'

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {} # Return empty dict if file is empty or invalid JSON

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

class User(UserMixin):
    def __init__(self, user_id, username, password, role):
        self.id = user_id
        self.username = username
        self.password = password
        self.role = role # 'consumer' or 'lender'

@login_manager.user_loader
def load_user(user_id):
    users = load_users()
    if user_id in users:
        user_data = users[user_id]
        return User(user_id, user_data['username'], user_data['password'], user_data['role'])
    return None

# --- Routes ---

@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.role == 'consumer':
            return redirect(url_for('consumer_dashboard'))
        elif current_user.role == 'lender':
            return redirect(url_for('lender_dashboard'))
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        users = load_users()
        for user_id, user_data in users.items():
            # In a real app, hash passwords and compare securely
            if user_data['username'] == username and user_data['password'] == password:
                user = User(user_id, user_data['username'], user_data['password'], user_data['role'])
                login_user(user)
                next_page = request.args.get('next')
                return redirect(next_page or url_for('index'))
        return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        role = request.form.get('role') # Get role from form ('consumer' or 'lender')

        users = load_users()
        if username in [u['username'] for u in users.values()]:
            return render_template('register.html', error="Username already exists")

        user_id = str(len(users) + 1)
        users[user_id] = {'username': username, 'password': password, 'role': role} # Store role
        save_users(users)

        # Optional: automatically log in the user after registration
        # user = User(user_id, username, password, role)
        # login_user(user)

        return redirect(url_for('login')) # Redirect to login after successful registration
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard/consumer')
@login_required
def consumer_dashboard():
    if current_user.role != 'consumer':
        return "Unauthorized", 403 # Or redirect to their correct dashboard

    # --- Consumer Dashboard Data Mockup ---
    dashboard_data = {
        'credit_score': 752,
        'credit_score_status': 'Good', # Added status for gauge
        'earnings': 6320,
        'credit_factors': {
            'Income': 'High',
            'Cash Flow': 'Very Good',
            'Employment': 'Stable'
        },
        'recent_activity': [
            {'date': 'Apr 22', 'description': 'Direct deposit', 'amount': '+$3,160.00'},
            {'date': 'Apr 17', 'description': 'Grover Energy', 'amount': '-$150.00'},
            {'date': 'Apr 15', 'description': 'Credit Card Payment', 'amount': '-$500.00'},
             {'date': 'Apr 10', 'description': 'Restaurant', 'amount': '-$45.50'},
        ],
        'prediction': 'Low Risk', # Example prediction
        'high_risk_probability': '15%', # Example probability
        'low_risk_probability': '85%', # Example probability
        'recommendations': [
            'Keep up the good credit habits!',
            'Consider increasing your credit limit to improve utilization.',
            'Explore options for reducing outstanding debt.'
        ],
        'spending_data': {
            'labels': ['Groceries', 'Utilities', 'Rent', 'Transport', 'Others'],
            'data': [300, 150, 1200, 100, 400],
            'backgroundColor': ['#4e73df', '#1cc88a', '#36b9cc', '#f6c23e', '#a0a0a0']
        },
        'credit_utilization_data': {
            'labels': ['Used Credit', 'Available Credit'],
            'data': [30, 70],
            'backgroundColor': ['#e74a3b', '#1cc88a']
        },
        'loan_applications': [
            {'type': 'Personal Loan', 'amount': '$10,000', 'status': 'Pending Review', 'date': '2023-10-26'},
            {'type': 'Auto Loan', 'amount': '$25,000', 'status': 'Approved', 'date': '2023-10-01'},
        ]
    }

     # Determine credit score status for the gauge
    score = dashboard_data['credit_score']
    if score < 580: dashboard_data['credit_score_status'] = 'Poor'
    elif score < 670: dashboard_data['credit_score_status'] = 'Fair'
    elif score < 740: dashboard_data['credit_score_status'] = 'Good'
    elif score < 800: dashboard_data['credit_score_status'] = 'Very Good'
    else: dashboard_data['credit_score_status'] = 'Excellent'

    return render_template('dashboard_consumer.html', data=dashboard_data)

@app.route('/dashboard/lender')
@login_required
def lender_dashboard():
    if current_user.role != 'lender':
        return "Unauthorized", 403 # Or redirect to their correct dashboard

    # --- Lender Dashboard Data Mockup ---
    dashboard_data = {
        'total_clients': 150,
        'total_assets': '5.2M', # Example data
        'avg_risk_score': 680,
        'client_list': [
            {'name': 'Client A', 'risk_score': 720, 'loan_amount': '$15,000', 'status': 'Approved'},
            {'name': 'Client B', 'risk_score': 550, 'loan_amount': '$5,000', 'status': 'Pending'},
             {'name': 'Client C', 'risk_score': 630, 'loan_amount': '$25,000', 'status': 'Approved'},
             {'name': 'Client D', 'risk_score': 780, 'loan_amount': '$50,000', 'status': 'Approved'},
             {'name': 'Client E', 'risk_score': 490, 'loan_amount': '$2,000', 'status': 'Rejected'},
        ],
        'risk_distribution_data': {
            'labels': ['Low Risk', 'Medium Risk', 'High Risk'],
            'data': [70, 20, 10],
            'backgroundColor': ['#27ae60', '#f39c12', '#e74c3c'],
        },
         'application_status_data': {
            'labels': ['Approved', 'Pending', 'Rejected'],
            'data': [120, 25, 5],
            'backgroundColor': ['#3498db', '#f1c40f', '#e74c3c'],
        },
        'recent_applications': [
             {'name': 'Applicant C', 'amount': '$20,000', 'score': 680, 'date': '2023-10-25', 'status': 'Under Review'},
              {'name': 'Applicant D', 'amount': '$5,000', 'score': 750, 'date': '2023-10-24', 'status': 'Approved'},
        ],
         'loan_performance_data': {
            'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'data': [65, 59, 80, 81, 56, 55],
            'borderColor': '#2d6cdf',
            'fill': False
        },
         'geographic_distribution_data': {
            'labels': ['Region A', 'Region B', 'Region C'],
            'data': [50, 30, 70],
            'backgroundColor': ['#1cc88a', '#f6c23e', '#36b9cc'],
        }

    }
    return render_template('dashboard_lender.html', data=dashboard_data)

# --- Credit Risk Prediction (Integrate existing logic, modify for web) ---
# Load the model (assuming 'models/credit_risk_model.joblib' exists)
try:
    from credit_risk_model import CreditRiskModel
    model = CreditRiskModel.load_model('models/credit_risk_model.joblib')
except FileNotFoundError:
    model = None # Handle case where model is not trained

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if current_user.role != 'consumer':
         return jsonify({'error': 'Only consumer accounts can make predictions'}), 403

    if model is None:
         return jsonify({'error': 'Model not trained. Please run the training script first.'}), 500

    try:
        # Get data from the form and map to model feature names
        data = {
            'person_age': float(request.form.get('age')), # Map 'age' to 'person_age'
            'person_income': float(request.form.get('income')), # Map 'income' to 'person_income'
            'loan_amnt': float(request.form.get('loan_amount')), # Map 'loan_amount' to 'loan_amnt'
            'person_emp_length': float(request.form.get('loan_term')), # Map 'loan_term' to 'person_emp_length' (using loan term as proxy)
            'cb_person_cred_hist_length': float(request.form.get('credit_score')), # Map 'credit_score' to 'cb_person_cred_hist_length' (using credit score as proxy)
            'person_home_ownership': request.form.get('employment_status'), # Map 'employment_status' to 'person_home_ownership' (using employment status as proxy)
            'loan_intent': request.form.get('education'), # Map 'education' to 'loan_intent' (using education as proxy)
            'loan_grade': request.form.get('marital_status'), # Map 'marital_status' to 'loan_grade' (using marital status as proxy)
            
            # Add placeholder values for features not in the form
            'loan_int_rate': 12.0, # Placeholder
            'loan_percent_income': 0.15, # Placeholder
            'cb_person_default_on_file': 'N' # Placeholder
        }

        # Ensure all required fields are present and valid (check original form fields)
        required_form_fields = ['age', 'income', 'loan_amount', 'loan_term', 'credit_score', 'employment_status', 'education', 'marital_status']
        if any(request.form.get(field) is None for field in required_form_fields):
             return jsonify({'error': 'Missing form data'}), 400

        # Convert to DataFrame
        input_data = pd.DataFrame([data])

        # Make prediction
        predictions, probabilities = model.predict(input_data)
        
        # Determine prediction result and recommendations
        predicted_risk = 'High Risk' if predictions[0] == 1 else 'Low Risk'
        high_risk_prob = f"{probabilities[0][1]:.2%}"
        low_risk_prob = f"{probabilities[0][0]:.2%}"

        recommendations = []
        # Generate recommendations based on input data
        if predictions[0] == 1:  # High risk
            if data['cb_person_cred_hist_length'] < 600:
                recommendations.append('Consider improving your credit score by paying bills on time and reducing debt.')
            if data['person_income'] < 30000:
                recommendations.append('Look for ways to increase your income or reduce the loan amount.')
            if data['loan_amnt'] > data['person_income'] * 0.5:
                recommendations.append('Consider requesting a smaller loan amount relative to your income.')
        else:  # Low risk
            recommendations.append('Your credit profile looks good! Maintain your current financial habits.')
            if data['cb_person_cred_hist_length'] < 700:
                recommendations.append('You could potentially get better rates by improving your credit score further.')
            if data['loan_amnt'] < data['person_income'] * 0.2:
                recommendations.append('You may be eligible for larger loan amounts if needed.')

        result = {
            'prediction': predicted_risk,
            'high_risk_probability': high_risk_prob,
            'low_risk_probability': low_risk_prob,
            'recommendations': recommendations
        }

        return jsonify(result)

    except Exception as e:
        # Log the error in a real application
        print(f"Prediction Error: {e}")
        return jsonify({'error': 'An error occurred during prediction. Please check inputs.'}), 500

@app.route('/api/credit-score/history')
@login_required
def get_credit_score_history():
    if current_user.role != 'consumer':
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Generate sample data for the last 6 months
    # In production, this should come from a database
    today = datetime.now()
    dates = [(today - timedelta(days=30*i)).strftime('%b') for i in range(5, -1, -1)]
    
    # Base score with some random variation
    base_score = 750
    scores = [base_score + random.randint(-20, 20) for _ in range(6)]
    
    return jsonify({
        'labels': dates,
        'scores': scores
    })

@app.route('/api/credit-score/simulate', methods=['POST'])
@login_required
def simulate_credit_score():
    if current_user.role != 'consumer':
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.get_json()
    simulation_type = data.get('type')
    current_score = data.get('currentScore', 750)  # Default to 750 if not provided
    
    # Simple simulation logic (replace with more sophisticated calculations in production)
    result = {
        'simulatedScore': current_score,
        'explanation': '',
        'factors': []
    }
    
    if simulation_type == 'payment':
        amount = float(data.get('amount', 0))
        increase = min(30, int(amount / 1000) * 5)
        result['simulatedScore'] = current_score + increase
        result['explanation'] = f'Making a payment of ${amount:,.2f} could improve your score by reducing your credit utilization.'
        result['factors'] = [
            {'name': 'Credit Utilization', 'impact': f'+{increase} points'},
            {'name': 'Payment History', 'impact': '+5 points'}
        ]
    
    elif simulation_type == 'credit':
        new_limit = float(data.get('limit', 0))
        increase = min(20, int(new_limit / 5000) * 5)
        result['simulatedScore'] = current_score + increase
        result['explanation'] = f'Increasing your credit limit to ${new_limit:,.2f} could improve your score by lowering your credit utilization ratio.'
        result['factors'] = [
            {'name': 'Credit Utilization', 'impact': f'+{increase} points'},
            {'name': 'Available Credit', 'impact': '+5 points'}
        ]
    
    elif simulation_type == 'loan':
        amount = float(data.get('amount', 0))
        term = int(data.get('term', 36))
        decrease = min(20, int(amount / 10000) * 5)
        result['simulatedScore'] = current_score - decrease
        result['explanation'] = f'Taking a new loan of ${amount:,.2f} for {term} months might temporarily lower your score.'
        result['factors'] = [
            {'name': 'New Credit Inquiry', 'impact': '-5 points'},
            {'name': 'Debt-to-Income Ratio', 'impact': f'-{decrease} points'}
        ]
    
    elif simulation_type == 'utilization':
        target_util = float(data.get('utilization', 30))
        current_util = 30  # This should come from actual data
        change = int((current_util - target_util) / 5) * 10
        result['simulatedScore'] = current_score + change
        result['explanation'] = f'Reducing your credit utilization from {current_util}% to {target_util}% could improve your score.'
        result['factors'] = [
            {'name': 'Credit Utilization', 'impact': f'{change:+d} points'},
            {'name': 'Credit Mix', 'impact': '+5 points'}
        ]
    
    return jsonify(result)

if __name__ == '__main__':
    # Create users.json if it doesn't exist
    if not os.path.exists(USERS_FILE):
        save_users({})
        print(f"Created empty {USERS_FILE}")

    app.run(debug=True) 