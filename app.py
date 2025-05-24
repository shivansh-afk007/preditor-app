from flask import Flask, render_template, redirect, url_for, request, jsonify, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import joblib
import pandas as pd
import os
import json # Using json for simple user data storage for now
from datetime import datetime, timedelta
import random  # For demo purposes, replace with actual data in production
import string
import uuid
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np

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
        # Create an empty users.json if it doesn't exist
        empty_users = {}
        with open(USERS_FILE, 'w') as f:
            json.dump(empty_users, f, indent=4)
    with open(USERS_FILE, 'r') as f:
        users_data = json.load(f)
    return users_data

def save_users(users_data):
    with open(USERS_FILE, 'w') as f:
        json.dump(users_data, f, indent=4)

class User(UserMixin):
    def __init__(self, id, password, role, name=None, predictions=[]):
        self.id = id
        self.password = password
        self.role = role
        self.name = name
        self.predictions = predictions

    @staticmethod
    def get(user_id):
        users = load_users()
        user_data = users.get(user_id)
        if user_data:
            return User(user_id, user_data['password'], user_data['role'], user_data.get('name'), user_data.get('predictions', []))
        return None

@login_manager.user_loader
def load_user(user_id):
    users = load_users()
    if user_id in users:
        user_data = users[user_id]
        return User(user_id, user_data['password'], user_data['role'], user_data.get('name'), user_data.get('predictions', []))
    return None

# --- Data Preparation Function ---
def get_consumer_dashboard_data(user_id):
    users_data = load_users()
    user_data = users_data.get(user_id)

    if not user_data or user_data.get('role') != 'consumer':
        return None # User not found or not a consumer

    latest_prediction = None
    if 'predictions' in user_data and user_data['predictions']:
        try:
            # Sort predictions by timestamp and get the latest
            latest_prediction = sorted(user_data['predictions'], key=lambda x: x['timestamp'])[-1]
        except (TypeError, IndexError):
            # Handle cases where predictions list is malformed or empty after check
            latest_prediction = None

    # --- Initialize Dashboard Data with default values ---
    dashboard_data = {
        'credit_score': user_data.get('credit_score', 752), # Use user's score if available, default otherwise
        'credit_score_status': 'Good', 
        'earnings': user_data.get('earnings', 6320), # Use user's earnings if available
        'credit_factors': user_data.get('credit_factors', {
            'Income': 'N/A',
            'Cash Flow': 'N/A',
            'Employment': 'N/A'
        }), # Use user's factors if available
        'recent_activity': user_data.get('recent_activity', []), # Use user's activity if available
        'prediction': 'No prediction yet.',
        'high_risk_probability': 'N/A',
        'low_risk_probability': 'N/A',
        'low_risk_probability_numeric': 0.5,
        'recommendations': ['Submit the form above to get your first prediction!'],
        'spending_data': user_data.get('spending_data', {'labels': [], 'data': [], 'backgroundColor': []}),
        'credit_utilization_data': user_data.get('credit_utilization_data', {'labels': [], 'data': [], 'backgroundColor': []}),
        'loan_applications': user_data.get('loan_applications', []),
    }

    # Override with latest prediction data if available
    if latest_prediction:
        dashboard_data['prediction'] = latest_prediction.get('prediction_result', 'No prediction result')
        
        # Safely get probabilities with default values
        probabilities = latest_prediction.get('probabilities', [0.5, 0.5])  # Default to 50-50 if not available
        if isinstance(probabilities, list) and len(probabilities) >= 2:
            try:
                high_risk_prob = float(probabilities[1])
                low_risk_prob = float(probabilities[0])
                dashboard_data['high_risk_probability'] = f"{high_risk_prob:.2%}"
                dashboard_data['low_risk_probability'] = f"{low_risk_prob:.2%}"
                dashboard_data['low_risk_probability_numeric'] = low_risk_prob
            except (ValueError, TypeError):
                dashboard_data['high_risk_probability'] = 'N/A'
                dashboard_data['low_risk_probability'] = 'N/A'
                dashboard_data['low_risk_probability_numeric'] = 0.5
        else:
            dashboard_data['high_risk_probability'] = 'N/A'
            dashboard_data['low_risk_probability'] = 'N/A'
            dashboard_data['low_risk_probability_numeric'] = 0.5
            
        dashboard_data['recommendations'] = latest_prediction.get('recommendations', ['No recommendations available'])
        
        # Update credit score and status based on the latest prediction if available
        # This is a simplified mapping, adjust as needed
        if 'low_risk_probability_numeric' in dashboard_data:
             prob = dashboard_data['low_risk_probability_numeric']
             if prob > 0.8: dashboard_data['credit_score'] = 780 # Excellent/Very Good range
             elif prob > 0.6: dashboard_data['credit_score'] = 720 # Good range
             elif prob > 0.4: dashboard_data['credit_score'] = 630 # Fair range
             else: dashboard_data['credit_score'] = 500 # Poor range

    # Determine credit score status based on the final score (either mock, user's, or prediction-derived)
    score = dashboard_data.get('credit_score', 0) # Use default 0 if somehow still missing
    if score < 580: dashboard_data['credit_score_status'] = 'Poor'
    elif score < 670: dashboard_data['credit_score_status'] = 'Fair'
    elif score < 740: dashboard_data['credit_score_status'] = 'Good'
    elif score < 800: dashboard_data['credit_score_status'] = 'Very Good'
    else: dashboard_data['credit_score_status'] = 'Excellent'

    return dashboard_data

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

        # Find the user by username
        user_id = None
        user_data = None
        for uid, data in users.items():
            if data.get('username') == username:
                user_id = uid
                user_data = data
                break

        if user_data and check_password_hash(user_data['password'], password):
            user = User(user_id, user_data['password'], user_data['role'])
            login_user(user)
            next_page = request.args.get('next')
            if user.role == 'consumer':
                return redirect(url_for('consumer_dashboard'))
            elif user.role == 'lender':
                return redirect(url_for('lender_dashboard'))
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
        users[user_id] = {'username': username, 'password': generate_password_hash(password), 'role': role} # Store role
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

    # Get data for the logged-in consumer
    dashboard_data = get_consumer_dashboard_data(current_user.id)
    
    if dashboard_data is None:
        # Handle case where user data is not found or not a consumer (shouldn't happen with @login_required and role check)
        flash('Could not load consumer dashboard data.', 'danger')
        return redirect(url_for('index'))

    return render_template('dashboard_consumer.html', data=dashboard_data, user_role=current_user.role, is_lender_view=False)

@app.route('/dashboard/lender')
@login_required
def lender_dashboard():
    if current_user.role != 'lender':
        return redirect(url_for('dashboard_consumer')) # Redirect if not a lender

    data = get_lender_dashboard_data()

    return render_template('dashboard_lender.html', data=data)

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
            'person_age': float(request.form.get('age')),
            'person_income': float(request.form.get('income')),
            'loan_amnt': float(request.form.get('loan_amount')),
            'person_emp_length': float(request.form.get('loan_term')),
            'cb_person_cred_hist_length': float(request.form.get('credit_score')),
            'person_home_ownership': request.form.get('employment_status'),
            'loan_intent': request.form.get('education'),
            'loan_grade': request.form.get('marital_status'),

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

        # Safely handle predictions and probabilities
        if not isinstance(predictions, (list, np.ndarray)) or len(predictions) == 0:
            return jsonify({'error': 'Invalid prediction result'}), 500

        if not isinstance(probabilities, (list, np.ndarray)) or len(probabilities) == 0:
            return jsonify({'error': 'Invalid probability result'}), 500

        # Get the first prediction and probabilities
        prediction = predictions[0]
        probs = probabilities[0] if isinstance(probabilities[0], (list, np.ndarray)) else probabilities

        # Ensure we have at least two probabilities
        if len(probs) < 2:
            return jsonify({'error': 'Invalid probability format'}), 500

        # Determine prediction result and recommendations
        predicted_risk = 'High Risk' if prediction == 1 else 'Low Risk'
        
        # Convert probabilities to float and ensure they're valid
        try:
            high_risk_prob_numeric = float(probs[1])
            low_risk_prob_numeric = float(probs[0])
        except (ValueError, TypeError, IndexError):
            return jsonify({'error': 'Invalid probability values'}), 500

        high_risk_prob_formatted = f"{high_risk_prob_numeric:.2%}"
        low_risk_prob_formatted = f"{low_risk_prob_numeric:.2%}"

        recommendations = []
        # Generate recommendations based on input data
        if prediction == 1:  # High risk
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
            'high_risk_probability': high_risk_prob_formatted,
            'low_risk_probability': low_risk_prob_formatted,
            'recommendations': recommendations,
            'low_risk_probability_numeric': low_risk_prob_numeric
        }

        # Save prediction to user data
        users_data = load_users()
        user_id = current_user.id
        if user_id in users_data and users_data[user_id]['role'] == 'consumer':
            users_data[user_id].setdefault('predictions', []).append({
                'timestamp': datetime.now().isoformat(),
                'input_data': data,
                'prediction_result': predicted_risk,
                'probabilities': [float(prob) for prob in probs],  # Convert all probabilities to float
                'recommendations': recommendations
            })
            save_users(users_data)

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

@app.route('/create-customer-form')
@login_required
def create_customer_form():
    if current_user.role != 'lender':
        flash('Access denied. Only lenders can create customer accounts.', 'danger')
        return redirect(url_for('index'))
    return render_template('create_customer.html')

@app.route('/create-customer', methods=['POST'])
@login_required
def create_customer():
    if current_user.role != 'lender':
        return jsonify({'error': 'Access denied'}), 403

    name = request.form.get('name')
    email = request.form.get('email')

    if not name or not email:
        return jsonify({'error': 'Name and email are required'}), 400

    # Generate a random password
    password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

    # Create new user
    new_user = {
        'id': str(uuid.uuid4()),
        'name': name,
        'email': email,
        'username': email,  # Using email as username
        'password': generate_password_hash(password),
        'role': 'consumer',
        'created_by': current_user.id,
        'predictions': []
    }

    # Load existing users
    users = load_users()

    # Check if username (email) already exists
    if email in [user.get('username') for user in users.values()]:
         flash('Email address already registered.', 'danger')
         return redirect(url_for('create_customer_form'))

    # Add the new user to the dictionary using their ID as the key
    users[new_user['id']] = new_user

    save_users(users)

    # Redirect to success page
    return render_template('customer_created.html',
                         username=new_user['username'],
                         password=password)

@app.route('/view_customer_prediction/<user_id>')
@login_required
def view_customer_prediction(user_id):
    if current_user.role != 'lender':
        return jsonify({'error': 'Unauthorized'}), 403

    # Use the helper function to get dashboard data for the specified user_id
    dashboard_data = get_consumer_dashboard_data(user_id)

    if dashboard_data is None:
        # Handle case where user data is not found or not a consumer
        return jsonify({'error': 'Customer not found or is not a consumer'}), 404

    # Render the consumer dashboard template, passing the data and a lender view flag
    return render_template('dashboard_consumer.html', data=dashboard_data, is_lender_view=True, user_role=current_user.role)

def get_lender_dashboard_data():
    users_data = load_users()
    consumer_users = [user for user_id, user in users_data.items() if user['role'] == 'consumer']

    # Placeholder data for lender dashboard charts and stats
    data = {
        'total_clients': len(consumer_users),
        'total_assets': 15000000, # Example data
        'avg_risk_score': 720, # Example data
        'client_list': consumer_users, # Pass consumer users to the template
        'risk_distribution_data': {'labels': ['Low Risk', 'Medium Risk', 'High Risk'], 'datasets': [{'data': [60, 30, 10], 'backgroundColor': ['#27ae60', '#f39c12', '#e74c3c']}]}, # Example chart data
        'application_status_data': {'labels': ['Approved', 'Pending', 'Rejected'], 'datasets': [{'data': [75, 15, 10], 'backgroundColor': ['#2563eb', '#f39c12', '#e74c3c']}]}, # Example chart data
        'loan_performance_data': {'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], 'datasets': [{'label': 'Approval Rate', 'data': [65, 70, 72, 75, 78, 80], 'borderColor': '#27ae60', 'fill': False}]}, # Example chart data
        'geographic_distribution_data': {'labels': ['North', 'South', 'East', 'West'], 'datasets': [{'label': 'Clients', 'data': [250, 150, 200, 100], 'backgroundColor': ['#2d6cdf', '#1cc88a', '#f6c23e', '#e74a4b']}]}, # Example chart data
        'recent_applications': [
            {'name': 'John Doe', 'amount': 10000, 'score': 750, 'date': '2023-10-26', 'status': 'Approved'},
            {'name': 'Jane Smith', 'amount': 5000, 'score': 680, 'date': '2023-10-25', 'status': 'Pending'},
        ] # Example data
    }

    return data

if __name__ == '__main__':
    # Create users.json if it doesn't exist
    if not os.path.exists(USERS_FILE):
        save_users({})
        print(f"Created empty {USERS_FILE}")

    app.run(debug=True) 