from flask import Flask, render_template, redirect, url_for, request, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import joblib
import pandas as pd
import os
import json # Using json for simple user data storage for now

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
    return redirect(url_for('login'))

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
        'credit_score': 752, # Example data
        'earnings': 6320, # Example data
        'credit_factors': {
            'Income': 'High',
            'Cash Flow': 'Very Good',
            'Employment': 'Stable'
        },
        'recent_activity': [
            {'date': 'Apr 22', 'description': 'Direct deposit', 'amount': '+3160.00'},
            {'date': 'Apr 17', 'description': 'Grover Energy', 'amount': '-150.00'}
        ],
        # Add more data for charts, etc.
    }
    return render_template('dashboard_consumer.html', data=dashboard_data)

@app.route('/dashboard/lender')
@login_required
def lender_dashboard():
    if current_user.role != 'lender':
        return "Unauthorized", 403 # Or redirect to their correct dashboard
    # --- Lender Dashboard Data Mockup ---
    dashboard_data = {
        'total_clients': 150,
        'total_assets': '5.2M',
        'avg_risk_score': 680,
        # Add more data for client list, charts, etc.
    }
    return render_template('dashboard_lender.html', data=dashboard_data)

# --- Credit Risk Prediction (Integrate existing logic, modify for web) ---
# Load the model (assuming 'models/credit_risk_model.joblib' exists)
try:
    model = joblib.load('models/credit_risk_model.joblib')
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
        # Get data from the form (assuming similar form fields as before)
        data = {
            'Age': float(request.form.get('age')),
            'Income': float(request.form.get('income')),
            'LoanAmount': float(request.form.get('loan_amount')),
            'LoanTerm': float(request.form.get('loan_term')),
            'CreditScore': float(request.form.get('credit_score')),
            'EmploymentStatus': request.form.get('employment_status'),
            'Education': request.form.get('education'),
            'MaritalStatus': request.form.get('marital_status')
        }

        # Ensure all required fields are present and valid
        # (More robust validation needed for production)
        if any(v is None for v in data.values()):
             return jsonify({'error': 'Missing form data'}), 400

        # Convert to DataFrame (handle categorical features - requires model preprocessing logic)
        # This part needs to match exactly how your model expects the input DataFrame
        # For a simple demo, assume model handles one-hot encoding internally or input is pre-encoded
        # A more realistic approach involves saving/loading the scaler and encoder used during training
        # For now, let's assume raw input works for the loaded model
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
        # Log the error in a real application
        print(f"Prediction Error: {e}")
        return jsonify({'error': 'An error occurred during prediction. Please check inputs.'}), 500


if __name__ == '__main__':
    # Create users.json if it doesn't exist
    if not os.path.exists(USERS_FILE):
        save_users({})
        print(f"Created empty {USERS_FILE}")

    app.run(debug=True) 