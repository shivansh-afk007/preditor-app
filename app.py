from flask import Flask, render_template, redirect, url_for, request, jsonify, flash, get_flashed_messages
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
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from credit_scoring_model import CreditScoringModel

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///credit_ml.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize credit scoring model
credit_model = CreditScoringModel()

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(20), nullable=False)  # 'lender' or 'consumer'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    application_date = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Prediction model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    score = db.Column(db.Integer, nullable=False)
    grade = db.Column(db.String(2), nullable=False)
    default_probability = db.Column(db.Float, nullable=False)
    recommendation = db.Column(db.String(50), nullable=False)
    rate_range = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Component scores
    income_stability = db.Column(db.String(2))
    payment_consistency = db.Column(db.String(2))
    asset_profile = db.Column(db.String(2))
    behavioral_factors = db.Column(db.String(2))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create database tables
with app.app_context():
    db.create_all()
    
    # Train the credit scoring model if it doesn't exist
    if not os.path.exists('models/credit_scoring_model.pkl'):
        print("Training credit scoring model...")
        credit_model.train_model()
        print("Model training completed.")

# --- Jinja2 Filters ---
@app.template_filter('format_currency')
def format_currency_filter(value):
    if value is None:
        return "N/A"
    # Format as currency (e.g., $1,234)
    return f"{value:,.0f}"

# --- User Management Functions ---
def get_user_by_username(username):
    return User.query.filter_by(username=username).first()

def get_user_by_email(email):
    return User.query.filter_by(email=email).first()

# --- Data Preparation Function ---
def get_consumer_dashboard_data(user_id):
    user = User.query.get(user_id)
    if not user or user.role != 'consumer':
        return None

    latest_prediction = None
    if user.predictions:
        latest_prediction = sorted(user.predictions, key=lambda x: x.created_at)[-1]

    # Initialize Dashboard Data with default values
    dashboard_data = {
        'credit_score': 752,
        'credit_score_status': 'Good',
        'earnings': 6320,
        'credit_factors': {
            'Income': 'N/A',
            'Cash Flow': 'N/A',
            'Employment': 'N/A'
        },
        'recent_activity': [],
        'prediction': 'No prediction yet.',
        'high_risk_probability': 'N/A',
        'low_risk_probability': 'N/A',
        'low_risk_probability_numeric': 0.5,
        'recommendations': ['Submit the form above to get your first prediction!'],
        'spending_data': {'labels': [], 'data': [], 'backgroundColor': []},
        'credit_utilization_data': {'labels': [], 'data': [], 'backgroundColor': []},
        'loan_applications': [],
    }

    if latest_prediction:
        dashboard_data['prediction'] = latest_prediction.recommendation
        dashboard_data['high_risk_probability'] = f"{latest_prediction.default_probability:.2%}"
        dashboard_data['low_risk_probability'] = f"{(1 - latest_prediction.default_probability):.2%}"
        dashboard_data['low_risk_probability_numeric'] = 1 - latest_prediction.default_probability
        dashboard_data['recommendations'] = [latest_prediction.recommendation]
        
        # Update credit score based on prediction
        prob = dashboard_data['low_risk_probability_numeric']
        if prob > 0.8: dashboard_data['credit_score'] = 780
        elif prob > 0.6: dashboard_data['credit_score'] = 720
        elif prob > 0.4: dashboard_data['credit_score'] = 630
        else: dashboard_data['credit_score'] = 500

    # Determine credit score status
    score = dashboard_data['credit_score']
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
        if current_user.role == 'consumer':
            return redirect(url_for('consumer_dashboard'))
        elif current_user.role == 'lender':
            return redirect(url_for('lender_dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
                login_user(user)
                if user.role == 'consumer':
                    return redirect(url_for('consumer_dashboard'))
                elif user.role == 'lender':
                    return redirect(url_for('lender_dashboard'))
        else:
            flash('Invalid username or password')

    # Clear flashed messages specifically for GET requests to the login page
    # This prevents messages from redirects (like successful registration) from appearing
    if request.method == 'GET':
        # Access and discard messages without displaying them
        _ = get_flashed_messages()

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        password = request.form.get('password')
        role = request.form.get('role') # Get role from form ('consumer' or 'lender')

        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))

        user = User(username=username, name=name, email=email, phone=phone, role=role)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

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

    # Get all consumer users
    consumers = User.query.filter_by(role='consumer').all()
    
    # Get their latest predictions
    consumer_list = []
    for consumer in consumers:
        latest_prediction = Prediction.query.filter_by(user_id=consumer.id).order_by(Prediction.created_at.desc()).first()
        consumer_list.append({
            'id': consumer.id,
            'name': consumer.username,
            'email': consumer.email,
            'predictions': latest_prediction
        })
    
    # Get dashboard statistics
    total_clients = len(consumers)
    total_assets = sum(p.score for p in Prediction.query.all() if p.score is not None)
    avg_risk_score = total_assets / total_clients if total_clients > 0 else 0
    
    # Get recent applications
    recent_applications = Prediction.query.order_by(Prediction.created_at.desc()).limit(5).all()
    
    # Prepare chart data
    risk_distribution_data = {
        'labels': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'datasets': [{
            'data': [20, 30, 25, 15, 5, 3, 2],
            'backgroundColor': ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#FF5722', '#F44336', '#D32F2F']
        }]
    }
    
    application_status_data = {
        'labels': ['Approved', 'Conditionally Approved', 'Denied'],
        'datasets': [{
            'data': [60, 25, 15],
            'backgroundColor': ['#4CAF50', '#FFC107', '#F44336']
        }]
    }
    
    loan_performance_data = {
        'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'datasets': [{
            'label': 'Performance',
            'data': [65, 70, 75, 72, 78, 80],
            'borderColor': '#2196F3',
            'fill': False
        }]
    }
    
    geographic_distribution_data = {
        'labels': ['North', 'South', 'East', 'West', 'Central'],
        'datasets': [{
            'data': [25, 20, 15, 20, 20],
            'backgroundColor': ['#2196F3', '#4CAF50', '#FFC107', '#FF9800', '#9C27B0']
        }]
    }
    
    return render_template('dashboard_lender.html',
                         data={
                             'total_clients': total_clients,
                             'total_assets': f"{total_assets:,.2f}",
                             'avg_risk_score': f"{avg_risk_score:.2f}",
                             'client_list': consumer_list,
                             'recent_applications': recent_applications,
                             'risk_distribution_data': risk_distribution_data,
                             'application_status_data': application_status_data,
                             'loan_performance_data': loan_performance_data,
                             'geographic_distribution_data': geographic_distribution_data
                         })

@app.route('/apply', methods=['GET', 'POST'])
@login_required
def apply():
    if current_user.role != 'consumer':
        flash('Access denied')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        try:
            # Get form data
            emp_length_str = request.form.get('emp_length')
            # Convert employment length from string to numeric
            emp_length_map = {
                '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
                '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
                '8 years': 8, '9 years': 9, '10+ years': 10
            }
            emp_length = emp_length_map.get(emp_length_str, 0)
            
            # Convert verification status from string to numeric
            verification_status_str = request.form.get('verification_status')
            verification_map = {
                'Not Verified': 0,
                'Verified': 1,
                'Source Verified': 2
            }
            verification_status = verification_map.get(verification_status_str, 0)
            
            input_data = {
                'emp_length': emp_length,
                'annual_inc': float(request.form.get('annual_inc')),
                'verification_status': verification_status,
                'delinq_2yrs': int(request.form.get('delinq_2yrs')),
                'pub_rec': int(request.form.get('pub_rec')),
                'revol_util': float(request.form.get('revol_util')),
                'home_ownership': request.form.get('home_ownership'),
                'mort_acc': int(request.form.get('mort_acc')),
                'dti': float(request.form.get('dti')),
                'open_acc': int(request.form.get('open_acc')),
                'total_acc': int(request.form.get('total_acc')),
                'inq_last_6mths': int(request.form.get('inq_last_6mths'))
            }
            
            print("Input data:", input_data)  # Debug log
            
            # Get prediction from model
            result = credit_model.predict_credit_score(input_data)
            print("Prediction result:", result)  # Debug log
            
            # Save prediction to database
            prediction = Prediction(
                user_id=current_user.id,
                score=result['score'],
                grade=result['grade'],
                default_probability=result['default_probability'],
                recommendation=result['recommendation'],
                rate_range=result['rate_range'],
                income_stability=result['breakdown']['income_stability'],
                payment_consistency=result['breakdown']['payment_consistency'],
                asset_profile=result['breakdown']['asset_profile'],
                behavioral_factors=result['breakdown']['behavioral_factors']
            )
            
            db.session.add(prediction)
            db.session.commit()
            print("Prediction saved to database")  # Debug log
            
            flash('Application submitted successfully')
            return redirect(url_for('consumer_dashboard'))

        except Exception as e:
            print("Error in apply route:", str(e))  # Debug log
            flash(f'Error processing application: {str(e)}')
            return redirect(url_for('apply'))
    
    return render_template('apply.html')

@app.route('/view_prediction/<int:user_id>')
@login_required
def view_customer_prediction(user_id):
    if current_user.role != 'lender':
        flash('Access denied')
        return redirect(url_for('index'))
    
    user = User.query.get_or_404(user_id)
    prediction = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.created_at.desc()).first()
    
    if not prediction:
        flash('No prediction found for this user')
        return redirect(url_for('lender_dashboard'))
    
    return render_template('view_prediction.html', customer=user, prediction=prediction)

@app.route('/create_customer', methods=['GET', 'POST'])
@login_required
def create_customer_form():
    if current_user.role != 'lender':
        flash('Access denied')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('create_customer_form'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('create_customer_form'))
        
        user = User(username=username, name=name, email=email, phone=phone, role='consumer')
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Customer created successfully')
        return redirect(url_for('lender_dashboard'))
    
    return render_template('create_customer.html')

if __name__ == '__main__':
    app.run(debug=True) 