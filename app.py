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
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # ID of the lender who created this user
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    created_consumers = db.relationship('User', backref=db.backref('creator', remote_side=[id]), lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Prediction model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    credit_score = db.Column(db.Integer, nullable=False)
    credit_grade = db.Column(db.String(2), nullable=False)
    default_probability = db.Column(db.Float, nullable=False)
    recommendation = db.Column(db.String(50), nullable=False)
    rate_range = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    # Input features
    loan_amnt = db.Column(db.Float, nullable=False)
    emp_length = db.Column(db.Integer, nullable=False)
    annual_inc = db.Column(db.Float, nullable=False)
    verification_status = db.Column(db.Integer, nullable=False)
    delinq_2yrs = db.Column(db.Integer, nullable=False)
    pub_rec = db.Column(db.Integer, nullable=False)
    revol_util = db.Column(db.Float, nullable=False)
    home_ownership = db.Column(db.String(20), nullable=False)
    mort_acc = db.Column(db.Integer, nullable=False)
    dti = db.Column(db.Float, nullable=False)
    open_acc = db.Column(db.Integer, nullable=False)
    total_acc = db.Column(db.Integer, nullable=False)
    inq_last_6mths = db.Column(db.Integer, nullable=False)
    breakdown = db.Column(db.JSON, nullable=True)

    def __repr__(self):
        return f"Prediction(id={self.id}, user_id={self.user_id}, credit_score={self.credit_score}, credit_grade={self.credit_grade}, default_probability={self.default_probability}, recommendation={self.recommendation}, rate_range={self.rate_range})"

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

    # Get only consumers created by this lender
    consumers = User.query.filter_by(role='consumer', created_by=current_user.id).all()
    
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
    total_assets = sum(p.loan_amnt for p in Prediction.query.filter(Prediction.user_id.in_([c.id for c in consumers])).all() if p.loan_amnt is not None)
    avg_risk_score = total_assets / total_clients if total_clients > 0 else 0
    
    # Get recent applications for this lender's consumers
    recent_applications = Prediction.query.filter(
        Prediction.user_id.in_([c.id for c in consumers])
    ).order_by(Prediction.created_at.desc()).limit(5).all()
    
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
    if request.method == 'POST':
        # List of required fields and their expected types
        required_fields = {
            'loan_amnt': float,
            'emp_length': int,
            'annual_inc': float,
            'verification_status': int,
            'delinq_2yrs': int,
            'pub_rec': int,
            'revol_util': float,
            'home_ownership': str,
            'mort_acc': int,
            'dti': float,
            'open_acc': int,
            'total_acc': int,
            'inq_last_6mths': int
        }
        
        # Check for missing fields
        missing_fields = []
        for field, field_type in required_fields.items():
            if field not in request.form or not request.form.get(field):
                missing_fields.append(field)
        
        if missing_fields:
            flash(f'Missing required fields: {", ".join(missing_fields)}', 'error')
            return redirect(url_for('apply'))
        
        try:
            # Get and validate form data
            input_data = {}
            for field, field_type in required_fields.items():
                try:
                    value = request.form.get(field)
                    if field_type == float:
                        input_data[field] = float(value)
                    elif field_type == int:
                        input_data[field] = int(value)
                    else:
                        input_data[field] = value
                except ValueError:
                    flash(f'Invalid value for {field}. Expected {field_type.__name__}.', 'error')
                    return redirect(url_for('apply'))
            
            # Validate home_ownership value
            valid_home_ownership = ['RENT', 'MORTGAGE', 'OWN', 'OTHER']
            if input_data['home_ownership'] not in valid_home_ownership:
                flash('Invalid home ownership value', 'error')
                return redirect(url_for('apply'))
            
            # Encode home_ownership as integer
            home_ownership_map = {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'OTHER': 3}
            input_data['home_ownership'] = home_ownership_map.get(input_data['home_ownership'], 0)
            
            # Get prediction
            prediction_result = credit_model.predict_credit_score(input_data)
            
            # Create prediction record
            prediction = Prediction(
                user_id=current_user.id,
                credit_score=prediction_result['credit_score'],
                credit_grade=prediction_result['credit_grade'],
                default_probability=prediction_result['default_probability'],
                recommendation=prediction_result['recommendation'],
                rate_range=prediction_result['rate_range'],
                breakdown=prediction_result['breakdown'],
                **input_data
            )
            
            db.session.add(prediction)
            db.session.commit()
            
            flash('Credit score prediction completed successfully!', 'success')
            return redirect(url_for('consumer_dashboard'))
            
        except Exception as e:
            flash(f'Error processing prediction: {str(e)}', 'error')
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
        
        user = User(
            username=username,
            name=name,
            email=email,
            phone=phone,
            role='consumer',
            created_by=current_user.id  # Set the creator to the current lender
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Customer created successfully')
        return redirect(url_for('lender_dashboard'))
    
    return render_template('create_customer.html')

if __name__ == '__main__':
    app.run(debug=True) 