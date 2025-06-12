from flask import Flask, render_template, redirect, url_for, request, jsonify, flash, get_flashed_messages
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
from clean_alternative_credit_scoring import AlternativeCreditScorer
import pickle
from models import db, User, Prediction # Import db, User, and Prediction from models.py

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///credit_ml.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app) # Initialize db with the app
migrate = Migrate(app, db)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize credit scoring model
model_path = 'models/alternative_credit_scorer.pkl'

# Load the model if it exists, otherwise initialize a new one
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        credit_model = pickle.load(f)
    print("Alternative credit scoring model loaded successfully.")
else:
    credit_model = AlternativeCreditScorer()
    print("Alternative credit scoring model initialized (not loaded from file).")

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create database tables (only if not already created by migrations)
with app.app_context():
    db.create_all() # Creating initial database schema
    
    # Train the credit scoring model if it doesn't exist
    if not os.path.exists('models/credit_scoring_model.pkl'):
        print("Training credit scoring model...")
        # If credit_model has a train_model method, call it here
        # credit_model.train_model()
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
        latest_prediction = sorted(user.predictions, key=lambda x: x.timestamp)[-1]

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
        dashboard_data['high_risk_probability'] = f"{latest_prediction.credit_score / 1000:.2%}"
        dashboard_data['low_risk_probability'] = f"{(1 - latest_prediction.credit_score / 1000):.2%}"
        dashboard_data['low_risk_probability_numeric'] = 1 - (latest_prediction.credit_score / 1000)
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
        return redirect(url_for('consumer_dashboard')) # Redirect if not a lender

    # Get only consumers created by this lender
    consumers = User.query.filter_by(role='consumer', created_by=current_user.id).all()
    
    # Get their latest predictions
    consumer_list = []
    for consumer in consumers:
        latest_prediction = Prediction.query.filter_by(user_id=consumer.id).order_by(Prediction.timestamp.desc()).first()
        consumer_list.append({
            'id': consumer.id,
            'name': consumer.username,
            'email': consumer.email,
            'predictions': latest_prediction
        })
    
    # Get dashboard statistics
    total_clients = len(consumers)
    total_assets = sum(p.loan_amount for p in Prediction.query.filter(Prediction.user_id.in_([c.id for c in consumers])).all() if p.loan_amount is not None)
    avg_risk_score = total_assets / total_clients if total_clients > 0 else 0
    
    # Get recent applications for this lender's consumers
    recent_applications = Prediction.query.filter(
        Prediction.user_id.in_([c.id for c in consumers])
    ).order_by(Prediction.timestamp.desc()).limit(5).all()
    
    # Prepare chart data
    risk_distribution_data = {
        'labels': ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'],
        'datasets': [{
            'data': [0, 0, 0, 0, 0],
            'backgroundColor': ['#28a745', '#20c997', '#007bff', '#ffc107', '#dc3545']
        }]
    }
    
    application_status_data = {
        'labels': ['Approved', 'Pending', 'Rejected'],
        'datasets': [{
            'data': [0, 0, 0],
            'backgroundColor': ['#28a745', '#ffc107', '#dc3545']
        }]
    }
    
    loan_performance_data = {
        'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        'datasets': [{
            'label': 'Loan Amount',
            'data': [0, 0, 0, 0, 0, 0],
            'borderColor': '#007bff',
            'fill': False
        }]
    }
    
    geographic_distribution_data = {
        'labels': ['North', 'South', 'East', 'West', 'Central'],
        'datasets': [{
            'label': 'Number of Loans',
            'data': [0, 0, 0, 0, 0],
            'backgroundColor': '#007bff'
        }]
    }
    
    # Prepare dashboard data
    dashboard_data = {
        'total_clients': total_clients,
        'total_assets': total_assets,
        'avg_risk_score': avg_risk_score,
        'consumer_list': consumer_list,
        'recent_applications': recent_applications,
        'credit_score': 752,  # Default value
        'credit_score_status': 'Good',  # Default value
        'earnings': 6320,  # Default value
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
        'risk_distribution_data': risk_distribution_data,
        'application_status_data': application_status_data,
        'loan_performance_data': loan_performance_data,
        'geographic_distribution_data': geographic_distribution_data
    }

    return render_template('dashboard_lender.html', 
                         data=dashboard_data,
                         user_role=current_user.role,
                         is_lender_view=True)

@app.route('/apply', methods=['GET', 'POST'])
@login_required
def apply():
    if request.method == 'POST':
        # Get all required fields
        required_fields = [
            'loan_amount', 'employment_length', 'annual_income', 'verification_status',
            'gig_platforms_count', 'gig_platform_rating', 'gig_completion_rate',
            'utility_payments_ontime', 'rent_payments_ontime', 'subscription_payments_ontime',
            'months_payment_history', 'late_payments_90d', 'delinq_2yrs', 'pub_rec', 'revol_util',
            'home_ownership', 'bank_balance_avg', 'bank_balance_min', 'investment_assets', 'mort_acc',
            'dti', 'cashflow_ratio', 'savings_rate', 'digital_footprint_score',
            'shopping_categories', 'gambling_expenses', 'education_level', 'open_acc'
        ]
        
        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in request.form]
        if missing_fields:
            flash(f'Missing required fields: {", ".join(missing_fields)}', 'error')
            return redirect(url_for('apply'))
        
        # Convert form data to appropriate types
        data = {
            'loan_amount': float(request.form['loan_amount']),
            'employment_length': float(request.form['employment_length']),
            'annual_income': float(request.form['annual_income']),
            'verification_status': request.form['verification_status'],
            'gig_platforms_count': float(request.form['gig_platforms_count']),
            'gig_platform_rating': float(request.form['gig_platform_rating']),
            'gig_completion_rate': float(request.form['gig_completion_rate']),
            'utility_payments_ontime': float(request.form['utility_payments_ontime']),
            'rent_payments_ontime': float(request.form['rent_payments_ontime']),
            'subscription_payments_ontime': float(request.form['subscription_payments_ontime']),
            'months_payment_history': float(request.form['months_payment_history']),
            'late_payments_90d': float(request.form['late_payments_90d']),
            'delinq_2yrs': float(request.form['delinq_2yrs']),
            'pub_rec': float(request.form['pub_rec']),
            'revol_util': float(request.form['revol_util']),
            'home_ownership': request.form['home_ownership'],
            'bank_balance_avg': float(request.form['bank_balance_avg']),
            'bank_balance_min': float(request.form['bank_balance_min']),
            'investment_assets': float(request.form['investment_assets']),
            'mort_acc': float(request.form['mort_acc']),
            'dti': float(request.form['dti']),
            'cashflow_ratio': float(request.form['cashflow_ratio']),
            'savings_rate': float(request.form['savings_rate']),
            'digital_footprint_score': float(request.form['digital_footprint_score']),
            'shopping_categories': float(request.form['shopping_categories']),
            'gambling_expenses': float(request.form['gambling_expenses']),
            'education_level': float(request.form['education_level']),
            'open_acc': float(request.form['open_acc'])
        }
        
        try:
            # Get prediction from model
            result = credit_model.score_profile(data)
            
            # Create new prediction record
            prediction = Prediction(
                user_id=current_user.id,
                loan_amount=data['loan_amount'],
                employment_length=data['employment_length'],
                annual_income=data['annual_income'],
                verification_status=data['verification_status'],
                gig_platforms_count=data['gig_platforms_count'],
                gig_platform_rating=data['gig_platform_rating'],
                gig_completion_rate=data['gig_completion_rate'],
                utility_payments_ontime=data['utility_payments_ontime'],
                rent_payments_ontime=data['rent_payments_ontime'],
                subscription_payments_ontime=data['subscription_payments_ontime'],
                months_payment_history=data['months_payment_history'],
                late_payments_90d=data['late_payments_90d'],
                delinq_2yrs=data['delinq_2yrs'],
                pub_rec=data['pub_rec'],
                revol_util=data['revol_util'],
                home_ownership=data['home_ownership'],
                bank_balance_avg=data['bank_balance_avg'],
                bank_balance_min=data['bank_balance_min'],
                investment_assets=data['investment_assets'],
                mort_acc=data['mort_acc'],
                dti=data['dti'],
                cashflow_ratio=data['cashflow_ratio'],
                savings_rate=data['savings_rate'],
                digital_footprint_score=data['digital_footprint_score'],
                shopping_categories=data['shopping_categories'],
                gambling_expenses=data['gambling_expenses'],
                education_level=data['education_level'],
                open_acc=data['open_acc'],
                credit_score=result['final_score'],
                grade=result['interpretation']['grade'],
                recommendation=result['interpretation']['recommendation'],
                rate_range=result['interpretation']['rate_range']
            )
            
            db.session.add(prediction)
            db.session.commit()
            
            flash('Credit score assessment completed successfully!', 'success')
            return redirect(url_for('consumer_dashboard'))
            
        except Exception as e:
            flash(f'Error processing prediction: {str(e)}', 'error')
            return redirect(url_for('apply'))
            
    return render_template('apply.html')

@app.route('/view_prediction/<int:user_id>')
@login_required
def view_customer_prediction(user_id):
    user = User.query.get_or_404(user_id)
    prediction = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.timestamp.desc()).first()
    
    if not prediction:
        flash('No prediction found for this user.', 'warning')
        return redirect(url_for('lender_dashboard'))

    # Convert Prediction object to a dictionary for easier display
    prediction_data = {
        'loan_amount': prediction.loan_amount,
        'employment_length': prediction.employment_length,
        'annual_income': prediction.annual_income,
        'verification_status': prediction.verification_status,
        'gig_platforms_count': prediction.gig_platforms_count,
        'gig_platform_rating': prediction.gig_platform_rating,
        'gig_completion_rate': prediction.gig_completion_rate,
        'utility_payments_ontime': prediction.utility_payments_ontime,
        'rent_payments_ontime': prediction.rent_payments_ontime,
        'subscription_payments_ontime': prediction.subscription_payments_ontime,
        'months_payment_history': prediction.months_payment_history,
        'late_payments_90d': prediction.late_payments_90d,
        'delinq_2yrs': prediction.delinq_2yrs,
        'pub_rec': prediction.pub_rec,
        'revol_util': prediction.revol_util,
        'home_ownership': prediction.home_ownership,
        'bank_balance_avg': prediction.bank_balance_avg,
        'bank_balance_min': prediction.bank_balance_min,
        'investment_assets': prediction.investment_assets,
        'mort_acc': prediction.mort_acc,
        'dti': prediction.dti,
        'cashflow_ratio': prediction.cashflow_ratio,
        'savings_rate': prediction.savings_rate,
        'digital_footprint_score': prediction.digital_footprint_score,
        'shopping_categories': prediction.shopping_categories,
        'gambling_expenses': prediction.gambling_expenses,
        'education_level': prediction.education_level,
        'open_acc': prediction.open_acc,
        'credit_score': prediction.credit_score,
        'grade': prediction.grade,
        'recommendation': prediction.recommendation,
        'rate_range': prediction.rate_range,
        'timestamp': prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return render_template('view_prediction.html', user=user, prediction=prediction_data)

@app.route('/create_customer', methods=['GET', 'POST'])
@login_required
def create_customer_form():
    if current_user.role != 'lender':
        flash('Only lenders can create new customers.', 'warning')
        return redirect(url_for('lender_dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
            return redirect(url_for('create_customer_form'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return redirect(url_for('create_customer_form'))

        new_consumer = User(
            username=username,
            name=name,
            email=email,
            phone=phone,
            role='consumer',
            created_by=current_user.id # Assign the current lender as the creator
        )
        new_consumer.set_password(password)
        db.session.add(new_consumer)
        db.session.commit()

        flash(f'Consumer {username} created successfully!', 'success')
        return redirect(url_for('lender_dashboard'))

    return render_template('create_customer.html', user_role=current_user.role)

if __name__ == '__main__':
    app.run(debug=True) 