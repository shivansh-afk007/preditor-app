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
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///credit_ml.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PREFERRED_URL_SCHEME'] = 'https'

# Configure SQLite to use a compatible version
from sqlalchemy import event
from sqlalchemy.engine import Engine
import sqlite3

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

# Remove HTTPS redirection
# @app.before_request
# def before_request():
#     if not request.is_secure and os.environ.get('FLASK_ENV') == 'production':
#         url = request.url.replace('http://', 'https://', 1)
#         return redirect(url, code=301)

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
        if current_user.role == 'consumer':
            return redirect(url_for('consumer_dashboard'))
        elif current_user.role == 'lender':
            return redirect(url_for('lender_dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        role = request.form.get('role')

        if get_user_by_username(username):
            flash('Username already exists')
            return render_template('register.html')

        if get_user_by_email(email):
            flash('Email already exists')
            return render_template('register.html')

        user = User(username=username, name=name, email=email, phone=phone, role=role)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Registration successful! Please login.')
        return redirect(url_for('login'))

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
        flash('Access denied. Consumer dashboard only.')
        return redirect(url_for('index'))

    dashboard_data = get_consumer_dashboard_data(current_user.id)
    return render_template('consumer_dashboard.html', dashboard_data=dashboard_data)

@app.route('/dashboard/lender')
@login_required
def lender_dashboard():
    if current_user.role != 'lender':
        flash('Access denied. Lender dashboard only.')
        return redirect(url_for('index'))

    # Get all consumers created by this lender
    consumers = User.query.filter_by(role='consumer', created_by=current_user.id).all()
    
    # Get recent predictions for these consumers
    recent_predictions = []
    for consumer in consumers:
        if consumer.predictions:
            latest_prediction = sorted(consumer.predictions, key=lambda x: x.created_at)[-1]
            recent_predictions.append({
                'consumer': consumer,
                'prediction': latest_prediction
            })
    
    # Sort by prediction date
    recent_predictions.sort(key=lambda x: x['prediction'].created_at, reverse=True)
    
    return render_template('lender_dashboard.html', consumers=consumers, recent_predictions=recent_predictions)

@app.route('/apply', methods=['GET', 'POST'])
@login_required
def apply():
    if current_user.role != 'consumer':
        flash('Access denied. Consumer only.')
        return redirect(url_for('index'))

    if request.method == 'POST':
        # Get form data
        loan_amnt = float(request.form.get('loan_amnt'))
        emp_length = int(request.form.get('emp_length'))
        annual_inc = float(request.form.get('annual_inc'))
        verification_status = int(request.form.get('verification_status'))
        delinq_2yrs = int(request.form.get('delinq_2yrs'))
        pub_rec = int(request.form.get('pub_rec'))
        revol_util = float(request.form.get('revol_util'))
        home_ownership = request.form.get('home_ownership')
        mort_acc = int(request.form.get('mort_acc'))
        dti = float(request.form.get('dti'))
        open_acc = int(request.form.get('open_acc'))
        total_acc = int(request.form.get('total_acc'))
        inq_last_6mths = int(request.form.get('inq_last_6mths'))

        # Prepare input data
        input_data = pd.DataFrame({
            'loan_amnt': [loan_amnt],
            'emp_length': [emp_length],
            'annual_inc': [annual_inc],
            'verification_status': [verification_status],
            'delinq_2yrs': [delinq_2yrs],
            'pub_rec': [pub_rec],
            'revol_util': [revol_util],
            'home_ownership': [home_ownership],
            'mort_acc': [mort_acc],
            'dti': [dti],
            'open_acc': [open_acc],
            'total_acc': [total_acc],
            'inq_last_6mths': [inq_last_6mths]
        })

        # Get prediction
        prediction_result = credit_model.predict(input_data)
        
        # Create prediction record
        prediction = Prediction(
            user_id=current_user.id,
            credit_score=prediction_result['credit_score'],
            credit_grade=prediction_result['credit_grade'],
            default_probability=prediction_result['default_probability'],
            recommendation=prediction_result['recommendation'],
            rate_range=prediction_result['rate_range'],
            loan_amnt=loan_amnt,
            emp_length=emp_length,
            annual_inc=annual_inc,
            verification_status=verification_status,
            delinq_2yrs=delinq_2yrs,
            pub_rec=pub_rec,
            revol_util=revol_util,
            home_ownership=home_ownership,
            mort_acc=mort_acc,
            dti=dti,
            open_acc=open_acc,
            total_acc=total_acc,
            inq_last_6mths=inq_last_6mths,
            breakdown=prediction_result.get('breakdown', {})
        )
        
        db.session.add(prediction)
        db.session.commit()

        flash('Application submitted successfully!')
        return redirect(url_for('consumer_dashboard'))

    return render_template('apply.html')

@app.route('/view_prediction/<int:user_id>')
@login_required
def view_customer_prediction(user_id):
    if current_user.role != 'lender':
        flash('Access denied. Lender only.')
        return redirect(url_for('index'))

    user = User.query.get_or_404(user_id)
    if user.role != 'consumer' or user.created_by != current_user.id:
        flash('Access denied. Not your customer.')
        return redirect(url_for('lender_dashboard'))

    latest_prediction = None
    if user.predictions:
        latest_prediction = sorted(user.predictions, key=lambda x: x.created_at)[-1]

    return render_template('view_prediction.html', user=user, prediction=latest_prediction)

@app.route('/create_customer', methods=['GET', 'POST'])
@login_required
def create_customer_form():
    if current_user.role != 'lender':
        flash('Access denied. Lender only.')
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')

        if get_user_by_username(username):
            flash('Username already exists')
            return render_template('create_customer.html')

        if get_user_by_email(email):
            flash('Email already exists')
            return render_template('create_customer.html')

        user = User(
            username=username,
            name=name,
            email=email,
            phone=phone,
            role='consumer',
            created_by=current_user.id
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Customer created successfully!')
        return redirect(url_for('lender_dashboard'))

    return render_template('create_customer.html')

if __name__ == '__main__':
    app.run(debug=True)