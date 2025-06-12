from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(20), nullable=False, default='consumer')  # 'consumer' or 'lender'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    application_date = db.Column(db.DateTime, default=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # ID of the lender who created this user
    
    # Relationships
    predictions = db.relationship('Prediction', back_populates='user', lazy=True)
    creator = db.relationship('User', remote_side=[id], backref=db.backref('created_consumers', lazy=True))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Input features
    loan_amount = db.Column(db.Float, nullable=False)
    employment_length = db.Column(db.Float, nullable=False)
    annual_income = db.Column(db.Float, nullable=False)
    verification_status = db.Column(db.String(20), nullable=False)
    gig_platforms_count = db.Column(db.Float, nullable=False)
    gig_platform_rating = db.Column(db.Float, nullable=False)
    gig_completion_rate = db.Column(db.Float, nullable=False)
    utility_payments_ontime = db.Column(db.Float, nullable=False)
    rent_payments_ontime = db.Column(db.Float, nullable=False)
    subscription_payments_ontime = db.Column(db.Float, nullable=False)
    months_payment_history = db.Column(db.Float, nullable=False)
    late_payments_90d = db.Column(db.Float, nullable=False)
    delinq_2yrs = db.Column(db.Float, nullable=False)
    pub_rec = db.Column(db.Float, nullable=False)
    revol_util = db.Column(db.Float, nullable=False)
    home_ownership = db.Column(db.String(20), nullable=False)
    bank_balance_avg = db.Column(db.Float, nullable=False)
    bank_balance_min = db.Column(db.Float, nullable=False)
    investment_assets = db.Column(db.Float, nullable=False)
    mort_acc = db.Column(db.Float, nullable=False)
    dti = db.Column(db.Float, nullable=False)
    cashflow_ratio = db.Column(db.Float, nullable=False)
    savings_rate = db.Column(db.Float, nullable=False)
    digital_footprint_score = db.Column(db.Float, nullable=False)
    shopping_categories = db.Column(db.Float, nullable=False)
    gambling_expenses = db.Column(db.Float, nullable=False)
    education_level = db.Column(db.Float, nullable=False)
    open_acc = db.Column(db.Float, nullable=False)
    
    # Output scores
    credit_score = db.Column(db.Integer, nullable=False)
    grade = db.Column(db.String(2), nullable=False)
    recommendation = db.Column(db.String(100), nullable=False)
    rate_range = db.Column(db.String(20), nullable=False)
    
    # Relationships
    user = db.relationship('User', back_populates='predictions', lazy=True) 