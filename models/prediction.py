class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    score = db.Column(db.Float, nullable=False)
    grade = db.Column(db.String(1), nullable=False)
    default_probability = db.Column(db.Float, nullable=False)
    recommendation = db.Column(db.String(50), nullable=False)
    rate_range = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    # New fields for detailed profile
    loan_amount = db.Column(db.Float, nullable=False)
    employment_length = db.Column(db.Integer, nullable=False)
    annual_income = db.Column(db.Float, nullable=False)
    verification_status = db.Column(db.Integer, nullable=False)
    delinquencies_2yrs = db.Column(db.Integer, nullable=False)
    public_records = db.Column(db.Integer, nullable=False)
    revolving_utilization = db.Column(db.Float, nullable=False)
    home_ownership = db.Column(db.String(20), nullable=False)
    mortgage_accounts = db.Column(db.Integer, nullable=False)
    debt_to_income = db.Column(db.Float, nullable=False)
    open_accounts = db.Column(db.Integer, nullable=False)
    total_accounts = db.Column(db.Integer, nullable=False)
    inquiries_6mths = db.Column(db.Integer, nullable=False)
    
    # Additional analysis fields
    income_stability = db.Column(db.String(20))
    payment_consistency = db.Column(db.String(20))
    asset_profile = db.Column(db.String(20))
    behavioral_factors = db.Column(db.String(20))
    
    user = db.relationship('User', backref=db.backref('predictions', lazy=True)) 