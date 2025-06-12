import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class CreditScoringModel:
    def __init__(self, model_path='models/alternative_credit_scorer.pkl'):
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
            
    def predict_credit_score(self, input_data):
        if not self.model:
            raise ValueError("Model not loaded")
            
        # Convert input data to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
            
        # Ensure all required features are present
        required_features = [
            'loan_amount', 'employment_length', 'annual_income', 'verification_status',
            'home_ownership', 'delinq_2yrs', 'pub_rec', 'revol_util', 'dti',
            'gig_platforms_count', 'gig_platform_rating', 'gig_completion_rate'
        ]
        
        # Check for missing features
        missing_features = [f for f in required_features if f not in input_data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {', '.join(missing_features)}")
            
        # Select only the required features
        input_data = input_data[required_features]
        
        # Convert home_ownership to string type for proper encoding
        input_data['home_ownership'] = input_data['home_ownership'].astype(str)
        
        # Get prediction using the alternative credit scorer
        prediction_result = self.model.score_profile(input_data.iloc[0])
        
        return {
            'credit_score': prediction_result['final_score'],
            'credit_grade': prediction_result['interpretation']['grade'],
            'default_probability': 1 - (prediction_result['final_score'] / 1000),  # Convert score to probability
            'recommendation': prediction_result['interpretation']['recommendation'],
            'rate_range': prediction_result['interpretation']['rate_range'],
            'breakdown': {
                'income_stability': prediction_result['category_scores']['income_stability'],
                'payment_consistency': prediction_result['category_scores']['payment_consistency'],
                'asset_value': prediction_result['category_scores']['asset_value'],
                'behavioral_factors': prediction_result['category_scores']['behavioral_factors']
            }
        }

    def create_sample_data(self, n_samples=5000):
        """Create sample data for training"""
        np.random.seed(42)
        
        data = {
            'emp_length': np.random.choice(['< 1 year', '1 year', '2 years', '3 years', '4 years', 
                                          '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'], n_samples),
            'annual_inc': np.random.uniform(20000, 200000, n_samples),
            'verification_status': np.random.choice(['Verified', 'Source Verified', 'Not Verified'], n_samples),
            'delinq_2yrs': np.random.choice([0, 1, 2], n_samples, p=[0.8, 0.15, 0.05]),
            'pub_rec': np.random.choice([0, 1, 2], n_samples, p=[0.85, 0.1, 0.05]),
            'revol_util': np.random.uniform(0, 100, n_samples),
            'home_ownership': np.random.choice(['RENT', 'MORTGAGE', 'OWN', 'OTHER'], n_samples),
            'mort_acc': np.random.choice([0, 1, 2, 3], n_samples),
            'dti': np.random.uniform(0, 30, n_samples),
            'open_acc': np.random.randint(1, 20, n_samples),
            'total_acc': np.random.randint(1, 50, n_samples),
            'inq_last_6mths': np.random.choice([0, 1, 2, 3, 4], n_samples),
            'loan_amount': np.random.uniform(1000, 50000, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable (1 = default, 0 = no default)
        conditions = [
            (df['emp_length'].isin(['< 1 year', '1 year'])) & (df['dti'] > 20),
            (df['pub_rec'] > 0) & (df['revol_util'] > 80),
            (df['delinq_2yrs'] > 0) & (df['inq_last_6mths'] > 2)
        ]
        df['is_default'] = np.where(np.any(conditions, axis=0), 1, 0)
        
        return df

    def preprocess_data(self, df):
        """Preprocess the data for modeling"""
        # Create a copy to avoid modifying the original dataframe
        data = df.copy()
        
        # Define target variable
        y = data['is_default']
        data = data.drop('is_default', axis=1)
        
        # Select features
        selected_features = [
            'loan_amnt',
            'emp_length',
            'annual_inc',
            'verification_status',
            'delinq_2yrs',
            'pub_rec',
            'revol_util',
            'home_ownership',
            'mort_acc',
            'dti',
            'open_acc',
            'total_acc',
            'inq_last_6mths'
        ]
        
        # Filter features that exist in the dataset
        existing_features = [f for f in selected_features if f in data.columns]
        X = data[existing_features].copy()
        
        # Handle employment length
        if 'emp_length' in X.columns:
            X['emp_length'] = X['emp_length'].astype(int)
        
        # Handle verification status
        if 'verification_status' in X.columns:
            X['verification_status'] = X['verification_status'].astype(int)
        
        return X, y

    def build_preprocessing_pipeline(self, X):
        """Build preprocessing pipeline"""
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor

    def train_model(self):
        """Train the credit scoring model"""
        print("Training credit scoring model...")
        
        # Create sample data
        try:
            df = pd.read_csv('dummy_credit_data.csv')
            print("Loaded data from dummy_credit_data.csv")
        except FileNotFoundError:
            print("dummy_credit_data.csv not found. Generating sample data instead.")
            df = self.create_sample_data()
        
        # Preprocess data
        X, y = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build preprocessing pipeline
        self.preprocessor = self.build_preprocessing_pipeline(X)
        
        # Create model pipeline
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('classifier', xgb.XGBClassifier(n_estimators=100, random_state=42))
        ])
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Save feature names
        self.feature_names = X.columns.tolist()
        
        # Save model
        os.makedirs('models', exist_ok=True)
        self.save_model()
        
        return self.model

    def save_model(self):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'pipeline': self.model,
            'feature_names': self.feature_names
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {self.model_path}")

    def train(self, X, y):
        """Train the model on the provided data."""
        # Build preprocessing pipeline
        self.preprocessor = self.build_preprocessing_pipeline(X)
        
        # Fit preprocessor
        X_processed = self.preprocessor.fit_transform(X)
        
        # Train XGBoost model
        self.model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X_processed, y)
        
        return self.model 