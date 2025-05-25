import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import pickle
import os
from xgboost import XGBClassifier

class CreditScoringModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.model_path = 'xgboost.pkl'
        
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

    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Handle both pipeline and direct model cases
        if isinstance(model_data, dict) and 'pipeline' in model_data:
            self.model = model_data['pipeline']
        else:
            # If it's just the model, create a pipeline
            self.model = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('classifier', model_data)
            ])
        
        # Force use of only the 13 core features
        self.feature_names = [
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
        return self.model

    def predict_credit_score(self, input_data):
        """Predict credit score for new data"""
        if self.model is None:
            self.load_model()
        
        # Convert input to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Ensure all required features are present
        required_features = [
            'loan_amnt', 'emp_length', 'annual_inc', 'verification_status',
            'delinq_2yrs', 'pub_rec', 'revol_util', 'home_ownership',
            'mort_acc', 'dti', 'open_acc', 'total_acc', 'inq_last_6mths'
        ]
        
        # Check for missing features
        missing_features = [f for f in required_features if f not in input_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select only the required features
        input_df = input_df[required_features]
        
        # Make prediction using the pipeline
        default_probability = float(self.model.predict_proba(input_df)[0, 1])
        
        # Calculate credit score (inverse of default probability)
        credit_score = int(1000 - (default_probability * 1000))
        credit_score = max(0, min(credit_score, 1000))
        
        # Get grade
        credit_grade = self.grade_from_score(credit_score)
        
        # Determine recommendation and rate range
        if credit_score >= 700:
            recommendation = "Strong application. Recommend approval with competitive rates."
            rate_range = "5.0% - 7.5%"
        elif credit_score >= 600:
            recommendation = "Moderate risk. Consider approval with standard rates."
            rate_range = "7.5% - 12.0%"
        else:
            recommendation = "High risk. Consider rejection or high-risk rates."
            rate_range = "12.0% - 18.0%"
        
        # Calculate breakdown scores
        breakdown = {
            'income_stability': float(input_df['emp_length'].iloc[0]) / 10,
            'payment_consistency': 1 - (float(input_df['delinq_2yrs'].iloc[0]) / 5),
            'asset_profile': float(input_df['mort_acc'].iloc[0]) / 5,
            'behavioral_factors': 1 - (float(input_df['inq_last_6mths'].iloc[0]) / 5)
        }
        
        # Ensure all values are within 0-1 range
        for key in breakdown:
            breakdown[key] = max(0, min(1, breakdown[key]))
        
        return {
            'credit_score': credit_score,
            'credit_grade': credit_grade,
            'default_probability': default_probability,
            'recommendation': recommendation,
            'rate_range': rate_range,
            'breakdown': breakdown
        }

    @staticmethod
    def grade_from_score(score):
        """Convert numeric score (0-1000) to letter grade"""
        if score >= 950:
            return "A+"
        elif score >= 900:
            return "A"
        elif score >= 850:
            return "A-"
        elif score >= 800:
            return "B+"
        elif score >= 750:
            return "B"
        elif score >= 700:
            return "B-"
        elif score >= 650:
            return "C+"
        elif score >= 600:
            return "C"
        elif score >= 550:
            return "C-"
        elif score >= 500:
            return "D+"
        elif score >= 450:
            return "D"
        elif score >= 400:
            return "D-"
        elif score >= 350:
            return "E+"
        elif score >= 300:
            return "E"
        elif score >= 250:
            return "E-"
        else:
            return "F"

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