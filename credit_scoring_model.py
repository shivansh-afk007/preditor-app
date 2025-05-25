import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

class CreditScoringModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.model_path = 'models/credit_scoring_model.pkl'
        
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
            'inq_last_6mths': np.random.choice([0, 1, 2, 3, 4], n_samples)
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
            'emp_length', 'annual_inc', 'verification_status',
            'delinq_2yrs', 'pub_rec', 'revol_util',
            'home_ownership', 'mort_acc',
            'dti', 'open_acc', 'total_acc', 'inq_last_6mths'
        ]
        
        # Filter features that exist in the dataset
        existing_features = [f for f in selected_features if f in data.columns]
        X = data[existing_features].copy()
        
        # Handle employment length
        if 'emp_length' in X.columns:
            emp_length_map = {
                '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
                '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
                '8 years': 8, '9 years': 9, '10+ years': 10
            }
            X['emp_length'] = X['emp_length'].map(lambda x: emp_length_map.get(x, np.nan))
        
        # Handle verification status
        if 'verification_status' in X.columns:
            verification_map = {
                'Not Verified': 0,
                'Verified': 1,
                'Source Verified': 2
            }
            X['verification_status'] = X['verification_status'].map(verification_map)
        
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
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
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
        
        self.model = model_data['pipeline']
        self.feature_names = model_data['feature_names']
        
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
        
        # Check for required features
        missing_features = [f for f in self.feature_names if f not in input_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select only the required features in the correct order
        input_df = input_df[self.feature_names]
        
        # Make prediction
        default_probability = float(self.model.predict_proba(input_df)[0, 1])  # Convert to float
        
        # Calculate credit score (inverse of default probability)
        # Scale to 0-1000 range
        credit_score = int(1000 - (default_probability * 1000)) # Scale to 0-1000
        credit_score = max(0, min(credit_score, 1000)) # Ensure score is within 0-1000
        
        # Determine credit grade based on 0-1000 scale (adjusting ranges)
        grade_ranges = {
            'A+': (950, 1000),
            'A': (900, 949),
            'A-': (850, 899),
            'B+': (800, 849),
            'B': (750, 799),
            'B-': (700, 749),
            'C+': (650, 699),
            'C': (600, 649),
            'C-': (550, 599),
            'D+': (500, 549),
            'D': (450, 499),
            'D-': (400, 449),
            'E+': (350, 399),
            'E': (300, 349),
            'E-': (250, 299),
            'F': (0, 249)
        }
        
        credit_grade = 'F'
        for grade, (min_score, max_score) in grade_ranges.items():
            if min_score <= credit_score <= max_score:
                credit_grade = grade
                break
        
        # Determine loan approval recommendation (adjusting score thresholds)
        if credit_score >= 700:  # Grade B- or better
            recommendation = "Approved"
            # Adjust rate range calculation for 1000 scale
            rate_range = f"{5 + (1000 - credit_score) / 60:.2f}% - {6 + (1000 - credit_score) / 50:.2f}%"
        elif credit_score >= 550:  # Grade C- to C+
            recommendation = "Conditionally Approved"
            # Adjust rate range calculation for 1000 scale
            rate_range = f"{8 + (700 - credit_score) / 30:.2f}% - {10 + (700 - credit_score) / 20:.2f}%"
        else:  # Grade D+ or lower
            recommendation = "Denied"
            rate_range = "N/A"
        
        # Create component scores (adjusting calculation for 1000 scale)
        component_scores = {
            'income_stability': int(credit_score * (0.9 + np.random.uniform(-0.1, 0.1)) * 1000 / 850), # Scale to 1000
            'payment_consistency': int(credit_score * (0.85 + np.random.uniform(-0.15, 0.15)) * 1000 / 850), # Scale to 1000
            'asset_profile': int(credit_score * (0.95 + np.random.uniform(-0.2, 0.1)) * 1000 / 850), # Scale to 1000
            'behavioral_factors': int(credit_score * (0.9 + np.random.uniform(-0.1, 0.1)) * 1000 / 850) # Scale to 1000
        }
        
        # Ensure component scores are within 0-1000
        for key in component_scores:
            component_scores[key] = max(0, min(component_scores[key], 1000))
        
        # Format results
        results = {
            'score': credit_score,
            'grade': credit_grade,
            'default_probability': default_probability,
            'recommendation': recommendation,
            'rate_range': rate_range,
            'breakdown': {
                'income_stability': self.grade_from_score(component_scores['income_stability']),
                'payment_consistency': self.grade_from_score(component_scores['payment_consistency']),
                'asset_profile': self.grade_from_score(component_scores['asset_profile']),
                'behavioral_factors': self.grade_from_score(component_scores['behavioral_factors'])
            }
        }
        
        return results

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