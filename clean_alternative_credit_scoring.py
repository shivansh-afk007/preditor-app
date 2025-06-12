import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class AlternativeCreditScorer:
    def __init__(self):
        self.category_weights = {
            'income_stability': 0.35,
            'payment_consistency': 0.30,
            'asset_value': 0.20,
            'behavioral_factors': 0.15
        }
        self.feature_weights = {
            'income_stability': {
                'employment_length': 0.20,
                'annual_income': 0.25,
                'income_to_loan_ratio': 0.15,
                'verification_status': 0.10,
                'gig_platforms_count': 0.10,
                'gig_platform_rating': 0.10,
                'gig_completion_rate': 0.10
            },
            'payment_consistency': {
                'utility_payments_ontime': 0.15,
                'rent_payments_ontime': 0.15,
                'subscription_payments_ontime': 0.10,
                'months_payment_history': 0.15,
                'late_payments_90d': 0.15,
                'delinq_2yrs': 0.15,
                'pub_rec': 0.10,
                'revol_util': 0.05
            },
            'asset_value': {
                'home_ownership': 0.25,
                'bank_balance_avg': 0.25,
                'bank_balance_min': 0.15,
                'investment_assets': 0.20,
                'mort_acc': 0.15
            },
            'behavioral_factors': {
                'dti': 0.20,
                'cashflow_ratio': 0.15,
                'savings_rate': 0.15,
                'digital_footprint_score': 0.10,
                'shopping_categories': 0.10,
                'gambling_expenses': 0.15,
                'education_level': 0.10,
                'open_acc': 0.05
            }
        }
        self.scaler = MinMaxScaler()

    def _normalize_feature(self, value, min_val, max_val, higher_is_better=True):
        value = np.clip(value, min_val, max_val)
        normalized = (value - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0
        return normalized if higher_is_better else 1 - normalized

    def _map_categorical(self, value, mapping, default_score=0.0):
        return mapping.get(str(value).upper(), default_score)

    def engineer_features(self, profile):
        # Ensure profile is a Series for consistent access
        if isinstance(profile, dict):
            profile = pd.Series(profile)
            
        # Fill NaNs with reasonable defaults
        default_values = {
            'employment_length': 0, 'annual_income': 30000, 'loan_amount': 10000, 'verification_status': 'NOT VERIFIED',
            'gig_platforms_count': 0, 'gig_platform_rating': 0, 'gig_completion_rate': 0,
            'utility_payments_ontime': 0.5, 'rent_payments_ontime': 0.5, 'subscription_payments_ontime': 0.5,
            'months_payment_history': 0, 'late_payments_90d': 0, 'delinq_2yrs': 0, 'pub_rec': 0, 'revol_util': 50,
            'home_ownership': 'RENT', 'bank_balance_avg': 1000, 'bank_balance_min': 100, 'investment_assets': 0, 'mort_acc': 0,
            'dti': 30, 'cashflow_ratio': 1.0, 'savings_rate': 0.0, 'digital_footprint_score': 500,
            'shopping_categories': 5, 'gambling_expenses': 0, 'education_level': 1, 'open_acc': 5
        }
        profile = profile.fillna(value=default_values)
        
        # Handle missing keys
        for key, default_val in default_values.items():
            if key not in profile:
                profile[key] = default_val
                
        eng_features = {}

        # Income Stability
        eng_features['employment_length_norm'] = self._normalize_feature(profile['employment_length'], 0, 10)
        eng_features['annual_income_norm'] = self._normalize_feature(profile['annual_income'], 10000, 200000)
        income_to_loan = profile['annual_income'] / profile['loan_amount'] if profile['loan_amount'] > 0 else 0
        eng_features['income_to_loan_ratio_norm'] = self._normalize_feature(income_to_loan, 0.5, 10)
        verification_map = {'VERIFIED': 1.0, 'SOURCE VERIFIED': 0.7, 'NOT VERIFIED': 0.3}
        eng_features['verification_status_norm'] = self._map_categorical(profile['verification_status'], verification_map, 0.3)
        eng_features['gig_platforms_count_norm'] = self._normalize_feature(profile['gig_platforms_count'], 0, 5)
        eng_features['gig_platform_rating_norm'] = self._normalize_feature(profile['gig_platform_rating'], 0, 5)
        eng_features['gig_completion_rate_norm'] = self._normalize_feature(profile['gig_completion_rate'], 0, 1)

        # Payment Consistency
        eng_features['utility_payments_ontime_norm'] = self._normalize_feature(profile['utility_payments_ontime'], 0, 1)
        eng_features['rent_payments_ontime_norm'] = self._normalize_feature(profile['rent_payments_ontime'], 0, 1)
        eng_features['subscription_payments_ontime_norm'] = self._normalize_feature(profile['subscription_payments_ontime'], 0, 1)
        eng_features['months_payment_history_norm'] = self._normalize_feature(profile['months_payment_history'], 0, 120)
        eng_features['late_payments_90d_norm'] = self._normalize_feature(profile['late_payments_90d'], 0, 5, higher_is_better=False)
        eng_features['delinq_2yrs_norm'] = self._normalize_feature(profile['delinq_2yrs'], 0, 5, higher_is_better=False)
        eng_features['pub_rec_norm'] = self._normalize_feature(profile['pub_rec'], 0, 3, higher_is_better=False)
        eng_features['revol_util_norm'] = self._normalize_feature(profile['revol_util'], 0, 100, higher_is_better=False)

        # Asset Value
        home_ownership_map = {'OWN': 1.0, 'MORTGAGE': 0.7, 'RENT': 0.3, 'OTHER': 0.2}
        eng_features['home_ownership_norm'] = self._map_categorical(profile['home_ownership'], home_ownership_map, 0.2)
        eng_features['bank_balance_avg_norm'] = self._normalize_feature(profile['bank_balance_avg'], 0, 50000)
        eng_features['bank_balance_min_norm'] = self._normalize_feature(profile['bank_balance_min'], 0, 10000)
        eng_features['investment_assets_norm'] = self._normalize_feature(profile['investment_assets'], 0, 100000)
        eng_features['mort_acc_norm'] = self._normalize_feature(profile['mort_acc'], 0, 4)

        # Behavioral Factors
        eng_features['dti_norm'] = self._normalize_feature(profile['dti'], 0, 50, higher_is_better=False)
        eng_features['cashflow_ratio_norm'] = self._normalize_feature(profile['cashflow_ratio'], 0.5, 3)
        eng_features['savings_rate_norm'] = self._normalize_feature(profile['savings_rate'], 0, 0.5)
        eng_features['digital_footprint_score_norm'] = self._normalize_feature(profile['digital_footprint_score'], 300, 850)
        eng_features['shopping_categories_norm'] = self._normalize_feature(profile['shopping_categories'], 1, 20)
        eng_features['gambling_expenses_norm'] = self._normalize_feature(profile['gambling_expenses'], 0, 1000, higher_is_better=False)
        eng_features['education_level_norm'] = self._normalize_feature(profile['education_level'], 0, 4)
        eng_features['open_acc_norm'] = self._normalize_feature(profile['open_acc'], 0, 40)

        return eng_features

    def calculate_category_scores(self, engineered_features):
        category_scores = {}
        for category, f_weights in self.feature_weights.items():
            score = 0
            for feature_name_base, weight in f_weights.items():
                norm_feature_name = f'{feature_name_base}_norm'
                feature_value = engineered_features.get(norm_feature_name, 0)
                score += feature_value * weight
            category_scores[category] = score
        return category_scores

    def calculate_final_score(self, category_scores):
        final_score = 0
        for category, weight in self.category_weights.items():
            final_score += category_scores.get(category, 0) * weight
        return int(final_score * 1000)

    def get_score_interpretation(self, final_score):
        if final_score >= 900: grade, desc, rec, rates = 'A+', 'Excellent', 'Approved', '4-7%'
        elif final_score >= 800: grade, desc, rec, rates = 'A', 'Very Good', 'Approved', '5-8%'
        elif final_score >= 700: grade, desc, rec, rates = 'B+', 'Good', 'Approved', '6-9%'
        elif final_score >= 650: grade, desc, rec, rates = 'B', 'Fair Good', 'Approved', '7-11%'
        elif final_score >= 600: grade, desc, rec, rates = 'C+', 'Fair', 'Conditionally Approved', '9-14%'
        elif final_score >= 550: grade, desc, rec, rates = 'C', 'Poor Fair', 'Conditionally Approved with Restrictions', '11-16%'
        elif final_score >= 500: grade, desc, rec, rates = 'D+', 'Poor', 'Conditionally Approved with High Restrictions', '14-18%'
        else: grade, desc, rec, rates = 'D', 'Very Poor', 'Denied', 'N/A'
        return {'grade': grade, 'description': desc, 'recommendation': rec, 'rate_range': rates}

    def score_profile(self, profile):
        engineered = self.engineer_features(profile)
        category_scores_raw = self.calculate_category_scores(engineered)
        final_score = self.calculate_final_score(category_scores_raw)
        interpretation = self.get_score_interpretation(final_score)
        
        # Category scores as percentage for display
        category_scores_display = {cat: int(score * 100) for cat, score in category_scores_raw.items()}
        
        return {
            'final_score': final_score,
            'interpretation': interpretation,
            'category_scores': category_scores_display,
            'engineered_features': engineered
        } 