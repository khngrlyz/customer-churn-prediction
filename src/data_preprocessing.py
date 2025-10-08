"""
Data preprocessing module for customer churn prediction.
Handles data loading, cleaning, feature engineering, and transformation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import json


class ChurnDataPreprocessor:
    """Preprocessor for customer churn data."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_features = [
            'gender', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]
        self.numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

    def load_data(self, filepath):
        """Load data from CSV file."""
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records")
        return df

    def clean_data(self, df):
        """Clean the dataset."""
        df = df.copy()

        # Drop customerID as it's not useful for prediction
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)

        # Handle TotalCharges - convert to numeric, handle spaces
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            # Fill missing TotalCharges with MonthlyCharges for new customers
            df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])

        # Convert SeniorCitizen to object for consistency
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
            self.categorical_features.append('SeniorCitizen')

        print(f"Data cleaned. Shape: {df.shape}")
        return df

    def engineer_features(self, df):
        """Create new features from existing ones."""
        df = df.copy()

        # Tenure groups - new customers churn way more, so bin them separately
        df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72],
                                     labels=['0-1 year', '1-2 years', '2-4 years', '4+ years'])

        # Average monthly spend - normalize by tenure
        df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'] + 1)  # +1 to avoid division by zero

        # Service count
        service_cols = ['PhoneService', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies']
        df['total_services'] = df[service_cols].apply(
            lambda x: sum((x != 'No') & (x != 'No internet service') & (x != 'No phone service')),
            axis=1
        )

        # Has internet
        df['has_internet'] = df['InternetService'].apply(lambda x: 0 if x == 'No' else 1)

        # Payment method risk (electronic check has higher churn)
        df['payment_electronic_check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)

        print(f"Features engineered. New shape: {df.shape}")
        return df

    def encode_features(self, df, fit=True):
        """Encode categorical variables."""
        df = df.copy()

        # Convert tenure_group to string for encoding
        if 'tenure_group' in df.columns:
            df['tenure_group'] = df['tenure_group'].astype(str)
            if 'tenure_group' not in self.categorical_features:
                self.categorical_features.append('tenure_group')

        for col in self.categorical_features:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        df[col] = df[col].astype(str)
                        known_labels = set(self.label_encoders[col].classes_)
                        df[col] = df[col].apply(
                            lambda x: x if x in known_labels else self.label_encoders[col].classes_[0]
                        )
                        df[col] = self.label_encoders[col].transform(df[col])

        print(f"Categorical features encoded")
        return df

    def scale_features(self, df, fit=True):
        """Scale numerical features."""
        df = df.copy()

        # Add engineered numerical features
        numerical_features = self.numerical_features + ['avg_monthly_spend', 'total_services']
        numerical_features = [f for f in numerical_features if f in df.columns]

        if fit:
            df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        else:
            df[numerical_features] = self.scaler.transform(df[numerical_features])

        print(f"Numerical features scaled")
        return df

    def prepare_data(self, df, target_col='Churn', fit=True):
        """Full preprocessing pipeline."""
        df = self.clean_data(df)
        df = self.engineer_features(df)

        # Separate target if present
        if target_col in df.columns:
            y = df[target_col].map({'Yes': 1, 'No': 0})
            X = df.drop(target_col, axis=1)
        else:
            y = None
            X = df

        X = self.encode_features(X, fit=fit)
        X = self.scale_features(X, fit=fit)

        if fit:
            self.feature_names = X.columns.tolist()

        return X, y

    def save(self, filepath='models/preprocessor.joblib'):
        """Save preprocessor state."""
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features
        }, filepath)
        print(f"Preprocessor saved to {filepath}")

    def load(self, filepath='models/preprocessor.joblib'):
        """Load preprocessor state."""
        state = joblib.load(filepath)
        self.scaler = state['scaler']
        self.label_encoders = state['label_encoders']
        self.feature_names = state['feature_names']
        self.categorical_features = state['categorical_features']
        self.numerical_features = state['numerical_features']
        print(f"Preprocessor loaded from {filepath}")


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = ChurnDataPreprocessor()
    df = preprocessor.load_data('data/telco_churn.csv')
    X, y = preprocessor.prepare_data(df)

    print("\nPreprocessed data shape:", X.shape)
    print("Target distribution:")
    print(y.value_counts())
    print("\nFeature names:", X.columns.tolist())
