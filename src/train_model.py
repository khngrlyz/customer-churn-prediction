"""
Model training module for customer churn prediction.
Includes model selection, hyperparameter tuning, and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import json
from data_preprocessing import ChurnDataPreprocessor


class ChurnModelTrainer:
    """Trainer for customer churn prediction models."""

    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.metrics = {}

    def train_test_split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Train churn rate: {y_train.mean():.2%}")
        print(f"Test churn rate: {y_test.mean():.2%}")
        return X_train, X_test, y_train, y_test

    def handle_imbalance(self, X_train, y_train):
        """Apply SMOTE to handle class imbalance."""
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {X_train_balanced.shape}")
        print(f"Churn rate after SMOTE: {y_train_balanced.mean():.2%}")
        return X_train_balanced, y_train_balanced

    def train_baseline_models(self, X_train, y_train, X_test, y_test):
        """Train multiple baseline models for comparison."""
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        }

        results = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            print(f"{name} - Accuracy: {results[name]['accuracy']:.4f}, "
                  f"F1: {results[name]['f1_score']:.4f}, "
                  f"ROC-AUC: {results[name]['roc_auc']:.4f}")

        return results

    def tune_xgboost(self, X_train, y_train):
        """Hyperparameter tuning for XGBoost."""
        print("\nPerforming hyperparameter tuning for XGBoost...")

        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'min_child_weight': [1, 3, 5]
        }

        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')

        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_

        return grid_search.best_estimator_

    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation."""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Classification metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        # ROC-AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")

        # Store metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        return self.metrics

    def get_feature_importance(self, feature_names):
        """Extract and sort feature importance."""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10))

            self.feature_importance = importance_df.to_dict('records')
            return importance_df
        return None

    def save_model(self, filepath='models/churn_model.joblib'):
        """Save trained model."""
        joblib.dump(self.model, filepath)
        print(f"\nModel saved to {filepath}")

    def save_metrics(self, filepath='models/metrics.json'):
        """Save evaluation metrics."""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {filepath}")

    def save_feature_importance(self, filepath='models/feature_importance.json'):
        """Save feature importance."""
        if self.feature_importance:
            with open(filepath, 'w') as f:
                json.dump(self.feature_importance, f, indent=2)
            print(f"Feature importance saved to {filepath}")


def main():
    """Main training pipeline."""
    # Configuration
    COMPARE_BASELINES = True  # Set to False to skip baseline comparison and save ~1 minute

    print("="*50)
    print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
    print("="*50)

    # 1. Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    preprocessor = ChurnDataPreprocessor()
    df = preprocessor.load_data('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    X, y = preprocessor.prepare_data(df, fit=True)
    preprocessor.save('models/preprocessor.joblib')

    # 2. Split data
    print("\n2. Splitting data...")
    trainer = ChurnModelTrainer()
    X_train, X_test, y_train, y_test = trainer.train_test_split_data(X, y)

    # 3. Handle class imbalance
    print("\n3. Handling class imbalance with SMOTE...")
    X_train_balanced, y_train_balanced = trainer.handle_imbalance(X_train, y_train)

    # 4. Train baseline models (optional - for comparison)
    if COMPARE_BASELINES:
        print("\n4. Training baseline models for comparison...")
        baseline_results = trainer.train_baseline_models(
            X_train_balanced, y_train_balanced, X_test, y_test
        )
        step_num = 5
    else:
        print("\n4. Skipping baseline comparison (using XGBoost directly)...")
        step_num = 4

    # 5. Hyperparameter tuning
    print(f"\n{step_num}. Hyperparameter tuning...")
    best_model = trainer.tune_xgboost(X_train_balanced, y_train_balanced)
    step_num += 1

    # Evaluate final model
    print(f"\n{step_num}. Evaluating final model...")
    metrics = trainer.evaluate_model(X_test, y_test)
    step_num += 1

    # Feature importance
    print(f"\n{step_num}. Analyzing feature importance...")
    trainer.get_feature_importance(X.columns.tolist())
    step_num += 1

    # Save everything
    print(f"\n{step_num}. Saving model and artifacts...")
    trainer.save_model('models/churn_model.joblib')
    trainer.save_metrics('models/metrics.json')
    trainer.save_feature_importance('models/feature_importance.json')

    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)


if __name__ == "__main__":
    main()
