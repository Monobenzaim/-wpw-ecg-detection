"""
WPW ECG Detection - Model Training Module

This module provides automated training for WPW (Wolff-Parkinson-White) syndrome
detection from ECG features using multiple machine learning algorithms.

Models supported:
- Logistic Regression (with polynomial features)
- Random Forest Classifier
- Support Vector Machine (SVM)
- Gradient Boosting
- XGBoost

Features:
- Automated hyperparameter tuning with GridSearchCV
- Cross-validation with StratifiedKFold
- Comprehensive evaluation metrics
- Model persistence with joblib
- Feature importance analysis

Author: Abdelmouaine Benzaim <benzaimabdelmouaine@gmail.com>
Date: 2026
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import xgboost (optional)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")


class WPWTrainer:
    """
    Comprehensive trainer for WPW syndrome detection models.

    This class handles the entire training pipeline from data loading
    to model persistence, with built-in hyperparameter optimization.
    """

    def __init__(self, data_path: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the WPW trainer.

        Args:
            data_path: Path to the WPW dataset CSV file
            test_size: Proportion of data to use for testing (0.0-1.0)
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state

        # Data attributes
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

        # Model attributes
        self.models = {}
        self.results = {}

        # Setup paths
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load and prepare the WPW dataset."""
        print("Loading WPW dataset...")
        self.df = pd.read_csv(self.data_path)

        print(f"Dataset shape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nClass distribution:")
        print(self.df['dx'].value_counts())
        print(f"  - Normal (0): {(self.df['dx']==0).sum()} samples")
        print(f"  - WPW (1): {(self.df['dx']==1).sum()} samples")

        return self.df

    def prepare_features(self):
        """Prepare features and target for training."""
        print("\nPreparing features...")

        # Define features (exclude record_id and target)
        exclude_cols = ['record_id', 'dx']
        self.feature_names = [col for col in self.df.columns if col not in exclude_cols]

        # Separate features and target
        X = self.df[self.feature_names].values
        y = self.df['dx'].values

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y  # Maintain class distribution
        )

        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Features: {len(self.feature_names)}")

    def train_logistic_regression(self, polynomial_degree: int = 2) -> Dict:
        """
        Train Logistic Regression with polynomial features and optimization.

        Args:
            polynomial_degree: Degree of polynomial features to generate

        Returns:
            Dictionary with model and performance metrics
        """
        print("\n" + "="*60)
        print("Training: Logistic Regression with Polynomial Features")
        print("="*60)

        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=polynomial_degree)),
            ('logreg', LogisticRegression(max_iter=1000, random_state=self.random_state))
        ])

        # Hyperparameter grid
        param_grid = {
            'poly__degree': [1, 2],
            'logreg__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'logreg__penalty': ['l1', 'l2'],
            'logreg__solver': ['liblinear', 'saga']
        }

        # Grid search with stratified k-fold
        grid_search = GridSearchCV(
            pipeline, param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        # Train
        grid_search.fit(self.X_train, self.y_train)

        # Best model
        best_model = grid_search.best_estimator_
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

        # Evaluate
        results = self._evaluate_model(best_model, "Logistic_Regression")

        # Save model
        self._save_model(best_model, "logistic_regression_wpw.pkl", results)

        self.models['logistic_regression'] = best_model
        self.results['logistic_regression'] = results

        return results

    def train_random_forest(self) -> Dict:
        """Train Random Forest Classifier with optimization."""
        print("\n" + "="*60)
        print("Training: Random Forest Classifier")
        print("="*60)

        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state=self.random_state))
        ])

        # Hyperparameter grid
        param_grid = {
            'rf__n_estimators': [100, 200, 300],
            'rf__max_depth': [5, 10, 15, None],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4],
            'rf__max_features': ['sqrt', 'log2']
        }

        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        # Train
        grid_search.fit(self.X_train, self.y_train)

        # Best model
        best_model = grid_search.best_estimator_
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

        # Evaluate
        results = self._evaluate_model(best_model, "Random_Forest")

        # Feature importance
        self._plot_feature_importance(
            best_model.named_steps['rf'],
            "Random Forest Feature Importance"
        )

        # Save model
        self._save_model(best_model, "random_forest_wpw.pkl", results)

        self.models['random_forest'] = best_model
        self.results['random_forest'] = results

        return results

    def train_svm(self) -> Dict:
        """Train Support Vector Machine with optimization."""
        print("\n" + "="*60)
        print("Training: Support Vector Machine (SVM)")
        print("="*60)

        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, random_state=self.random_state))
        ])

        # Hyperparameter grid
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__kernel': ['linear', 'rbf', 'poly'],
            'svm__gamma': ['scale', 'auto', 0.001, 0.01]
        }

        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        # Train
        grid_search.fit(self.X_train, self.y_train)

        # Best model
        best_model = grid_search.best_estimator_
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

        # Evaluate
        results = self._evaluate_model(best_model, "SVM")

        # Save model
        self._save_model(best_model, "svm_wpw.pkl", results)

        self.models['svm'] = best_model
        self.results['svm'] = results

        return results

    def train_gradient_boosting(self) -> Dict:
        """Train Gradient Boosting Classifier with optimization."""
        print("\n" + "="*60)
        print("Training: Gradient Boosting Classifier")
        print("="*60)

        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('gb', GradientBoostingClassifier(random_state=self.random_state))
        ])

        # Hyperparameter grid
        param_grid = {
            'gb__n_estimators': [100, 200, 300],
            'gb__learning_rate': [0.01, 0.1, 0.2],
            'gb__max_depth': [3, 5, 7],
            'gb__min_samples_split': [2, 5, 10],
            'gb__subsample': [0.8, 1.0]
        }

        # Grid search
        grid_search = GridSearchCV(
            pipeline, param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        # Train
        grid_search.fit(self.X_train, self.y_train)

        # Best model
        best_model = grid_search.best_estimator_
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

        # Evaluate
        results = self._evaluate_model(best_model, "Gradient_Boosting")

        # Save model
        self._save_model(best_model, "gradient_boosting_wpw.pkl", results)

        self.models['gradient_boosting'] = best_model
        self.results['gradient_boosting'] = results

        return results

    def _evaluate_model(self, model, model_name: str) -> Dict:
        """
        Comprehensive model evaluation.

        Args:
            model: Trained sklearn model
            model_name: Name for the model

        Returns:
            Dictionary with all evaluation metrics
        """
        # Predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]

        # Calculate metrics
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results = {
            'model_name': model_name,
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred),
            'recall': recall_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'mcc': matthews_corrcoef(self.y_test, y_pred),
            'confusion_matrix': cm,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }

        # Print results
        print(f"\n{'='*60}")
        print(f"Evaluation Results: {model_name}")
        print(f"{'='*60}")
        print(f"Accuracy:     {results['accuracy']:.4f}")
        print(f"Precision:    {results['precision']:.4f}")
        print(f"Recall:       {results['recall']:.4f}")
        print(f"F1-Score:     {results['f1_score']:.4f}")
        print(f"ROC-AUC:      {results['roc_auc']:.4f}")
        print(f"Specificity:  {results['specificity']:.4f}")
        print(f"Sensitivity:  {results['sensitivity']:.4f}")
        print(f"MCC:          {results['mcc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {tn}  FP: {fp}")
        print(f"  FN: {fn}  TP: {tp}")

        return results

    def _plot_feature_importance(self, model, title: str):
        """Plot feature importance for tree-based models."""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(10, 6))
            plt.title(title)
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)),
                      [self.feature_names[i] for i in indices],
                      rotation=45, ha='right')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.tight_layout()

            # Save plot
            plot_path = self.models_dir / f"{title.replace(' ', '_').lower()}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {plot_path}")
            plt.close()

    def _save_model(self, model, filename: str, results: Dict):
        """Save trained model and metadata."""
        model_path = self.models_dir / filename

        # Save model
        joblib.dump(model, model_path)
        print(f"\nModel saved to: {model_path}")

        # Save metadata
        metadata = {
            'model_name': results['model_name'],
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'features': self.feature_names,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'metrics': {k: v for k, v in results.items() if k != 'confusion_matrix'},
            'dataset_path': self.data_path
        }

        metadata_path = self.models_dir / filename.replace('.pkl', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Metadata saved to: {metadata_path}")

    def train_all_models(self) -> Dict:
        """Train all available models and return results."""
        print("\n" + "="*70)
        print("STARTING COMPREHENSIVE WPW DETECTION MODEL TRAINING")
        print("="*70)

        # Load and prepare data
        self.load_data()
        self.prepare_features()

        # Train all models
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_svm()
        self.train_gradient_boosting()

        # Summary
        print("\n" + "="*70)
        print("TRAINING COMPLETE - MODEL COMPARISON")
        print("="*70)

        comparison_df = pd.DataFrame([
            {
                'Model': results['model_name'],
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
                'ROC-AUC': f"{results['roc_auc']:.4f}"
            }
            for results in self.results.values()
        ])

        print(comparison_df.to_string(index=False))

        # Find best model
        best_model_name = max(self.results.keys(),
                             key=lambda k: self.results[k]['roc_auc'])
        print(f"\n🏆 Best Model (by ROC-AUC): {best_model_name.replace('_', ' ').title()}")

        return self.results


def main():
    """Main training pipeline."""
    # Configuration
    DATA_PATH = "../data/wpw_dataset_features.csv"

    # Initialize trainer
    trainer = WPWTrainer(
        data_path=DATA_PATH,
        test_size=0.2,
        random_state=42
    )

    # Train all models
    results = trainer.train_all_models()

    print("\n✅ Training pipeline completed successfully!")
    print(f"Models saved to: {trainer.models_dir}")


if __name__ == "__main__":
    main()
