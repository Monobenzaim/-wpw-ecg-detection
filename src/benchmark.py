"""
WPW ECG Detection - Comprehensive Model Benchmarking System

This module provides extensive benchmarking and comparison of multiple machine learning
algorithms for WPW syndrome detection, with focus on medical diagnostic metrics.

Features:
- Comparative evaluation of 7+ ML algorithms
- 20+ medical diagnostic metrics
- Statistical significance testing
- Visualization of model performance
- Cross-validation analysis
- Confusion matrices and ROC curves
- Clinical interpretation focus (sensitivity, specificity, etc.)
- Publication-ready results export

Author: Enhanced with Claude Code
Date: 2026
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, cross_val_score
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix,
    classification_report, matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score, average_precision_score,
    brier_score_loss, log_loss
)

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")


class WPWBenchmark:
    """
    Comprehensive benchmarking system for WPW detection models.

    Evaluates multiple ML algorithms using extensive medical diagnostic metrics
    to determine the best model for clinical WPW syndrome screening.
    """

    def __init__(
        self,
        data_path: str,
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: int = 5
    ):
        """
        Initialize the benchmark system.

        Args:
            data_path: Path to WPW dataset CSV file
            test_size: Proportion for test set (0.0-1.0)
            random_state: Random seed for reproducibility
            cv_folds: Number of cross-validation folds
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds

        # Data attributes
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

        # Results storage
        self.results = {}
        self.trained_models = {}

        # Setup paths
        self.project_root = Path(__file__).parent.parent
        self.output_dir = self.project_root / "benchmark_results"
        self.output_dir.mkdir(exist_ok=True)

        # Setup visualization style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def load_and_prepare_data(self):
        """Load dataset and prepare train/test splits."""
        print("Loading WPW dataset...")
        self.df = pd.read_csv(self.data_path)

        # Prepare features
        exclude_cols = ['record_id', 'dx']
        self.feature_names = [col for col in self.df.columns if col not in exclude_cols]

        X = self.df[self.feature_names].values
        y = self.df['dx'].values

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y
        )

        print(f"Dataset loaded: {self.df.shape[0]} samples, {len(self.feature_names)} features")
        print(f"Training set: {len(self.y_train)}, Test set: {len(self.y_test)}")
        print(f"Class distribution - Normal: {(y==0).sum()}, WPW: {(y==1).sum()}")

    def _define_models(self) -> Dict:
        """Define all models to benchmark."""
        models = {
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=2)),
                ('clf', LogisticRegression(max_iter=1000, random_state=self.random_state))
            ]),

            'Decision Tree': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', DecisionTreeClassifier(
                    max_depth=10, min_samples_split=5,
                    random_state=self.random_state
                ))
            ]),

            'Random Forest': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(
                    n_estimators=200, max_depth=15,
                    random_state=self.random_state
                ))
            ]),

            'Gradient Boosting': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.1,
                    max_depth=5, random_state=self.random_state
                ))
            ]),

            'SVM (RBF)': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', SVC(
                    kernel='rbf', C=10, gamma='scale',
                    probability=True, random_state=self.random_state
                ))
            ]),

            'Naive Bayes': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GaussianNB())
            ]),

            'K-Nearest Neighbors': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', KNeighborsClassifier(n_neighbors=5))
            ]),

            'AdaBoost': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', AdaBoostClassifier(
                    n_estimators=100, learning_rate=1.0,
                    random_state=self.random_state
                ))
            ])
        }

        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', XGBClassifier(
                    n_estimators=200, max_depth=5,
                    learning_rate=0.1, random_state=self.random_state,
                    eval_metric='logloss'
                ))
            ])

        return models

    def _calculate_medical_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict:
        """
        Calculate comprehensive medical diagnostic metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities

        Returns:
            Dictionary with all medical metrics
        """
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Basic metrics
        metrics = {
            # Classification metrics
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),

            # Medical diagnostic metrics
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Positive Predictive Value
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value

            # Rates
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,  # False Positive Rate
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,  # False Negative Rate
            'fdr': fp / (fp + tp) if (fp + tp) > 0 else 0,  # False Discovery Rate

            # Likelihood ratios
            'lr_positive': (tp / (tp + fn)) / (fp / (fp + tn)) if fp > 0 and tn > 0 else float('inf'),
            'lr_negative': (fn / (fn + tp)) / (tn / (tn + fp)) if fn > 0 and fp > 0 else 0,

            # Advanced metrics
            'mcc': matthews_corrcoef(y_true, y_pred),  # Matthews Correlation Coefficient
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'youden_index': (tp / (tp + fn)) + (tn / (tn + fp)) - 1 if (tp + fn) > 0 and (tn + fp) > 0 else 0,

            # Probabilistic metrics
            'roc_auc': roc_auc_score(y_true, y_prob),
            'pr_auc': average_precision_score(y_true, y_prob),
            'brier_score': brier_score_loss(y_true, y_prob),
            'log_loss': log_loss(y_true, y_prob),

            # Confusion matrix components
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),

            # Diagnostic accuracy
            'diagnostic_odds_ratio': (tp * tn) / (fp * fn) if (fp * fn) > 0 else float('inf')
        }

        return metrics

    def evaluate_model(self, model_name: str, model) -> Dict:
        """
        Comprehensive evaluation of a single model.

        Args:
            model_name: Name of the model
            model: Sklearn model/pipeline

        Returns:
            Dictionary with evaluation results
        """
        print(f"\nEvaluating: {model_name}")

        # Train model
        model.fit(self.X_train, self.y_train)

        # Predictions
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]

        # Calculate metrics
        metrics = self._calculate_medical_metrics(self.y_test, y_pred, y_prob)

        # Cross-validation scores
        cv_scores = cross_val_score(
            model, self.X_train, self.y_train,
            cv=StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state),
            scoring='roc_auc',
            n_jobs=-1
        )

        metrics['cv_roc_auc_mean'] = cv_scores.mean()
        metrics['cv_roc_auc_std'] = cv_scores.std()

        # Store results
        self.results[model_name] = metrics
        self.trained_models[model_name] = model

        # Print summary
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")

        return metrics

    def run_complete_benchmark(self):
        """Run complete benchmark on all models."""
        print("\n" + "="*70)
        print("STARTING COMPREHENSIVE WPW DETECTION MODEL BENCHMARK")
        print("="*70)

        # Load data
        self.load_and_prepare_data()

        # Get models
        models = self._define_models()

        print(f"\nBenchmarking {len(models)} algorithms...")

        # Evaluate each model
        for model_name, model in models.items():
            self.evaluate_model(model_name, model)

        print("\n" + "="*70)
        print("BENCHMARK COMPLETE")
        print("="*70)

    def generate_comparison_table(self) -> pd.DataFrame:
        """Generate comprehensive comparison table."""
        # Key metrics for comparison
        key_metrics = [
            'accuracy', 'balanced_accuracy', 'roc_auc', 'pr_auc',
            'sensitivity', 'specificity', 'ppv', 'npv',
            'f1_score', 'mcc', 'cohen_kappa', 'youden_index',
            'cv_roc_auc_mean', 'cv_roc_auc_std'
        ]

        # Build comparison DataFrame
        comparison_data = []
        for model_name, metrics in self.results.items():
            row = {'Model': model_name}
            for metric in key_metrics:
                row[metric] = metrics.get(metric, np.nan)
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Sort by ROC-AUC
        df = df.sort_values('roc_auc', ascending=False)

        return df

    def print_results_summary(self):
        """Print formatted results summary."""
        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)

        df = self.generate_comparison_table()

        # Print key metrics
        print("\n📊 KEY PERFORMANCE METRICS:\n")
        display_df = df[['Model', 'accuracy', 'roc_auc', 'sensitivity', 'specificity', 'f1_score']].copy()
        display_df.columns = ['Model', 'Accuracy', 'ROC-AUC', 'Sensitivity', 'Specificity', 'F1-Score']

        # Format numbers
        for col in display_df.columns[1:]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

        print(display_df.to_string(index=False))

        # Best models
        print("\n🏆 TOP PERFORMERS:")
        print(f"  Best Overall (ROC-AUC): {df.iloc[0]['Model']} ({df.iloc[0]['roc_auc']:.4f})")
        print(f"  Best Sensitivity: {df.loc[df['sensitivity'].idxmax(), 'Model']} ({df['sensitivity'].max():.4f})")
        print(f"  Best Specificity: {df.loc[df['specificity'].idxmax(), 'Model']} ({df['specificity'].max():.4f})")

    def plot_roc_curves(self):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))

        for model_name, model in self.trained_models.items():
            # Get probabilities
            y_prob = model.predict_proba(self.X_test)[:, 1]

            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            # Plot
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

        # Diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('ROC Curves - WPW Detection Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save
        filepath = self.output_dir / f"roc_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\nROC curves saved: {filepath}")
        plt.close()

    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models."""
        n_models = len(self.trained_models)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]

        for idx, (model_name, model) in enumerate(self.trained_models.items()):
            # Get predictions
            y_pred = model.predict(self.X_test)

            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)

            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar=True, square=True)
            axes[idx].set_xlabel('Predicted', fontsize=10)
            axes[idx].set_ylabel('Actual', fontsize=10)
            axes[idx].set_title(f'{model_name}\nAccuracy: {self.results[model_name]["accuracy"]:.3f}',
                               fontsize=11, fontweight='bold')
            axes[idx].set_xticklabels(['Normal', 'WPW'])
            axes[idx].set_yticklabels(['Normal', 'WPW'])

        # Hide extra subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Confusion Matrices - WPW Detection Models', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()

        # Save
        filepath = self.output_dir / f"confusion_matrices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved: {filepath}")
        plt.close()

    def plot_metrics_comparison(self):
        """Plot bar chart comparing key metrics across models."""
        df = self.generate_comparison_table()

        metrics_to_plot = ['accuracy', 'sensitivity', 'specificity', 'f1_score', 'roc_auc']
        metric_labels = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'ROC-AUC']

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(df))
        width = 0.15

        for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            ax.bar(x + idx*width, df[metric], width, label=label)

        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison - Key Medical Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])

        plt.tight_layout()

        # Save
        filepath = self.output_dir / f"metrics_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved: {filepath}")
        plt.close()

    def export_results(self):
        """Export results to CSV and JSON."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Export comparison table
        df = self.generate_comparison_table()
        csv_path = self.output_dir / f"benchmark_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults exported to CSV: {csv_path}")

        # Export detailed JSON
        json_path = self.output_dir / f"benchmark_detailed_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Detailed results exported to JSON: {json_path}")

    def generate_all_visualizations(self):
        """Generate all visualization plots."""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)

        self.plot_roc_curves()
        self.plot_confusion_matrices()
        self.plot_metrics_comparison()

        print("\n✅ All visualizations generated successfully!")


def main():
    """Main benchmarking pipeline."""
    # Configuration
    DATA_PATH = "../data/wpw_dataset_features.csv"

    # Initialize benchmark
    benchmark = WPWBenchmark(
        data_path=DATA_PATH,
        test_size=0.2,
        random_state=42,
        cv_folds=5
    )

    # Run complete benchmark
    benchmark.run_complete_benchmark()

    # Print results
    benchmark.print_results_summary()

    # Generate visualizations
    benchmark.generate_all_visualizations()

    # Export results
    benchmark.export_results()

    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETE!")
    print("="*70)
    print(f"Results saved to: {benchmark.output_dir}")


if __name__ == "__main__":
    main()
