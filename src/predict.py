"""
WPW ECG Detection - Prediction Module

This module provides prediction capabilities for WPW syndrome detection
from ECG features using trained machine learning models.

Features:
- Load trained models from disk
- Make predictions on new ECG data
- Calculate prediction probabilities and confidence
- Risk level classification
- Batch prediction support
- Input validation and preprocessing

Author: Abdelmouaine Benzaim <benzaimabdelmouaine@gmail.com>
Date: 2026
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime


class WPWPredictor:
    """
    WPW syndrome predictor using trained machine learning models.

    This class handles loading models and making predictions on new ECG data,
    with comprehensive risk assessment and confidence scoring.
    """

    # Expected feature names (must match training data)
    EXPECTED_FEATURES = [
        'age', 'sex', 'hr_mean', 'hr_std', 'hr_max',
        'hrv_sdnn', 'hrv_rmssd', 'hrv_pnn50',
        'qrs_duration_ms', 'qrs_amplitude',
        'signal_range', 'signal_std'
    ]

    # Risk level thresholds based on probability
    RISK_THRESHOLDS = {
        'low': 0.3,
        'moderate': 0.6,
        'high': 0.8
    }

    def __init__(self, model_path: str):
        """
        Initialize the WPW predictor.

        Args:
            model_path: Path to the trained model file (.pkl)
        """
        self.model_path = model_path
        self.model = None
        self.metadata = None
        self.model_name = None

        self._load_model()

    def _load_model(self):
        """Load the trained model and its metadata."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        print(f"Loading model from: {self.model_path}")
        self.model = joblib.load(self.model_path)

        # Load metadata if available
        metadata_path = self.model_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.model_name = self.metadata.get('model_name', 'Unknown')
                print(f"Model: {self.model_name}")
                print(f"Training date: {self.metadata.get('training_date', 'Unknown')}")
        else:
            self.model_name = Path(self.model_path).stem
            print("Warning: Metadata file not found")

    def validate_features(self, features: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Validate and prepare input features.

        Args:
            features: Dictionary or DataFrame with ECG features

        Returns:
            Validated DataFrame with features in correct order

        Raises:
            ValueError: If required features are missing or invalid
        """
        # Convert dict to DataFrame if needed
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        elif not isinstance(features, pd.DataFrame):
            raise ValueError("Features must be a dictionary or DataFrame")

        # Check for missing features
        missing_features = set(self.EXPECTED_FEATURES) - set(features.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Encode sex if needed (Male=1, Female=0)
        if 'sex' in features.columns and features['sex'].dtype == 'object':
            features['sex'] = features['sex'].map({'Male': 1, 'Female': 0})

        # Reorder columns to match expected order
        features = features[self.EXPECTED_FEATURES]

        # Validate data types and ranges
        self._validate_ranges(features)

        return features

    def _validate_ranges(self, features: pd.DataFrame):
        """Validate that feature values are within reasonable ranges."""
        validations = {
            'age': (0, 120),
            'hr_mean': (30, 220),
            'hr_std': (0, 100),
            'hr_max': (40, 250),
            'hrv_sdnn': (0, 1),
            'hrv_rmssd': (0, 1),
            'hrv_pnn50': (0, 100),
            'qrs_duration_ms': (0, 200),
            'qrs_amplitude': (0, 1000),
            'signal_range': (0, 10000),
            'signal_std': (0, 1000)
        }

        for feature, (min_val, max_val) in validations.items():
            if feature in features.columns:
                values = features[feature]
                if (values < min_val).any() or (values > max_val).any():
                    print(f"Warning: {feature} values outside expected range [{min_val}, {max_val}]")

    def predict(self, features: Union[Dict, pd.DataFrame]) -> int:
        """
        Make a binary prediction (0=Normal, 1=WPW).

        Args:
            features: ECG features as dictionary or DataFrame

        Returns:
            Binary prediction (0 or 1)
        """
        features_df = self.validate_features(features)
        prediction = self.model.predict(features_df.values)
        return int(prediction[0])

    def predict_proba(self, features: Union[Dict, pd.DataFrame]) -> Dict[str, float]:
        """
        Get prediction probabilities for both classes.

        Args:
            features: ECG features as dictionary or DataFrame

        Returns:
            Dictionary with probabilities for each class
        """
        features_df = self.validate_features(features)
        probabilities = self.model.predict_proba(features_df.values)[0]

        return {
            'normal': float(probabilities[0]),
            'wpw': float(probabilities[1])
        }

    def get_risk_level(self, wpw_probability: float) -> str:
        """
        Determine risk level based on WPW probability.

        Args:
            wpw_probability: Probability of WPW syndrome (0-1)

        Returns:
            Risk level: 'low', 'moderate', 'high', or 'very_high'
        """
        if wpw_probability < self.RISK_THRESHOLDS['low']:
            return 'low'
        elif wpw_probability < self.RISK_THRESHOLDS['moderate']:
            return 'moderate'
        elif wpw_probability < self.RISK_THRESHOLDS['high']:
            return 'high'
        else:
            return 'very_high'

    def explain_prediction(self, features: Union[Dict, pd.DataFrame]) -> Dict:
        """
        Generate comprehensive prediction explanation.

        Args:
            features: ECG features as dictionary or DataFrame

        Returns:
            Dictionary with prediction, probabilities, risk level, and confidence
        """
        # Get prediction and probabilities
        prediction = self.predict(features)
        probabilities = self.predict_proba(features)
        wpw_prob = probabilities['wpw']

        # Determine risk level
        risk_level = self.get_risk_level(wpw_prob)

        # Calculate confidence (distance from decision boundary)
        confidence = abs(wpw_prob - 0.5) * 2  # Scale to 0-1

        # Interpret prediction
        if prediction == 1:
            interpretation = "WPW Syndrome Detected"
            recommendation = "Cardiology consultation recommended for ECG review and clinical correlation."
        else:
            interpretation = "Normal Sinus Rhythm"
            recommendation = "No WPW pattern detected. Continue routine cardiac monitoring as clinically indicated."

        return {
            'prediction': prediction,
            'interpretation': interpretation,
            'probabilities': probabilities,
            'wpw_probability': wpw_prob,
            'risk_level': risk_level,
            'confidence': confidence,
            'recommendation': recommendation,
            'model_name': self.model_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for multiple samples.

        Args:
            features_df: DataFrame with ECG features (multiple rows)

        Returns:
            DataFrame with predictions and probabilities
        """
        features_df = self.validate_features(features_df)

        # Get predictions and probabilities
        predictions = self.model.predict(features_df.values)
        probabilities = self.model.predict_proba(features_df.values)

        # Create results DataFrame
        results = pd.DataFrame({
            'prediction': predictions,
            'normal_probability': probabilities[:, 0],
            'wpw_probability': probabilities[:, 1],
            'risk_level': [self.get_risk_level(prob) for prob in probabilities[:, 1]],
            'interpretation': ['WPW Detected' if p == 1 else 'Normal' for p in predictions]
        })

        return results

    def get_feature_description(self) -> Dict[str, str]:
        """Get human-readable descriptions of ECG features."""
        return {
            'age': 'Patient age in years',
            'sex': 'Patient sex (Male=1, Female=0)',
            'hr_mean': 'Mean heart rate (bpm)',
            'hr_std': 'Heart rate standard deviation',
            'hr_max': 'Maximum heart rate (bpm)',
            'hrv_sdnn': 'HRV: Standard deviation of NN intervals',
            'hrv_rmssd': 'HRV: Root mean square of successive differences',
            'hrv_pnn50': 'HRV: Percentage of NN intervals > 50ms different',
            'qrs_duration_ms': 'QRS complex duration (milliseconds)',
            'qrs_amplitude': 'QRS complex amplitude',
            'signal_range': 'ECG signal range',
            'signal_std': 'ECG signal standard deviation'
        }

    def format_prediction_report(self, features: Union[Dict, pd.DataFrame]) -> str:
        """
        Generate a formatted text report of the prediction.

        Args:
            features: ECG features

        Returns:
            Formatted string report
        """
        result = self.explain_prediction(features)

        report = f"""
{'='*70}
WPW SYNDROME DETECTION REPORT
{'='*70}

TIMESTAMP: {result['timestamp']}
MODEL: {result['model_name']}

PREDICTION RESULTS:
------------------
Diagnosis:        {result['interpretation']}
WPW Probability:  {result['wpw_probability']:.2%}
Risk Level:       {result['risk_level'].upper()}
Confidence:       {result['confidence']:.2%}

PROBABILITY BREAKDOWN:
----------------------
Normal:           {result['probabilities']['normal']:.2%}
WPW Syndrome:     {result['probabilities']['wpw']:.2%}

CLINICAL RECOMMENDATION:
------------------------
{result['recommendation']}

{'='*70}
"""
        return report


def load_best_model(models_dir: str = "../models") -> WPWPredictor:
    """
    Load the best performing model based on metadata.

    Args:
        models_dir: Directory containing trained models

    Returns:
        WPWPredictor instance with the best model
    """
    models_path = Path(models_dir)

    # Find all model files
    model_files = list(models_path.glob("*_wpw.pkl"))

    if not model_files:
        raise FileNotFoundError(f"No trained models found in {models_dir}")

    # Load metadata for each model and find best ROC-AUC
    best_model_path = None
    best_roc_auc = 0

    for model_file in model_files:
        metadata_file = model_file.with_name(model_file.stem + '_metadata.json')
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                roc_auc = metadata.get('metrics', {}).get('roc_auc', 0)
                if roc_auc > best_roc_auc:
                    best_roc_auc = roc_auc
                    best_model_path = model_file

    if best_model_path is None:
        # If no metadata, just use the first model
        best_model_path = model_files[0]

    print(f"Loading best model: {best_model_path.name} (ROC-AUC: {best_roc_auc:.4f})")
    return WPWPredictor(str(best_model_path))


def main():
    """Example usage of the WPW predictor."""
    # Load model
    predictor = load_best_model()

    # Example ECG features
    example_features = {
        'age': 45,
        'sex': 'Male',
        'hr_mean': 128.79,
        'hr_std': 23.57,
        'hr_max': 148.51,
        'hrv_sdnn': 0.129,
        'hrv_rmssd': 0.189,
        'hrv_pnn50': 35.0,
        'qrs_duration_ms': 29.62,
        'qrs_amplitude': 85.43,
        'signal_range': 4304,
        'signal_std': 256.25
    }

    # Make prediction
    print("\n" + "="*70)
    print("EXAMPLE PREDICTION")
    print("="*70)

    result = predictor.explain_prediction(example_features)

    print(f"\nDiagnosis: {result['interpretation']}")
    print(f"WPW Probability: {result['wpw_probability']:.2%}")
    print(f"Risk Level: {result['risk_level'].upper()}")
    print(f"Confidence: {result['confidence']:.2%}")

    # Generate full report
    print(predictor.format_prediction_report(example_features))


if __name__ == "__main__":
    main()
