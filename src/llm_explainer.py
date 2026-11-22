"""
WPW ECG Detection - LLM-Powered Medical Explanation Module

This module integrates Groq's LLM API to provide intelligent, context-aware
medical explanations for WPW syndrome predictions from ECG analysis.

Features:
- AI-powered interpretation of ECG features and predictions
- Clinical context generation
- Patient-friendly explanations
- Risk factor analysis specific to WPW syndrome
- Medical recommendations based on prediction results
- Support for Groq's Llama 3.1 models

Author: Abdelmouaine Benzaim <benzaimabdelmouaine@gmail.com>
Date: 2026
"""

import os
from typing import Dict, Optional, List
from datetime import datetime

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Warning: Groq library not installed. Install with: pip install groq")


class ECGExplainer:
    """
    Medical explanation generator using Groq's LLM API.

    This class generates comprehensive, clinically-relevant explanations
    for WPW syndrome predictions based on ECG features.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.1-70b-versatile"):
        """
        Initialize the ECG explainer.

        Args:
            api_key: Groq API key (if None, reads from GROQ_API_KEY env variable)
            model: Groq model to use for explanations
        """
        if not GROQ_AVAILABLE:
            raise ImportError(
                "Groq library not installed. Install with: pip install groq\n"
                "Get free API key at: https://console.groq.com/keys"
            )

        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not provided. Either pass api_key parameter or "
                "set GROQ_API_KEY environment variable.\n"
                "Get free API key at: https://console.groq.com/keys"
            )

        self.client = Groq(api_key=self.api_key)
        self.model = model

    def format_ecg_data(self, features: Dict) -> str:
        """
        Format ECG features into human-readable clinical description.

        Args:
            features: Dictionary of ECG features

        Returns:
            Formatted string with clinical data
        """
        # Map sex encoding to string
        sex_str = 'Male' if features.get('sex', 1) == 1 else 'Female'

        return f"""
**Demographics:**
- Age: {features.get('age', 'N/A')} years
- Sex: {sex_str}

**Heart Rate Analysis:**
- Mean Heart Rate: {features.get('hr_mean', 'N/A'):.1f} bpm
- HR Standard Deviation: {features.get('hr_std', 'N/A'):.2f}
- Maximum Heart Rate: {features.get('hr_max', 'N/A'):.1f} bpm

**Heart Rate Variability (HRV):**
- SDNN: {features.get('hrv_sdnn', 'N/A'):.4f}
- RMSSD: {features.get('hrv_rmssd', 'N/A'):.4f}
- pNN50: {features.get('hrv_pnn50', 'N/A'):.1f}%

**QRS Complex Analysis:**
- QRS Duration: {features.get('qrs_duration_ms', 'N/A'):.2f} ms
- QRS Amplitude: {features.get('qrs_amplitude', 'N/A'):.2f}

**ECG Signal Characteristics:**
- Signal Range: {features.get('signal_range', 'N/A'):.0f}
- Signal Std Dev: {features.get('signal_std', 'N/A'):.2f}
"""

    def explain_prediction(
        self,
        features: Dict,
        prediction: int,
        probability: float,
        risk_level: str,
        model_name: str = "Machine Learning Model"
    ) -> str:
        """
        Generate comprehensive AI-powered medical explanation.

        Args:
            features: Dictionary of ECG features
            prediction: Binary prediction (0=Normal, 1=WPW)
            probability: Probability of WPW syndrome (0-1)
            risk_level: Risk level classification
            model_name: Name of the ML model used

        Returns:
            Detailed medical explanation as markdown-formatted string
        """
        # Format patient data
        patient_data = self.format_ecg_data(features)

        # Diagnosis string
        diagnosis = "WPW Syndrome Pattern Detected" if prediction == 1 else "Normal Sinus Rhythm"

        # Create comprehensive prompt
        prompt = f"""You are a board-certified cardiologist and electrophysiology specialist providing expert analysis of ECG-based WPW (Wolff-Parkinson-White) syndrome detection results.

**Context:**
This analysis is based on automated ECG feature extraction and machine learning prediction. WPW syndrome is a pre-excitation disorder caused by an accessory pathway (Bundle of Kent) that can lead to tachyarrhythmias.

**Patient ECG Data:**
{patient_data}

**AI-Assisted Diagnostic Assessment:**
- Model: {model_name}
- Prediction: {diagnosis}
- WPW Probability: {probability:.1%}
- Risk Classification: {risk_level.upper()}
- Confidence Level: {abs(probability - 0.5) * 2:.1%}

**Please provide a comprehensive clinical analysis with the following sections:**

## 1. Clinical Summary
Provide a brief, clinically-oriented summary of the ECG findings and their significance. Reference specific features from the data above.

## 2. ECG Feature Analysis
Analyze the key ECG parameters:
- **QRS Duration**: Interpret the QRS duration in context of WPW (normal <120ms, WPW often shows delta wave causing wider QRS)
- **Heart Rate**: Assess the heart rate patterns and variability
- **HRV Metrics**: Explain what the HRV measurements suggest about autonomic function and arrhythmia risk

## 3. WPW Syndrome Assessment
If WPW is detected (prediction=1):
- Explain the likely accessory pathway characteristics
- Discuss arrhythmia risk (AVRT, atrial fibrillation with rapid ventricular response)
- Address sudden cardiac death risk stratification

If Normal (prediction=0):
- Confirm absence of pre-excitation pattern
- Address any borderline findings if present

## 4. Clinical Recommendations
Provide evidence-based recommendations:
- Need for electrophysiology consultation
- Exercise testing considerations
- Risk stratification for invasive procedures (catheter ablation)
- Activity restrictions if applicable
- Follow-up monitoring plan

## 5. Patient Education Points
Explain in patient-friendly language:
- What these findings mean for the patient
- Symptoms to watch for (palpitations, syncope, chest pain)
- When to seek immediate medical attention
- Lifestyle considerations

**Important Guidelines:**
- Base your analysis on the specific ECG values provided
- Use evidence-based medical reasoning
- Be specific and actionable in recommendations
- Acknowledge the AI-assisted nature of this assessment
- Emphasize this is a screening tool requiring physician correlation
- Use appropriate medical terminology while remaining clear
"""

        try:
            # Call Groq API
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert cardiologist specializing in cardiac electrophysiology and WPW syndrome. Provide detailed, evidence-based clinical analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=2048,
                top_p=0.9
            )

            explanation = chat_completion.choices[0].message.content

            return explanation

        except Exception as e:
            # Fallback if API call fails
            return self._generate_fallback_explanation(
                features, prediction, probability, risk_level
            )

    def _generate_fallback_explanation(
        self,
        features: Dict,
        prediction: int,
        probability: float,
        risk_level: str
    ) -> str:
        """Generate a basic explanation if API call fails."""
        diagnosis = "WPW Syndrome Pattern" if prediction == 1 else "Normal Sinus Rhythm"

        explanation = f"""
## Clinical Summary

Based on ECG feature analysis, the AI model indicates: **{diagnosis}**

**Key Findings:**
- WPW Probability: {probability:.1%}
- Risk Level: {risk_level.upper()}
- QRS Duration: {features.get('qrs_duration_ms', 'N/A'):.1f} ms
- Mean Heart Rate: {features.get('hr_mean', 'N/A'):.1f} bpm

## Interpretation

"""
        if prediction == 1:
            explanation += """
The ECG features suggest possible WPW syndrome pattern, which is characterized by:
- Accessory pathway conduction (Bundle of Kent)
- Potential for pre-excitation and delta wave
- Increased risk of supraventricular tachycardia (AVRT)

**Recommended Actions:**
1. Cardiology/EP consultation for comprehensive ECG review
2. Consider exercise stress test for risk stratification
3. Evaluate for symptoms: palpitations, syncope, chest pain
4. Discuss catheter ablation if symptomatic or high-risk

"""
        else:
            explanation += """
The ECG features indicate normal sinus rhythm without evidence of pre-excitation.
- No WPW pattern detected
- QRS duration within normal limits
- Regular rhythm characteristics

**Recommended Actions:**
1. Continue routine cardiac monitoring as indicated
2. Report any new cardiac symptoms
3. Standard cardiovascular risk factor management

"""

        explanation += """
## Important Note

This is an AI-assisted screening tool. All findings require clinical correlation
and confirmation by a qualified physician through standard 12-lead ECG interpretation.

---
*Note: LLM API unavailable - showing fallback explanation*
"""

        return explanation

    def explain_risk_factors(self, features: Dict, risk_level: str) -> str:
        """
        Generate explanation of WPW-specific risk factors.

        Args:
            features: ECG features dictionary
            risk_level: Classified risk level

        Returns:
            Risk factor analysis as string
        """
        prompt = f"""As a cardiac electrophysiology specialist, analyze the following ECG data for WPW syndrome risk factors:

{self.format_ecg_data(features)}

**Risk Classification:** {risk_level.upper()}

Provide a focused analysis of:
1. **Arrhythmia Risk Factors**: What features increase risk of AVRT or atrial fibrillation?
2. **Sudden Cardiac Death Risk**: Are there concerning features (very short RR interval in AF, etc.)?
3. **Exercise/Activity Considerations**: Any restrictions needed?
4. **Modifiable Risk Factors**: What can be addressed?

Keep response concise (200-300 words) and actionable.
"""

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a cardiac electrophysiology specialist."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=800
            )

            return chat_completion.choices[0].message.content

        except Exception as e:
            return f"Risk factor analysis unavailable (API error: {str(e)})"

    def generate_patient_report_summary(
        self,
        features: Dict,
        prediction: int,
        probability: float
    ) -> str:
        """
        Generate patient-friendly summary for PDF reports.

        Args:
            features: ECG features dictionary
            prediction: Binary prediction
            probability: WPW probability

        Returns:
            Patient-friendly summary text
        """
        diagnosis = "abnormal WPW pattern" if prediction == 1 else "normal rhythm"

        prompt = f"""Create a brief, patient-friendly explanation of these ECG findings:

**Finding:** The ECG analysis shows {diagnosis}
**Confidence:** {probability:.0%} probability

**Patient Age:** {features.get('age', 'N/A')} years
**Heart Rate:** {features.get('hr_mean', 'N/A'):.0f} bpm average

Write 2-3 short paragraphs (150-200 words total) explaining:
1. What this finding means in simple terms
2. Why it matters for heart health
3. What the patient should do next

Use clear, non-technical language suitable for patients without medical background.
Avoid jargon. Be reassuring but accurate.
"""

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a physician explaining medical results to a patient."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=500
            )

            return chat_completion.choices[0].message.content

        except Exception as e:
            # Fallback patient-friendly summary
            if prediction == 1:
                return """The ECG analysis detected a pattern that may indicate WPW syndrome, a condition where the heart has an extra electrical pathway. This can sometimes cause a fast heart rhythm. Your doctor will review these findings and may recommend seeing a heart specialist for further evaluation. Many people with WPW live normal, active lives, especially with proper monitoring and treatment if needed."""
            else:
                return """The ECG analysis shows a normal heart rhythm pattern without signs of WPW syndrome. Your heart's electrical system appears to be functioning normally based on these measurements. Continue with routine health monitoring and consult your doctor if you experience any new symptoms like rapid heartbeat, dizziness, or chest discomfort."""


def main():
    """Example usage of the ECG explainer."""
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set")
        print("Get free API key at: https://console.groq.com/keys")
        return

    # Initialize explainer
    explainer = ECGExplainer(api_key=api_key)

    # Example ECG features
    example_features = {
        'age': 45,
        'sex': 1,  # Male
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

    print("\n" + "="*70)
    print("AI-POWERED ECG ANALYSIS EXAMPLE")
    print("="*70)

    # Generate explanation
    explanation = explainer.explain_prediction(
        features=example_features,
        prediction=1,
        probability=0.87,
        risk_level='high',
        model_name='Random Forest Classifier'
    )

    print("\n" + explanation)


if __name__ == "__main__":
    main()
