"""
WPW ECG Detection - Enhanced Web Application

A comprehensive Streamlit web application for WPW syndrome detection from ECG features,
featuring AI-powered explanations, PDF report generation, and model benchmarking.

Features:
- Interactive ECG feature input
- Real-time WPW syndrome prediction
- AI-powered medical explanations (Groq LLM)
- Professional PDF report generation
- Comprehensive model benchmarking
- Beautiful, medical-grade UI

Author: Enhanced with Claude Code
Date: 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import custom modules
try:
    from predict import WPWPredictor, load_best_model
    from llm_explainer import ECGExplainer
    from report_generator import WPWReportGenerator
    from benchmark import WPWBenchmark
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    st.error(f"⚠️ Error importing modules: {e}")

# Page config
st.set_page_config(
    page_title="WPW ECG Detection System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 4px solid #d32f2f;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .risk-moderate {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application function."""

    # Header
    st.markdown('<p class="main-header">❤️ WPW ECG Detection System</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-Powered Wolff-Parkinson-White Syndrome Screening</p>',
        unsafe_allow_html=True
    )

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/electrocardiogram.png", width=80)
        st.title("⚙️ Configuration")

        # API Key input
        st.subheader("🔑 Groq API Settings")
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Enter your Groq API key for AI-powered medical explanations. Get free key at: https://console.groq.com/keys"
        )

        if groq_api_key:
            st.success("✅ API key configured")
        else:
            st.info("💡 Add API key to enable AI explanations")

        st.markdown("---")

        # Model selection
        st.subheader("🤖 Model Selection")
        model_choice = st.selectbox(
            "Select Prediction Model",
            ["Best Model (Auto)", "Logistic Regression", "Random Forest", "SVM", "Gradient Boosting"],
            help="Choose which trained model to use for predictions"
        )

        st.markdown("---")

        # Report options
        st.subheader("📄 Report Options")
        generate_pdf = st.checkbox("Generate PDF Report", value=True)

        if generate_pdf:
            doctor_name = st.text_input("Physician Name", value="Dr. Cardiologist")
            hospital_name = st.text_input("Institution", value="Medical Center")

        st.markdown("---")

        # Links
        st.markdown("### 📚 Resources")
        st.markdown("- [What is WPW?](https://www.heart.org/en/health-topics/arrhythmia/about-arrhythmia/wolff-parkinson-white-syndrome)")
        st.markdown("- [Get Groq API Key](https://console.groq.com/keys)")
        st.markdown("- [Project Documentation](#)")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 WPW Detection",
        "📊 Model Benchmarking",
        "📖 About WPW Syndrome",
        "ℹ️ User Guide"
    ])

    # Tab 1: WPW Detection
    with tab1:
        st.header("ECG-Based WPW Syndrome Detection")

        # Input method selection
        input_method = st.radio(
            "Choose Input Method",
            ["Manual Entry", "Upload CSV", "Use Example Data"],
            horizontal=True
        )

        if input_method == "Manual Entry":
            features = manual_input_form()
        elif input_method == "Upload CSV":
            features = upload_csv_form()
        else:
            features = load_example_data()

        if features is not None:
            st.markdown("---")

            # Prediction button
            if st.button("🔬 Analyze ECG Features", type="primary", use_container_width=True):
                with st.spinner("Analyzing ECG data..."):
                    run_prediction(
                        features, groq_api_key, generate_pdf,
                        doctor_name if generate_pdf else "Dr. Cardiologist",
                        hospital_name if generate_pdf else "Medical Center",
                        model_choice
                    )

    # Tab 2: Benchmarking
    with tab2:
        st.header("Model Performance Benchmarking")
        st.markdown("""
        Compare multiple machine learning algorithms for WPW detection using
        comprehensive medical diagnostic metrics.
        """)

        if st.button("🚀 Run Complete Benchmark", type="primary"):
            run_benchmark()

        # Display existing results if available
        display_benchmark_results()

    # Tab 3: About WPW
    with tab3:
        display_wpw_information()

    # Tab 4: User Guide
    with tab4:
        display_user_guide()


def manual_input_form() -> dict:
    """Create manual input form for ECG features."""
    st.subheader("📝 Enter ECG Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=45, step=1)
        sex = st.selectbox("Sex", ["Male", "Female"])

        st.markdown("**Heart Rate**")
        hr_mean = st.number_input("Mean Heart Rate (bpm)", min_value=30.0, max_value=220.0, value=75.0, step=1.0)
        hr_std = st.number_input("HR Std Deviation", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
        hr_max = st.number_input("Maximum HR (bpm)", min_value=40.0, max_value=250.0, value=85.0, step=1.0)

    with col2:
        st.markdown("**Heart Rate Variability**")
        hrv_sdnn = st.number_input("HRV SDNN", min_value=0.0, max_value=1.0, value=0.05, step=0.001, format="%.4f")
        hrv_rmssd = st.number_input("HRV RMSSD", min_value=0.0, max_value=1.0, value=0.03, step=0.001, format="%.4f")
        hrv_pnn50 = st.number_input("HRV pNN50 (%)", min_value=0.0, max_value=100.0, value=20.0, step=0.1)

    with col3:
        st.markdown("**QRS Complex**")
        qrs_duration = st.number_input("QRS Duration (ms)", min_value=0.0, max_value=200.0, value=90.0, step=1.0)
        qrs_amplitude = st.number_input("QRS Amplitude", min_value=0.0, max_value=1000.0, value=100.0, step=1.0)

        st.markdown("**Signal Characteristics**")
        signal_range = st.number_input("Signal Range", min_value=0.0, max_value=10000.0, value=3000.0, step=10.0)
        signal_std = st.number_input("Signal Std Dev", min_value=0.0, max_value=1000.0, value=200.0, step=1.0)

    features = {
        'age': age,
        'sex': sex,
        'hr_mean': hr_mean,
        'hr_std': hr_std,
        'hr_max': hr_max,
        'hrv_sdnn': hrv_sdnn,
        'hrv_rmssd': hrv_rmssd,
        'hrv_pnn50': hrv_pnn50,
        'qrs_duration_ms': qrs_duration,
        'qrs_amplitude': qrs_amplitude,
        'signal_range': signal_range,
        'signal_std': signal_std
    }

    return features


def upload_csv_form():
    """CSV upload form."""
    st.subheader("📂 Upload ECG Data")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} records")

            # Show preview
            st.dataframe(df.head(), use_container_width=True)

            # Select record
            if len(df) > 1:
                record_idx = st.selectbox("Select Record", range(len(df)))
                return df.iloc[record_idx].to_dict()
            else:
                return df.iloc[0].to_dict()

        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return None

    return None


def load_example_data() -> dict:
    """Load example ECG data."""
    st.subheader("📋 Example ECG Data")

    examples = {
        "Example 1 (WPW Pattern)": {
            'age': 45, 'sex': 'Male',
            'hr_mean': 128.79, 'hr_std': 23.57, 'hr_max': 148.51,
            'hrv_sdnn': 0.129, 'hrv_rmssd': 0.189, 'hrv_pnn50': 35.0,
            'qrs_duration_ms': 29.62, 'qrs_amplitude': 85.43,
            'signal_range': 4304, 'signal_std': 256.25
        },
        "Example 2 (Normal)": {
            'age': 50, 'sex': 'Female',
            'hr_mean': 72.0, 'hr_std': 8.5, 'hr_max': 85.0,
            'hrv_sdnn': 0.045, 'hrv_rmssd': 0.030, 'hrv_pnn50': 18.5,
            'qrs_duration_ms': 95.0, 'qrs_amplitude': 120.0,
            'signal_range': 2800, 'signal_std': 180.0
        }
    }

    example_name = st.selectbox("Select Example", list(examples.keys()))
    features = examples[example_name]

    # Display as table
    df_display = pd.DataFrame([features]).T
    df_display.columns = ['Value']
    st.dataframe(df_display, use_container_width=True)

    return features


def run_prediction(features, groq_api_key, generate_pdf, doctor_name, hospital_name, model_choice):
    """Run prediction and display results."""
    try:
        # Load model
        with st.spinner("Loading prediction model..."):
            if model_choice == "Best Model (Auto)":
                predictor = load_best_model("models")
            else:
                model_map = {
                    "Logistic Regression": "logistic_regression_wpw.pkl",
                    "Random Forest": "random_forest_wpw.pkl",
                    "SVM": "svm_wpw.pkl",
                    "Gradient Boosting": "gradient_boosting_wpw.pkl"
                }
                model_file = model_map.get(model_choice)
                if model_file:
                    predictor = WPWPredictor(f"models/{model_file}")
                else:
                    predictor = load_best_model("models")

        # Make prediction
        result = predictor.explain_prediction(features)

        # Display results
        st.success("✅ Analysis Complete!")

        # Main result card
        prediction = result['prediction']
        wpw_prob = result['wpw_probability']
        risk_level = result['risk_level']

        if prediction == 1:
            st.markdown(f"""
            <div class="risk-high">
                <h2>⚠️ WPW Syndrome Pattern Detected</h2>
                <p><strong>WPW Probability:</strong> {wpw_prob:.1%}</p>
                <p><strong>Risk Level:</strong> {risk_level.upper()}</p>
                <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
                <h2>✓ Normal Sinus Rhythm</h2>
                <p><strong>WPW Probability:</strong> {wpw_prob:.1%}</p>
                <p><strong>Risk Level:</strong> {risk_level.upper()}</p>
                <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

        # Probability visualization
        st.subheader("📊 Probability Distribution")
        fig = go.Figure(go.Bar(
            x=['Normal', 'WPW Syndrome'],
            y=[result['probabilities']['normal'], result['probabilities']['wpw']],
            marker_color=['green', 'red']
        ))
        fig.update_layout(
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

        # Clinical recommendation
        st.subheader("🏥 Clinical Recommendation")
        st.info(result['recommendation'])

        # AI Explanation
        if groq_api_key:
            with st.expander("🤖 AI-Powered Medical Analysis", expanded=False):
                with st.spinner("Generating AI explanation..."):
                    try:
                        explainer = ECGExplainer(api_key=groq_api_key)
                        explanation = explainer.explain_prediction(
                            features, prediction, wpw_prob, risk_level, predictor.model_name
                        )
                        st.markdown(explanation)
                    except Exception as e:
                        st.error(f"Error generating explanation: {e}")
        else:
            st.info("💡 Add Groq API key in sidebar to enable AI-powered explanations")

        # PDF Report
        if generate_pdf:
            with st.spinner("Generating PDF report..."):
                try:
                    generator = WPWReportGenerator()

                    # Get AI explanation if available
                    llm_explanation = None
                    if groq_api_key:
                        try:
                            explainer = ECGExplainer(api_key=groq_api_key)
                            llm_explanation = explainer.explain_prediction(
                                features, prediction, wpw_prob, risk_level, predictor.model_name
                            )
                        except:
                            pass

                    # Patient info
                    patient_info = {
                        'name': 'Patient Name',
                        'id': 'P' + str(np.random.randint(100000, 999999)),
                        'age': features['age'],
                        'sex': features['sex']
                    }

                    # Generate report
                    report_path = generator.generate_report(
                        patient_info=patient_info,
                        features=features,
                        prediction_result=result,
                        llm_explanation=llm_explanation,
                        model_name=predictor.model_name,
                        doctor_name=doctor_name,
                        hospital_name=hospital_name
                    )

                    # Download button
                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=f,
                            file_name=Path(report_path).name,
                            mime="application/pdf",
                            use_container_width=True
                        )

                    st.success(f"✅ Report generated: {Path(report_path).name}")

                except Exception as e:
                    st.error(f"Error generating PDF: {e}")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
        st.exception(e)


def run_benchmark():
    """Run model benchmarking."""
    with st.spinner("Running comprehensive benchmark... This may take several minutes."):
        try:
            benchmark = WPWBenchmark(
                data_path="data/wpw_dataset_features.csv",
                test_size=0.2,
                random_state=42
            )

            # Progress updates
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Loading dataset...")
            progress_bar.progress(10)

            # Run benchmark
            benchmark.run_complete_benchmark()
            progress_bar.progress(60)

            status_text.text("Generating visualizations...")
            benchmark.generate_all_visualizations()
            progress_bar.progress(80)

            status_text.text("Exporting results...")
            benchmark.export_results()
            progress_bar.progress(100)

            status_text.text("✅ Benchmark complete!")

            # Display results
            st.balloons()
            display_benchmark_results()

        except Exception as e:
            st.error(f"Error running benchmark: {e}")
            st.exception(e)


def display_benchmark_results():
    """Display benchmark results if available."""
    results_dir = Path("benchmark_results")

    if not results_dir.exists():
        st.info("No benchmark results available yet. Run a benchmark to see results.")
        return

    # Find latest results
    csv_files = list(results_dir.glob("benchmark_results_*.csv"))

    if csv_files:
        latest_csv = max(csv_files, key=os.path.getctime)

        st.subheader("📊 Latest Benchmark Results")
        df = pd.read_csv(latest_csv)

        # Display table
        st.dataframe(df.style.highlight_max(axis=0, subset=['accuracy', 'roc_auc', 'sensitivity', 'specificity']),
                    use_container_width=True)

        # Display images
        image_files = {
            "ROC Curves": list(results_dir.glob("roc_curves_*.png")),
            "Confusion Matrices": list(results_dir.glob("confusion_matrices_*.png")),
            "Metrics Comparison": list(results_dir.glob("metrics_comparison_*.png"))
        }

        for title, files in image_files.items():
            if files:
                latest_image = max(files, key=os.path.getctime)
                with st.expander(f"📈 {title}"):
                    st.image(str(latest_image), use_container_width=True)


def display_wpw_information():
    """Display information about WPW syndrome."""
    st.subheader("About Wolff-Parkinson-White (WPW) Syndrome")

    st.markdown("""
    ### What is WPW Syndrome?

    **Wolff-Parkinson-White (WPW) syndrome** is a congenital heart condition characterized by
    the presence of an **accessory electrical pathway** (Bundle of Kent) between the atria
    and ventricles.

    ### Clinical Significance

    - **Pre-excitation Pattern**: Extra pathway causes early ventricular activation
    - **Arrhythmia Risk**: Can lead to supraventricular tachycardia (SVT)
    - **ECG Findings**: Short PR interval, delta wave, widened QRS complex
    - **Prevalence**: Affects 0.1-0.3% of the general population

    ### Symptoms

    - Rapid heartbeat (palpitations)
    - Dizziness or lightheadedness
    - Shortness of breath
    - Chest discomfort
    - Syncope (fainting)

    ### Treatment Options

    1. **Observation**: Asymptomatic cases may require only monitoring
    2. **Medications**: Antiarrhythmic drugs for symptom control
    3. **Catheter Ablation**: Definitive treatment to eliminate accessory pathway
    4. **Lifestyle Modifications**: Avoiding triggers, stress management

    ### Diagnostic Approach

    - **12-lead ECG**: Gold standard for diagnosis
    - **Holter Monitor**: 24-hour rhythm assessment
    - **Electrophysiology Study**: Detailed pathway mapping
    - **Exercise Stress Test**: Risk stratification

    ### Our AI System

    This application uses machine learning to analyze extracted ECG features and
    identify patterns consistent with WPW syndrome. It serves as a **screening tool**
    to assist healthcare providers in identifying patients who may require further
    evaluation by a cardiologist or electrophysiologist.

    **Important**: This system supports, but does not replace, clinical judgment
    and standard ECG interpretation by qualified medical professionals.
    """)


def display_user_guide():
    """Display user guide."""
    st.subheader("📖 User Guide")

    st.markdown("""
    ### Getting Started

    1. **Configure API Key** (Optional)
       - Get free Groq API key at [console.groq.com/keys](https://console.groq.com/keys)
       - Enter in sidebar to enable AI explanations

    2. **Input ECG Data**
       - **Manual Entry**: Enter ECG parameters individually
       - **CSV Upload**: Upload file with ECG features
       - **Example Data**: Use pre-loaded examples to test

    3. **Analyze**
       - Click "Analyze ECG Features" to run prediction
       - Review probability scores and risk classification
       - Read AI-generated medical analysis (if API key provided)
       - Download PDF report for clinical documentation

    ### ECG Parameters Explained

    - **Age/Sex**: Patient demographics
    - **Heart Rate**: Average, variability, and maximum HR
    - **HRV Metrics**: Heart rate variability indicators
    - **QRS Complex**: Ventricular depolarization characteristics
    - **Signal Properties**: ECG signal quality metrics

    ### Model Selection

    - **Best Model**: Automatically uses highest-performing model
    - **Specific Models**: Choose Logistic Regression, Random Forest, SVM, or Gradient Boosting

    ### Benchmarking

    - Compare multiple ML algorithms
    - View ROC curves, confusion matrices, and performance metrics
    - Identify best model for specific use cases

    ### PDF Reports

    Professional medical reports include:
    - Patient information
    - ECG feature analysis
    - AI prediction results
    - Clinical recommendations
    - AI-generated medical analysis

    ### Need Help?

    - Review the "About WPW Syndrome" tab for clinical background
    - Check [Project Documentation](#) for technical details
    - Contact support for assistance
    """)


if __name__ == "__main__":
    if MODULES_AVAILABLE:
        main()
    else:
        st.error("⚠️ Required modules not available. Please ensure all src files are present.")
