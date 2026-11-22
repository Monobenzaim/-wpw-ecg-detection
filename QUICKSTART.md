# ⚡ Quick Start Guide

Get the WPW ECG Detection System running in 5 minutes!

## Prerequisites

- **Python 3.8+** installed
- **pip** package manager
- **Terminal/Command Prompt** access
- (Optional) **Groq API key** for AI explanations

## 🚀 5-Minute Setup

### Step 1: Get the Code

```bash
# If you have git
git clone <repository-url>
cd wpw-ecg-detection

# Or download and extract ZIP
# Navigate to the extracted folder
cd wpw-ecg-detection
```

### Step 2: Create Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

This will install all necessary packages (~5 minutes depending on your internet speed).

### Step 4: Setup Data

Your dataset should be in `data/wpw_dataset_features.csv` with this structure:

```
record_id,age,sex,dx,hr_mean,hr_std,hr_max,hrv_sdnn,hrv_rmssd,hrv_pnn50,
qrs_duration_ms,qrs_amplitude,signal_range,signal_std
JS36160,40,Female,1,128.79,23.57,148.51,0.129,0.188,35.0,29.62,85.43,4304,256.25
...
```

If you don't have data yet, the system includes example data for testing.

### Step 5: Train Models

```bash
cd src
python train.py
```

This will:
- Load your dataset
- Train multiple ML models
- Save trained models to `models/` directory
- Display performance metrics

**Expected output:**
```
Training: Logistic Regression with Polynomial Features
...
Best CV ROC-AUC: 0.9712
Accuracy:     0.9654
...
Model saved to: ../models/logistic_regression_wpw.pkl
```

### Step 6: Launch Web App

```bash
# Go back to project root
cd ..

# Launch Streamlit
streamlit run app_enhanced.py
```

**Your browser will automatically open to: `http://localhost:8501`**

If not, manually navigate to that URL.

## 🎯 First Prediction

### Option A: Use Example Data

1. In the web interface, select **"Use Example Data"**
2. Choose "Example 1 (WPW Pattern)"
3. Click **"🔬 Analyze ECG Features"**
4. View results and probability distribution

### Option B: Manual Entry

1. Select **"Manual Entry"**
2. Fill in ECG parameters:
   - Age: 45
   - Sex: Male
   - Mean Heart Rate: 128.79 bpm
   - QRS Duration: 29.62 ms
   - (etc.)
3. Click **"🔬 Analyze ECG Features"**

### Option C: Upload CSV

1. Prepare CSV file with ECG features
2. Select **"Upload CSV"**
3. Choose your file
4. Select record to analyze
5. Click **"🔬 Analyze ECG Features"**

## 🤖 Enable AI Explanations (Optional)

### Get Free Groq API Key

1. Visit **[https://console.groq.com/keys](https://console.groq.com/keys)**
2. Sign up (free, no credit card required)
3. Create new API key
4. Copy the key

### Add to Application

**Method 1: In the Web Interface (Quick)**
- Paste API key in sidebar under "🔑 Groq API Settings"
- API key is session-only (not saved permanently)

**Method 2: Environment Variable (Persistent)**

```bash
# Copy template
cp .env.example .env

# Edit .env file
nano .env  # or use notepad/VS Code

# Add your key:
GROQ_API_KEY=your_actual_key_here

# Save and restart app
```

Now AI-powered medical explanations will appear automatically!

## 📄 Generate PDF Reports

1. In sidebar, check **"Generate PDF Report"**
2. Enter physician name and institution (optional)
3. Run analysis
4. Click **"📥 Download PDF Report"** button
5. PDF is saved to `reports/` directory

## 📊 Run Benchmarking

### In Web Interface:
1. Go to **"📊 Model Benchmarking"** tab
2. Click **"🚀 Run Complete Benchmark"**
3. Wait for analysis (~2-5 minutes)
4. View results, ROC curves, confusion matrices

### Via Command Line:
```bash
cd src
python benchmark.py
```

Results saved to `benchmark_results/` directory.

## 🎓 For Thesis/Research

### Generate Publication-Ready Results

```bash
# 1. Train all models
cd src
python train.py

# 2. Run comprehensive benchmark
python benchmark.py

# 3. Results are in:
ls ../benchmark_results/
# - benchmark_results_YYYYMMDD_HHMMSS.csv
# - roc_curves_YYYYMMDD_HHMMSS.png
# - confusion_matrices_YYYYMMDD_HHMMSS.png
# - metrics_comparison_YYYYMMDD_HHMMSS.png
```

### Export Model Performance Table

Open `benchmark_results/benchmark_results_YYYYMMDD_HHMMSS.csv` in Excel or Google Sheets.

This contains all metrics ready for your thesis tables!

## 🛠️ Troubleshooting

### "Module not found" errors

```bash
# Ensure virtual environment is activated
# You should see (venv) in prompt

# Reinstall requirements
pip install -r requirements.txt
```

### Models directory doesn't exist

```bash
# Create manually
mkdir models

# Or train models (creates automatically)
cd src
python train.py
```

### Port 8501 already in use

```bash
# Use different port
streamlit run app_enhanced.py --server.port 8502
```

### Dataset not found

Ensure `data/wpw_dataset_features.csv` exists with correct format.

### Groq API errors

- Verify API key is correct
- Check internet connection
- Ensure no rate limits exceeded (wait a few minutes)
- Try regenerating API key at console.groq.com

## 📚 Next Steps

- **Read [README.md](README.md)** for comprehensive documentation
- **Check [GROQ_SETUP_GUIDE.md](GROQ_SETUP_GUIDE.md)** for detailed API setup
- **Review [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)** for evaluation details
- **Study [SECURITY.md](SECURITY.md)** for best practices
- **Explore source code** in `src/` directory

## 💡 Tips

- **Use example data** to test without training models
- **Train models overnight** if you have large dataset
- **Enable AI explanations** for better clinical insights
- **Export PDF reports** for documentation
- **Run benchmarks** to compare model performance
- **Keep API keys secure** - never commit to git

## ✅ Success Checklist

- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] Data in correct format
- [ ] Models trained successfully
- [ ] Web app launches without errors
- [ ] Can make predictions
- [ ] (Optional) Groq API key configured
- [ ] (Optional) PDF reports generate correctly

## 🆘 Still Having Issues?

1. Check [README.md](README.md) troubleshooting section
2. Review error messages carefully
3. Ensure all prerequisites are met
4. Try a fresh virtual environment
5. Open GitHub issue with error details

---

**Estimated Setup Time: 5-10 minutes** ⏱️

**You're ready to detect WPW syndrome with AI!** 🎉
