# Models Directory

This directory contains trained machine learning models for WPW syndrome detection.

## 📦 What Gets Stored Here

After running `python src/train.py`, you'll find:

- **Model Files** (`.pkl` files):
  - `logistic_regression_wpw.pkl`
  - `random_forest_wpw.pkl`
  - `svm_wpw.pkl`
  - `gradient_boosting_wpw.pkl`
  - Additional models as trained

- **Metadata Files** (`.json` files):
  - `logistic_regression_wpw_metadata.json`
  - `random_forest_wpw_metadata.json`
  - etc.

- **Feature Importance Plots** (`.png` files):
  - Generated for tree-based models
  - Shows which ECG features are most predictive

## 🔒 Git Ignore

**Model files (`.pkl`) are ignored by Git** because:
- They can be large (10MB+)
- They're regeneratable from source code
- Different for each dataset/training run
- May cause merge conflicts

**Metadata files (`.json`) are kept** because:
- They're small (<1KB)
- They document model configuration
- They help reproduce results

## 🚀 Usage

### Train Models

```bash
cd src
python train.py
```

### Load Model for Prediction

```python
from src.predict import WPWPredictor

# Load specific model
predictor = WPWPredictor("models/random_forest_wpw.pkl")

# Or load best model automatically
from src.predict import load_best_model
predictor = load_best_model("models")
```

## 📊 Model Comparison

To compare all models, use the benchmarking system:

```bash
cd src
python benchmark.py
```

## 🔄 Regenerating Models

If models directory is empty (e.g., after cloning from Git):

```bash
# Navigate to src
cd src

# Train all models
python train.py

# Models will be saved to ../models/
```

Expected output:
```
Training: Logistic Regression with Polynomial Features
...
Model saved to: ../models/logistic_regression_wpw.pkl
Metadata saved to: ../models/logistic_regression_wpw_metadata.json
```

## 📝 Model Metadata

Each model has accompanying metadata including:
- Model name and type
- Training date and time
- Feature names used
- Test set performance metrics
- Random seed for reproducibility
- Dataset path used for training

Example metadata:
```json
{
  "model_name": "Random_Forest",
  "training_date": "2026-11-22 14:30:45",
  "features": ["age", "sex", "hr_mean", ...],
  "metrics": {
    "accuracy": 0.9823,
    "roc_auc": 0.9856,
    "sensitivity": 0.9654,
    ...
  },
  "random_state": 42
}
```

## 🛠️ Troubleshooting

### No models found

**Problem**: Directory empty or models not generated

**Solution**:
```bash
cd src
python train.py
```

### Model loading errors

**Problem**: Model trained with different scikit-learn version

**Solution**:
- Retrain models with current environment
- Or match scikit-learn version from metadata

### Large model files

**Problem**: Models too large for some deployment scenarios

**Solutions**:
- Use simpler model (Logistic Regression vs Random Forest)
- Reduce model complexity (fewer trees, lower depth)
- Use model compression techniques
- Deploy via model serving API

## 📚 For Production Deployment

When deploying to production:

1. **Version control models separately**:
   - Use Git LFS for large files
   - Or store in model registry (MLflow, etc.)
   - Or cloud storage (S3, Azure Blob)

2. **Include checksums**:
   ```bash
   sha256sum random_forest_wpw.pkl > checksums.txt
   ```

3. **Document model versions**:
   - Keep changelog of model updates
   - Tag significant model releases
   - Track performance over time

4. **Implement model monitoring**:
   - Track prediction distribution
   - Monitor for data drift
   - Set up alerts for degraded performance

---

**Models will be generated here after training!** 🤖
