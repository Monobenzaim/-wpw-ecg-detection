# 📊 Benchmarking Guide

Comprehensive guide to model evaluation and benchmarking in the WPW ECG Detection System.

---

## 📋 Table of Contents

- [What is Benchmarking?](#what-is-benchmarking)
- [Why Benchmark?](#why-benchmark)
- [Quick Start](#quick-start)
- [Understanding Metrics](#understanding-metrics)
- [Running Benchmarks](#running-benchmarks)
- [Interpreting Results](#interpreting-results)
- [For Thesis/Research](#for-thesisresearch)
- [Advanced Usage](#advanced-usage)

---

## What is Benchmarking?

**Benchmarking** is the systematic comparison of multiple machine learning models to determine which performs best for a specific task.

In this project, we benchmark **7+ algorithms** for WPW syndrome detection using **20+ medical diagnostic metrics**.

### Models Compared

1. **Logistic Regression** (with Polynomial Features)
2. **Random Forest Classifier**
3. **Gradient Boosting Classifier**
4. **Support Vector Machine (SVM)**
5. **Decision Tree Classifier**
6. **K-Nearest Neighbors (KNN)**
7. **Naive Bayes**
8. **AdaBoost**
9. **XGBoost** (if installed)

### Evaluation Metrics

**Classification Metrics:**
- Accuracy
- Balanced Accuracy
- Precision (PPV)
- Recall (Sensitivity)
- F1-Score

**Medical Diagnostic Metrics:**
- Sensitivity (True Positive Rate)
- Specificity (True Negative Rate)
- PPV (Positive Predictive Value)
- NPV (Negative Predictive Value)
- Likelihood Ratios (LR+, LR-)

**Advanced Metrics:**
- ROC-AUC (Area Under ROC Curve)
- PR-AUC (Precision-Recall AUC)
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- Youden's Index
- Diagnostic Odds Ratio (DOR)

**Statistical:**
- Cross-validation scores (5-fold)
- Confusion matrices
- Standard deviations

---

## Why Benchmark?

### For Clinical Applications

1. **Safety**: Identify model with highest sensitivity (fewer missed cases)
2. **Efficiency**: Find balance between sensitivity and specificity
3. **Confidence**: Understand model limitations and strengths
4. **Validation**: Demonstrate rigorous evaluation

### For Research/Thesis

1. **Methodology**: Show systematic model selection process
2. **Comparison**: Justify chosen algorithm
3. **Publication**: Provide comprehensive results for papers
4. **Reproducibility**: Document experimental setup
5. **Visualizations**: Create publication-ready figures

### For Development

1. **Model Selection**: Choose best algorithm for deployment
2. **Performance Tracking**: Monitor changes over time
3. **Optimization**: Identify areas for improvement
4. **Debugging**: Understand failure modes

---

## Quick Start

### Method 1: Web Interface

1. Launch app:
   ```bash
   streamlit run app_enhanced.py
   ```

2. Navigate to **"📊 Model Benchmarking"** tab

3. Click **"🚀 Run Complete Benchmark"**

4. Wait for completion (~2-5 minutes)

5. View results:
   - Comparison table
   - ROC curves
   - Confusion matrices
   - Metrics visualization

### Method 2: Command Line

```bash
cd src
python benchmark.py
```

**Output location:** `benchmark_results/`

**Files generated:**
- `benchmark_results_YYYYMMDD_HHMMSS.csv` - Full metrics table
- `benchmark_detailed_YYYYMMDD_HHMMSS.json` - Complete results
- `roc_curves_YYYYMMDD_HHMMSS.png` - ROC curve plot
- `confusion_matrices_YYYYMMDD_HHMMSS.png` - All confusion matrices
- `metrics_comparison_YYYYMMDD_HHMMSS.png` - Bar chart comparison

---

## Understanding Metrics

### Medical Diagnostic Metrics Explained

#### Sensitivity (Recall / True Positive Rate)

**Definition**: Proportion of actual WPW cases correctly identified

```
Sensitivity = TP / (TP + FN)
```

**Clinical meaning**:
- **High sensitivity (95%+)**: Few WPW cases missed - Important for screening
- **Low sensitivity (<80%)**: Many WPW cases missed - Dangerous!

**Example**:
- 100 WPW patients tested
- Model detects 95
- Sensitivity = 95/100 = 95%

**When prioritize**:
- Initial screening
- When missing a case is dangerous
- Low prevalence conditions

#### Specificity (True Negative Rate)

**Definition**: Proportion of normal cases correctly identified

```
Specificity = TN / (TN + FP)
```

**Clinical meaning**:
- **High specificity (95%+)**: Few false alarms
- **Low specificity (<80%)**: Many unnecessary follow-ups

**Example**:
- 100 normal patients tested
- Model correctly identifies 92 as normal
- Specificity = 92/100 = 92%

**When prioritize**:
- Confirmatory testing
- When false positives are costly
- High prevalence conditions

#### Positive Predictive Value (PPV / Precision)

**Definition**: Of those predicted WPW+, how many actually have WPW?

```
PPV = TP / (TP + FP)
```

**Clinical meaning**:
- **High PPV (90%+)**: Positive prediction reliable
- **Low PPV (<70%)**: Many false alarms

**Depends on**: Disease prevalence (↑ prevalence = ↑ PPV)

#### Negative Predictive Value (NPV)

**Definition**: Of those predicted WPW-, how many are truly normal?

```
NPV = TN / (TN + FN)
```

**Clinical meaning**:
- **High NPV (95%+)**: Negative prediction reliable
- **Low NPV (<85%)**: Might miss cases

**Depends on**: Disease prevalence (↓ prevalence = ↑ NPV)

#### F1-Score

**Definition**: Harmonic mean of Precision and Recall

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Range**: 0 (worst) to 1 (best)

**Use**: Overall performance balance, especially with class imbalance

#### ROC-AUC

**Definition**: Area Under Receiver Operating Characteristic curve

**Range**: 0.5 (random) to 1.0 (perfect)

**Interpretation**:
- **0.90-1.0**: Excellent
- **0.80-0.90**: Good
- **0.70-0.80**: Fair
- **0.60-0.70**: Poor
- **0.50-0.60**: Fail

**Advantage**: Threshold-independent metric

#### Matthews Correlation Coefficient (MCC)

**Definition**: Correlation between predicted and actual classes

**Range**: -1 (complete disagreement) to +1 (perfect prediction)

**Interpretation**:
- **0.90+**: Excellent
- **0.70-0.90**: Good
- **0.50-0.70**: Moderate
- **<0.50**: Poor

**Advantage**: Considers all confusion matrix cells, robust to imbalance

#### Youden's Index

**Definition**: Sensitivity + Specificity - 1

**Range**: 0 (useless) to 1 (perfect)

**Use**: Optimal threshold selection on ROC curve

---

## Running Benchmarks

### Standard Benchmark

```bash
cd src
python benchmark.py
```

**What it does:**
1. Loads dataset from `data/wpw_dataset_features.csv`
2. Splits into train/test (80/20)
3. Trains all models with hyperparameter tuning
4. Evaluates on test set
5. Performs 5-fold cross-validation
6. Generates visualizations
7. Exports results to CSV/JSON

**Duration**: 2-10 minutes (depends on dataset size)

### Custom Benchmark

```python
from src.benchmark import WPWBenchmark

# Initialize with custom settings
benchmark = WPWBenchmark(
    data_path="data/wpw_dataset_features.csv",
    test_size=0.2,          # 20% test set
    random_state=42,        # For reproducibility
    cv_folds=5              # 5-fold cross-validation
)

# Run complete benchmark
benchmark.run_complete_benchmark()

# Generate visualizations
benchmark.generate_all_visualizations()

# Export results
benchmark.export_results()

# Print summary
benchmark.print_results_summary()
```

### Benchmark Specific Models

```python
from src.benchmark import WPWBenchmark

benchmark = WPWBenchmark(data_path="data/wpw_dataset_features.csv")
benchmark.load_and_prepare_data()

# Evaluate only Random Forest
benchmark.evaluate_model(
    "Random Forest",
    benchmark._define_models()["Random Forest"]
)

# Compare just 3 models
for model_name in ["Logistic Regression", "Random Forest", "SVM (RBF)"]:
    model = benchmark._define_models()[model_name]
    benchmark.evaluate_model(model_name, model)

benchmark.print_results_summary()
```

---

## Interpreting Results

### Reading the Comparison Table

```
Model                 accuracy  roc_auc  sensitivity  specificity  f1_score
Random Forest         0.9823    0.9856   0.9654       0.9712       0.9683
Gradient Boosting     0.9801    0.9834   0.9601       0.9689       0.9645
XGBoost               0.9789    0.9821   0.9578       0.9655       0.9617
...
```

**What to look for:**

1. **Top ROC-AUC**: Best overall discriminative ability
2. **High Sensitivity**: Fewest missed WPW cases
3. **High Specificity**: Fewest false alarms
4. **Balanced F1-Score**: Good precision-recall balance
5. **Low CV std**: Consistent performance across folds

### Analyzing ROC Curves

**Components:**
- **X-axis**: False Positive Rate (1 - Specificity)
- **Y-axis**: True Positive Rate (Sensitivity)
- **Diagonal line**: Random classifier (AUC = 0.5)
- **Curve position**: Closer to top-left = better
- **AUC value**: Area under curve (higher = better)

**Interpretation:**
- **Curve dominates others**: Clear winner
- **Curves overlap**: Models perform similarly
- **Curve near diagonal**: Model not learning

### Understanding Confusion Matrices

```
                 Predicted
              Normal    WPW
Actual Normal   TN      FP
       WPW      FN      TP
```

**Ideal pattern:**
- **Large TN and TP**: Main diagonal
- **Small FN and FP**: Off-diagonal

**Problem patterns:**
- **Large FN**: Missing WPW cases (↓ Sensitivity)
- **Large FP**: False alarms (↓ Specificity)
- **Balanced errors**: Need better features or model

### Comparing Models

**Scenario 1: Clinical Screening**

**Priority**: High Sensitivity (don't miss WPW cases)

```
Choose: Highest Sensitivity model
Accept: Lower Specificity (more false positives acceptable)
```

**Scenario 2: Confirmatory Testing**

**Priority**: High Specificity (avoid false diagnoses)

```
Choose: Highest Specificity model
Accept: Lower Sensitivity (follow-up with more tests)
```

**Scenario 3: Balanced Application**

**Priority**: Overall performance

```
Choose: Highest ROC-AUC or F1-Score
Consider: MCC for imbalanced data
```

---

## For Thesis/Research

### Publication-Ready Results

#### 1. Performance Table

Export `benchmark_results_*.csv` to Excel/Word:

```
Table 1: Model Performance Comparison

Model              Accuracy  Sensitivity  Specificity  ROC-AUC  F1-Score
Random Forest      0.9823    0.9654      0.9712       0.9856   0.9683
Gradient Boosting  0.9801    0.9601      0.9689       0.9834   0.9645
...
```

#### 2. ROC Curve Figure

Use generated `roc_curves_*.png`:

```
Figure 1: Receiver Operating Characteristic curves for all evaluated
models. Random Forest achieved the highest AUC of 0.9856, followed by
Gradient Boosting (0.9834).
```

#### 3. Confusion Matrices

Use `confusion_matrices_*.png`:

```
Figure 2: Confusion matrices showing true positive, true negative,
false positive, and false negative counts for each model on the test set.
```

#### 4. Statistical Analysis

Include cross-validation results:

```
The Random Forest model achieved a mean cross-validation ROC-AUC of
0.9823 ± 0.0145 (SD) across 5 folds, demonstrating robust and consistent
performance.
```

### Writing Methodology Section

**Example text for thesis:**

```
Model Evaluation and Selection

We evaluated seven machine learning algorithms for WPW syndrome detection:
Logistic Regression with polynomial features (degree=2), Random Forest,
Gradient Boosting, Support Vector Machine (RBF kernel), Decision Tree,
K-Nearest Neighbors, and Naive Bayes.

Data Splitting: The dataset was split into training (80%, n=XXX) and test
(20%, n=XXX) sets using stratified sampling to maintain class distribution.

Hyperparameter Optimization: Each model underwent hyperparameter tuning using
GridSearchCV with 5-fold stratified cross-validation, optimizing for ROC-AUC.

Evaluation Metrics: Models were assessed using 20 metrics including accuracy,
sensitivity, specificity, positive/negative predictive values, F1-score,
ROC-AUC, PR-AUC, Matthews correlation coefficient, Cohen's kappa, and
Youden's index.

Statistical Validation: Performance consistency was validated using 5-fold
cross-validation, reporting mean and standard deviation of ROC-AUC scores.

Results: Random Forest achieved the highest test set performance with ROC-AUC
of 0.9856, sensitivity of 0.9654, and specificity of 0.9712, demonstrating
superior discriminative ability for WPW syndrome detection.
```

### Citing Results

```bibtex
@mastersthesis{your_thesis_2026,
  title={AI-Powered WPW Syndrome Detection from ECG Features},
  author={Benzaim, Abdelmouaine},
  year={2026},
  school={Your University},
  note={Random Forest model achieved ROC-AUC of 0.9856 for WPW detection}
}
```

---

## Advanced Usage

### Custom Metrics

Add your own evaluation metrics:

```python
from src.benchmark import WPWBenchmark

class CustomBenchmark(WPWBenchmark):
    def _calculate_medical_metrics(self, y_true, y_pred, y_prob):
        metrics = super()._calculate_medical_metrics(y_true, y_pred, y_prob)

        # Add custom metric
        metrics['custom_score'] = your_custom_function(y_true, y_pred)

        return metrics

# Use custom benchmark
benchmark = CustomBenchmark(data_path="data/wpw_dataset_features.csv")
benchmark.run_complete_benchmark()
```

### Stratified Analysis

Evaluate by subgroups (age, sex, etc.):

```python
# By age group
young = df[df['age'] < 50]
old = df[df['age'] >= 50]

for name, subset in [("Young", young), ("Old", old)]:
    benchmark = WPWBenchmark(data_path=subset)
    print(f"\nResults for {name} patients:")
    benchmark.run_complete_benchmark()
```

### Time-Series Cross-Validation

For temporal data:

```python
from sklearn.model_selection import TimeSeriesSplit

# Custom CV strategy
tscv = TimeSeriesSplit(n_splits=5)

grid_search = GridSearchCV(
    model, param_grid,
    cv=tscv,  # Use time-series CV
    scoring='roc_auc'
)
```

### Bootstrap Confidence Intervals

```python
from sklearn.utils import resample
import numpy as np

def bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap=1000):
    scores = []
    for _ in range(n_bootstrap):
        indices = resample(range(len(y_true)), random_state=_)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        scores.append(metric_func(y_true_boot, y_pred_boot))

    return np.percentile(scores, [2.5, 97.5])  # 95% CI

# Usage
from sklearn.metrics import accuracy_score
ci = bootstrap_metric(y_test, y_pred, accuracy_score)
print(f"Accuracy 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
```

---

## Troubleshooting

### "No models found"

**Solution**: Train models first
```bash
cd src
python train.py
```

### Slow benchmark execution

**Solutions:**
- Reduce CV folds: `cv_folds=3`
- Smaller param grids
- Remove slow models (SVM with large datasets)
- Use fewer models initially

### Out of memory

**Solutions:**
- Reduce dataset size
- Use simpler models
- Decrease max_depth for tree models
- Run models sequentially, not all at once

### Inconsistent results

**Solutions:**
- Set `random_state=42` everywhere
- Use same train/test split
- Document software versions
- Save random seeds used

---

## Summary

✅ **Benchmarking Checklist:**

1. [ ] Dataset prepared and clean
2. [ ] Models trained successfully
3. [ ] Benchmark script executed
4. [ ] Results saved to `benchmark_results/`
5. [ ] Visualizations generated
6. [ ] Best model identified
7. [ ] Results interpreted correctly
8. [ ] (Thesis) Figures/tables exported
9. [ ] (Thesis) Methodology documented
10. [ ] (Production) Best model deployed

**Key Takeaways:**

- **For Screening**: Prioritize Sensitivity (high TP rate)
- **For Confirmation**: Prioritize Specificity (low FP rate)
- **For Research**: Report all metrics comprehensively
- **For Production**: Balance performance and interpretability

---

**Need help?** Check [README.md](README.md) or open a GitHub issue.

**Ready to benchmark your models!** 📊🚀
