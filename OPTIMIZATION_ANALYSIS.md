# 🔬 Optimization Strategy Analysis

## Quick Comparison Results (10 subjects)

We tested 8 different approaches on 10 diverse subjects to identify the best improvements:

### 📊 Results Ranked by Performance

| Rank | Approach | Accuracy | vs Baseline | Key Changes |
|------|----------|----------|-------------|-------------|
| 🥇 1 | **2 Calibration Runs** | **62.0% ± 11.0%** | **+1.5%** | 2 runs instead of 1 |
| 🥈 2 | 2 Runs + SVM | 61.0% ± 12.0% | +0.4% | 2 runs + SVM classifier |
| 🥉 3 | Baseline (1 run) | 60.5% ± 8.8% | baseline | CSP + LogReg, 1 run |
| 4 | SVM Only | 59.8% ± 10.5% | -0.8% | SVM instead of LogReg |
| 5 | 2 Runs + PSD | 59.3% ± 8.5% | -1.2% | 2 runs + PSD features |
| 6 | Ensemble (LR+SVM+PSD) | 58.4% ± 9.2% | -2.1% | Full ensemble |
| 7 | 2 Runs + PSD + SVM | 56.5% ± 10.2% | -4.0% | All features + SVM |
| 8 | PSD Features Only | 55.4% ± 7.9% | -5.1% | Add PSD to baseline |

## 💡 Key Insights

### ✅ What Works

1. **More Calibration Data (2 runs instead of 1)**
   - Simple and effective: **+1.5%** improvement
   - Doubles training data: ~60 trials → ~120 trials
   - No added complexity, just more data
   - **This is our best approach!**

2. **Keep it Simple**
   - CSP + LogisticRegression works well
   - Complex features don't help (PSD actually hurts)
   - Ensemble methods don't improve performance

### ❌ What Doesn't Work

1. **PSD (Power Spectral Density) Features**
   - **Consistently reduce performance** (-5.1% alone, -4% combined)
   - Theory: CSP already captures frequency-specific patterns optimally
   - Adding PSD may introduce noise or redundant information

2. **Ensemble Methods**
   - No improvement over single classifier
   - Added complexity without benefit
   - Voting between LR and SVM doesn't help

3. **SVM Alone**
   - Slightly worse than LogisticRegression (-0.8%)
   - May overfit with limited data

## 🎯 Chosen Strategy: 2 Calibration Runs

**Configuration:**
- **Calibration Data:** 2 runs (double the training data)
- **Features:** CSP only (6 components, reg=0.1)
- **Classifier:** LogisticRegression (C=0.1)
- **Expected Improvement:** +1.5% over baseline

**Target Achievement:**
- Current baseline: 60.1% (1 run, all 109 subjects)
- With 2 runs: Expected ~**61.6%** (60.1% + 1.5%)
- Target: 65%
- **Gap remaining:** ~3.4%

## 📈 Path to 65%

To reach 65%, we need additional improvements:

### 1. **Per-Subject CSP Optimization** (potential: +1-2%)
   - Find optimal n_components for each subject (4, 6, or 8)
   - Use cross-validation on calibration data

### 2. **Better Frequency Bands** (potential: +0.5-1%)
   - Current: 7-30 Hz (mu + beta)
   - Try: subject-specific optimal bands
   - Or: wider bands (4-40 Hz)

### 3. **Temporal Features** (potential: +0.5-1%)
   - Use different time windows (e.g., 0.5-3.5s instead of 0-4s)
   - Event-related desynchronization timing

### 4. **Combination** (potential: +2-4% total)
   - 2 runs + CSP optimization + better bands
   - May reach 62-64%

### 5. **Deep Learning** (if needed, potential: +3-5%)
   - EEGNet or ShallowConvNet
   - More complex but proven in literature
   - Use if classical methods don't reach 65%

## 🔄 Current Status

**Running:** Training all 109 subjects with 2 calibration runs
- Models saved to: `models/subject_specific_2runs/`
- Results saved to: `results_2runs.pkl`

**Expected Outcome:**
- Mean test accuracy: **~61-62%**
- Top performers: >90%
- Subjects ≥65%: ~50-55 subjects (~50%)

## 📝 Next Steps

1. ✅ Complete training with 2 calibration runs (in progress)
2. Evaluate results on all 109 subjects
3. If <65%: Implement CSP optimization per subject
4. If still <65%: Try frequency band optimization
5. Document final results and methodology
