#!/usr/bin/env python3
"""
Optimized Subject-Specific BCI Training with Multiple Improvements:
1. Use 2 calibration runs instead of 1 (more training data)
2. Optimize CSP components based on validation
3. Add PSD features in addition to CSP
4. Use SVM with RBF kernel for better non-linear separation
5. Implement basic ensemble (voting between classifiers)

Target: ≥65% mean test accuracy
"""

import os
import numpy as np
import mne
from mne.decoding import CSP
from scipy.signal import welch
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import joblib
import pickle
from tqdm import tqdm


def compute_band_power(epochs_data, sfreq=160, bands=None):
    """
    Compute band power features (PSD) in specific frequency bands.
    
    Args:
        epochs_data: (n_epochs, n_channels, n_times)
        sfreq: sampling frequency
        bands: dict of frequency bands (default: motor imagery bands)
    
    Returns:
        features: (n_epochs, n_channels * n_bands)
    """
    if bands is None:
        # Motor imagery relevant bands
        bands = {
            'mu': (8, 13),      # Mu rhythm
            'beta': (13, 30),   # Beta rhythm
            'low_gamma': (30, 45)  # Low gamma
        }
    
    n_epochs, n_channels, n_times = epochs_data.shape
    n_bands = len(bands)
    features = np.zeros((n_epochs, n_channels * n_bands))
    
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            signal = epochs_data[epoch_idx, ch_idx, :]
            # Compute PSD using Welch method
            freqs, psd = welch(signal, fs=sfreq, nperseg=min(256, n_times))
            
            # Extract power in each band
            for band_idx, (band_name, (low, high)) in enumerate(bands.items()):
                freq_mask = (freqs >= low) & (freqs <= high)
                band_power = np.trapz(psd[freq_mask], freqs[freq_mask])
                features[epoch_idx, ch_idx * n_bands + band_idx] = np.log10(band_power + 1e-10)
    
    return features


def load_and_prepare_subject_data(subject_id, task='left_right', n_calib_runs=2):
    """
    Load data for one subject with multiple calibration runs.
    
    Args:
        subject_id: subject number (1-109)
        task: 'left_right' or 'hands_feet'
        n_calib_runs: number of runs to use for calibration (1 or 2)
    
    Returns:
        X_calib, y_calib, X_test, y_test
    """
    # Task configuration
    if task == 'left_right':
        event_ids = {'left': 2, 'right': 3}
        calib_runs = [3, 7][:n_calib_runs]  # Real movement runs
        test_runs = [4, 8, 12]               # Imagery runs
    else:  # hands_feet
        event_ids = {'hands': 2, 'feet': 3}
        calib_runs = [5, 9][:n_calib_runs]   # Real movement runs
        test_runs = [6, 10, 14]              # Imagery runs
    
    # Load calibration data
    X_calib_list = []
    y_calib_list = []
    
    for run in calib_runs:
        raw = mne.io.read_raw_edf(
            f'src/data/MNE-eegbci-data/files/eegmmidb/1.0.0/S{subject_id:03d}/S{subject_id:03d}R{run:02d}.edf',
            preload=True, verbose=False
        )
        raw.pick_types(eeg=True)
        raw.filter(7., 30., fir_design='firwin', verbose=False)
        
        events, _ = mne.events_from_annotations(raw, verbose=False)
        epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=0, tmax=4,
                           baseline=None, preload=True, verbose=False)
        
        X_calib_list.append(epochs.get_data())
        y_calib_list.append(epochs.events[:, -1])
    
    X_calib = np.concatenate(X_calib_list, axis=0)
    y_calib = np.concatenate(y_calib_list, axis=0)
    
    # Load test data
    X_test_list = []
    y_test_list = []
    
    for run in test_runs:
        raw = mne.io.read_raw_edf(
            f'src/data/MNE-eegbci-data/files/eegmmidb/1.0.0/S{subject_id:03d}/S{subject_id:03d}R{run:02d}.edf',
            preload=True, verbose=False
        )
        raw.pick_types(eeg=True)
        raw.filter(7., 30., fir_design='firwin', verbose=False)
        
        events, _ = mne.events_from_annotations(raw, verbose=False)
        epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=0, tmax=4,
                           baseline=None, preload=True, verbose=False)
        
        X_test_list.append(epochs.get_data())
        y_test_list.append(epochs.events[:, -1])
    
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    
    return X_calib, y_calib, X_test, y_test


def find_optimal_csp_components(X_calib, y_calib, max_components=8):
    """
    Find optimal number of CSP components using internal cross-validation.
    """
    from sklearn.model_selection import StratifiedKFold
    
    n_splits = min(3, min(np.bincount(y_calib)))
    if n_splits < 2:
        return 6  # Default
    
    best_n = 6
    best_score = 0
    
    for n_comp in [4, 6, 8]:
        if n_comp > X_calib.shape[1]:
            continue
        
        scores = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for train_idx, val_idx in skf.split(X_calib, y_calib):
            X_train, X_val = X_calib[train_idx], X_calib[val_idx]
            y_train, y_val = y_calib[train_idx], y_calib[val_idx]
            
            csp = CSP(n_components=n_comp, reg=0.1, log=True, norm_trace=False)
            X_train_csp = csp.fit_transform(X_train, y_train)
            X_val_csp = csp.transform(X_val)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_csp)
            X_val_scaled = scaler.transform(X_val_csp)
            
            clf = LogisticRegression(max_iter=500, random_state=42)
            clf.fit(X_train_scaled, y_train)
            scores.append(clf.score(X_val_scaled, y_val))
        
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_n = n_comp
    
    return best_n


def train_subject_optimized(subject_id, task='left_right', n_calib_runs=2, use_ensemble=True):
    """
    Train optimized model for one subject with multiple improvements.
    
    Returns:
        model_dict: dictionary containing all model components
        calib_acc: calibration accuracy
        test_acc: test accuracy
    """
    # Load data
    X_calib, y_calib, X_test, y_test = load_and_prepare_subject_data(
        subject_id, task, n_calib_runs
    )
    
    # Find optimal CSP components
    n_components = find_optimal_csp_components(X_calib, y_calib)
    
    # Extract CSP features
    csp = CSP(n_components=n_components, reg=0.1, log=True, norm_trace=False)
    X_calib_csp = csp.fit_transform(X_calib, y_calib)
    X_test_csp = csp.transform(X_test)
    
    # Extract PSD features
    X_calib_psd = compute_band_power(X_calib)
    X_test_psd = compute_band_power(X_test)
    
    # Combine CSP and PSD features
    X_calib_combined = np.hstack([X_calib_csp, X_calib_psd])
    X_test_combined = np.hstack([X_test_csp, X_test_psd])
    
    # Scale features
    scaler = StandardScaler()
    X_calib_scaled = scaler.fit_transform(X_calib_combined)
    X_test_scaled = scaler.transform(X_test_combined)
    
    if use_ensemble:
        # Create ensemble of classifiers
        clf_lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        clf_svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        
        clf = VotingClassifier(
            estimators=[('lr', clf_lr), ('svm', clf_svm)],
            voting='soft'
        )
    else:
        # Use single SVM classifier
        clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    
    # Train
    clf.fit(X_calib_scaled, y_calib)
    
    # Evaluate
    calib_acc = clf.score(X_calib_scaled, y_calib)
    test_acc = clf.score(X_test_scaled, y_test)
    
    # Package model
    model_dict = {
        'csp': csp,
        'scaler': scaler,
        'classifier': clf,
        'n_components': n_components,
        'n_calib_runs': n_calib_runs,
        'use_ensemble': use_ensemble
    }
    
    return model_dict, calib_acc, test_acc


def train_all_subjects_optimized(n_calib_runs=2, use_ensemble=True):
    """
    Train optimized models for all subjects.
    """
    results = {
        'left_right': {'calib_accs': [], 'test_accs': [], 'subjects': []},
        'hands_feet': {'calib_accs': [], 'test_accs': [], 'subjects': []}
    }
    
    os.makedirs('models/subject_specific_optimized/left_right', exist_ok=True)
    os.makedirs('models/subject_specific_optimized/hands_feet', exist_ok=True)
    
    for subject_id in tqdm(range(1, 110), desc="Training subjects"):
        for task in ['left_right', 'hands_feet']:
            try:
                model_dict, calib_acc, test_acc = train_subject_optimized(
                    subject_id, task, n_calib_runs, use_ensemble
                )
                
                # Save model
                model_path = f'models/subject_specific_optimized/{task}/subject_{subject_id:03d}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model_dict, f)
                
                # Record results
                results[task]['calib_accs'].append(calib_acc)
                results[task]['test_accs'].append(test_acc)
                results[task]['subjects'].append(subject_id)
                
            except Exception as e:
                print(f"\n⚠️  Subject {subject_id}, Task {task}: {e}")
                continue
    
    # Print summary
    print("\n" + "="*80)
    print("🎉 OPTIMIZED TRAINING COMPLETE!")
    print("="*80)
    
    for task in ['left_right', 'hands_feet']:
        task_name = "LEFT vs RIGHT" if task == 'left_right' else "HANDS vs FEET"
        test_accs = np.array(results[task]['test_accs']) * 100
        calib_accs = np.array(results[task]['calib_accs']) * 100
        
        print(f"\n{task_name}:")
        print(f"  Mean Test Accuracy: {test_accs.mean():.1f}% ± {test_accs.std():.1f}%")
        print(f"  Mean Calibration Accuracy: {calib_accs.mean():.1f}% ± {calib_accs.std():.1f}%")
        print(f"  Best Subject: {test_accs.max():.1f}%")
        print(f"  Worst Subject: {test_accs.min():.1f}%")
        print(f"  Subjects > 65%: {(test_accs > 65).sum()} ({(test_accs > 65).sum() / len(test_accs) * 100:.1f}%)")
        print(f"  Subjects > 60%: {(test_accs > 60).sum()} ({(test_accs > 60).sum() / len(test_accs) * 100:.1f}%)")
    
    # Save results
    with open('optimized_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n✅ Results saved to 'optimized_results.pkl'")
    print("✅ Models saved to 'models/subject_specific_optimized/'")
    
    return results


if __name__ == '__main__':
    print("🚀 Starting Optimized Subject-Specific Training")
    print("="*80)
    print("\nImprovements:")
    print("  1️⃣  Using 2 calibration runs (2x more training data)")
    print("  2️⃣  Optimizing CSP components per subject")
    print("  3️⃣  Adding PSD features (mu, beta, gamma bands)")
    print("  4️⃣  Using SVM with RBF kernel")
    print("  5️⃣  Ensemble voting classifier (LR + SVM)")
    print("\n🎯 Target: ≥65% mean test accuracy")
    print("="*80 + "\n")
    
    results = train_all_subjects_optimized(n_calib_runs=2, use_ensemble=True)
