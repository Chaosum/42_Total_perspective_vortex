#!/usr/bin/env python3
"""
Subject-specific training with CSP optimization.
For each subject, finds the optimal number of CSP components using cross-validation.
This should provide +1-2% improvement by adapting to each subject's brain patterns.
"""

import os
import numpy as np
import mne
from mne.decoding import CSP
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import pickle
from tqdm import tqdm
import warnings

# Suppress MNE warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('ERROR')


def load_subject_data(subject_id, task='left_right', n_calib_runs=1):
    """Load data for one subject."""
    if task == 'left_right':
        event_ids = {'left': 2, 'right': 3}
        calib_runs = [3, 7][:n_calib_runs]
        test_runs = [4, 8, 12]
    else:
        event_ids = {'hands': 2, 'feet': 3}
        calib_runs = [5, 9][:n_calib_runs]
        test_runs = [6, 10, 14]
    
    # Load calibration data
    X_calib_list, y_calib_list = [], []
    for run in calib_runs:
        raw = mne.io.read_raw_edf(
            f'src/data/MNE-eegbci-data/files/eegmmidb/1.0.0/S{subject_id:03d}/S{subject_id:03d}R{run:02d}.edf',
            preload=True, verbose=False
        )
        raw.pick('eeg')
        raw.filter(7., 30., fir_design='firwin', verbose=False)
        events, _ = mne.events_from_annotations(raw, verbose=False)
        epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=0, tmax=4,
                           baseline=None, preload=True, verbose=False)
        X_calib_list.append(epochs.get_data())
        y_calib_list.append(epochs.events[:, -1])
    
    X_calib = np.concatenate(X_calib_list, axis=0)
    y_calib = np.concatenate(y_calib_list, axis=0)
    
    # Load test data
    X_test_list, y_test_list = [], []
    for run in test_runs:
        raw = mne.io.read_raw_edf(
            f'src/data/MNE-eegbci-data/files/eegmmidb/1.0.0/S{subject_id:03d}/S{subject_id:03d}R{run:02d}.edf',
            preload=True, verbose=False
        )
        raw.pick('eeg')
        raw.filter(7., 30., fir_design='firwin', verbose=False)
        events, _ = mne.events_from_annotations(raw, verbose=False)
        epochs = mne.Epochs(raw, events, event_id=event_ids, tmin=0, tmax=4,
                           baseline=None, preload=True, verbose=False)
        X_test_list.append(epochs.get_data())
        y_test_list.append(epochs.events[:, -1])
    
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)
    
    return X_calib, y_calib, X_test, y_test


def optimize_csp_components(X_calib, y_calib, n_splits=3):
    """
    Find optimal number of CSP components using cross-validation.
    
    Returns:
        best_n_components: optimal number (4, 6, or 8)
        best_score: CV score with optimal components
    """
    # Check if we have enough samples for CV
    n_samples_per_class = np.bincount(y_calib)
    if len(n_samples_per_class) < 2 or min(n_samples_per_class) < n_splits:
        # Not enough samples for CV, use default
        return 6, 0.0
    
    best_n = 6
    best_score = 0
    
    # Try different numbers of components
    for n_comp in [4, 6, 8]:
        if n_comp > X_calib.shape[1]:  # More components than channels
            continue
        
        scores = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        try:
            for train_idx, val_idx in skf.split(X_calib, y_calib):
                X_train, X_val = X_calib[train_idx], X_calib[val_idx]
                y_train, y_val = y_calib[train_idx], y_calib[val_idx]
                
                # CSP
                csp = CSP(n_components=n_comp, reg=0.1, log=True, norm_trace=False)
                X_train_csp = csp.fit_transform(X_train, y_train)
                X_val_csp = csp.transform(X_val)
                
                # Scale
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_csp)
                X_val_scaled = scaler.transform(X_val_csp)
                
                # Classifier
                clf = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
                clf.fit(X_train_scaled, y_train)
                scores.append(clf.score(X_val_scaled, y_val))
            
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_n = n_comp
        except:
            continue
    
    return best_n, best_score


def train_subject_optimized(subject_id, task='left_right'):
    """Train model for one subject with optimized CSP components."""
    X_calib, y_calib, X_test, y_test = load_subject_data(subject_id, task, n_calib_runs=1)
    
    # Optimize CSP components
    n_components, cv_score = optimize_csp_components(X_calib, y_calib)
    
    # Train with optimal components
    csp = CSP(n_components=n_components, reg=0.1, log=True, norm_trace=False)
    X_calib_csp = csp.fit_transform(X_calib, y_calib)
    X_test_csp = csp.transform(X_test)
    
    # Scale
    scaler = StandardScaler()
    X_calib_scaled = scaler.fit_transform(X_calib_csp)
    X_test_scaled = scaler.transform(X_test_csp)
    
    # Classifier
    clf = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
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
        'cv_score': cv_score
    }
    
    return model_dict, calib_acc, test_acc, n_components


def train_all_subjects_optimized():
    """Train models for all 109 subjects with optimized CSP."""
    results = {
        'left_right': {'calib_accs': [], 'test_accs': [], 'subjects': [], 'n_components': []},
        'hands_feet': {'calib_accs': [], 'test_accs': [], 'subjects': [], 'n_components': []}
    }
    
    os.makedirs('models/subject_specific_optimized/left_right', exist_ok=True)
    os.makedirs('models/subject_specific_optimized/hands_feet', exist_ok=True)
    
    print("🚀 Training all 109 subjects with CSP optimization")
    print("="*80)
    print("Strategy: Find optimal n_components (4, 6, or 8) per subject via CV")
    print("="*80)
    
    for subject_id in tqdm(range(1, 110), desc="Training subjects"):
        for task in ['left_right', 'hands_feet']:
            try:
                model_dict, calib_acc, test_acc, n_comp = train_subject_optimized(subject_id, task)
                
                # Save model
                model_path = f'models/subject_specific_optimized/{task}/subject_{subject_id:03d}.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model_dict, f)
                
                # Record results
                results[task]['calib_accs'].append(calib_acc)
                results[task]['test_accs'].append(test_acc)
                results[task]['subjects'].append(subject_id)
                results[task]['n_components'].append(n_comp)
                
            except Exception as e:
                print(f"\n⚠️  Subject {subject_id}, Task {task}: {e}")
                continue
    
    # Print summary
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE WITH CSP OPTIMIZATION!")
    print("="*80)
    
    for task in ['left_right', 'hands_feet']:
        task_name = "LEFT vs RIGHT" if task == 'left_right' else "HANDS vs FEET"
        test_accs = np.array(results[task]['test_accs']) * 100
        calib_accs = np.array(results[task]['calib_accs']) * 100
        n_comps = np.array(results[task]['n_components'])
        
        print(f"\n{task_name}:")
        print(f"  Mean Test Accuracy: {test_accs.mean():.1f}% ± {test_accs.std():.1f}%")
        print(f"  Mean Calibration Accuracy: {calib_accs.mean():.1f}% ± {calib_accs.std():.1f}%")
        print(f"  Best Subject: {test_accs.max():.1f}%")
        print(f"  Worst Subject: {test_accs.min():.1f}%")
        print(f"  Subjects ≥ 65%: {(test_accs >= 65).sum()} ({(test_accs >= 65).sum() / len(test_accs) * 100:.1f}%)")
        print(f"  Subjects ≥ 60%: {(test_accs >= 60).sum()} ({(test_accs >= 60).sum() / len(test_accs) * 100:.1f}%)")
        print(f"  Subjects ≥ 50%: {(test_accs >= 50).sum()} ({(test_accs >= 50).sum() / len(test_accs) * 100:.1f}%)")
        
        print(f"\n  CSP Components Distribution:")
        for nc in [4, 6, 8]:
            count = (n_comps == nc).sum()
            pct = 100 * count / len(n_comps)
            print(f"    {nc} components: {count} subjects ({pct:.1f}%)")
    
    # Save results
    with open('results_optimized.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n✅ Results saved to 'results_optimized.pkl'")
    print("✅ Models saved to 'models/subject_specific_optimized/'")
    
    return results


if __name__ == '__main__':
    results = train_all_subjects_optimized()
