#!/usr/bin/env python3
"""
Simple improved training: Use 2 calibration runs instead of 1.
This was identified as the best simple improvement (+1.5%) from quick comparison.
"""

import os
import numpy as np
import mne
from mne.decoding import CSP
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
from tqdm import tqdm
import warnings

# Suppress MNE warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('ERROR')


def load_subject_data_2runs(subject_id, task='left_right'):
    """Load data for one subject with 2 calibration runs."""
    if task == 'left_right':
        event_ids = {'left': 2, 'right': 3}
        calib_runs = [3, 7]  # 2 real movement runs
        test_runs = [4, 8, 12]  # 3 imagery runs
    else:
        event_ids = {'hands': 2, 'feet': 3}
        calib_runs = [5, 9]  # 2 real movement runs
        test_runs = [6, 10, 14]  # 3 imagery runs
    
    # Load calibration data (2 runs)
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


def train_subject_2runs(subject_id, task='left_right'):
    """Train model for one subject using 2 calibration runs."""
    X_calib, y_calib, X_test, y_test = load_subject_data_2runs(subject_id, task)
    
    # CSP
    csp = CSP(n_components=6, reg=0.1, log=True, norm_trace=False)
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
        'n_calib_runs': 2
    }
    
    return model_dict, calib_acc, test_acc


def train_all_subjects_2runs():
    """Train models for all 109 subjects with 2 calibration runs."""
    results = {
        'left_right': {'calib_accs': [], 'test_accs': [], 'subjects': []},
        'hands_feet': {'calib_accs': [], 'test_accs': [], 'subjects': []}
    }
    
    os.makedirs('models/subject_specific_2runs/left_right', exist_ok=True)
    os.makedirs('models/subject_specific_2runs/hands_feet', exist_ok=True)
    
    print("🚀 Training all 109 subjects with 2 calibration runs")
    print("="*80)
    
    for subject_id in tqdm(range(1, 110), desc="Training subjects"):
        for task in ['left_right', 'hands_feet']:
            try:
                model_dict, calib_acc, test_acc = train_subject_2runs(subject_id, task)
                
                # Save model
                model_path = f'models/subject_specific_2runs/{task}/subject_{subject_id:03d}.pkl'
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
    print("🎉 TRAINING COMPLETE WITH 2 CALIBRATION RUNS!")
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
        print(f"  Subjects ≥ 65%: {(test_accs >= 65).sum()} ({(test_accs >= 65).sum() / len(test_accs) * 100:.1f}%)")
        print(f"  Subjects ≥ 60%: {(test_accs >= 60).sum()} ({(test_accs >= 60).sum() / len(test_accs) * 100:.1f}%)")
        print(f"  Subjects ≥ 50%: {(test_accs >= 50).sum()} ({(test_accs >= 50).sum() / len(test_accs) * 100:.1f}%)")
    
    # Save results
    with open('results_2runs.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n✅ Results saved to 'results_2runs.pkl'")
    print("✅ Models saved to 'models/subject_specific_2runs/'")
    
    return results


if __name__ == '__main__':
    results = train_all_subjects_2runs()
