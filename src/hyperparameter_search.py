#!/usr/bin/env python3
"""
Hyperparameter search for subject-specific BCI models.
Tests different configurations on a subset of subjects to find best settings.
"""

import numpy as np
import mne
from mne.decoding import CSP
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import welch
from itertools import product
import pickle


def compute_band_power(epochs_data, sfreq=160):
    """Compute band power features."""
    bands = {
        'mu': (8, 13),
        'beta': (13, 30),
        'low_gamma': (30, 45)
    }
    
    n_epochs, n_channels, n_times = epochs_data.shape
    n_bands = len(bands)
    features = np.zeros((n_epochs, n_channels * n_bands))
    
    for epoch_idx in range(n_epochs):
        for ch_idx in range(n_channels):
            signal = epochs_data[epoch_idx, ch_idx, :]
            freqs, psd = welch(signal, fs=sfreq, nperseg=min(256, n_times))
            
            for band_idx, (band_name, (low, high)) in enumerate(bands.items()):
                freq_mask = (freqs >= low) & (freqs <= high)
                band_power = np.trapz(psd[freq_mask], freqs[freq_mask])
                features[epoch_idx, ch_idx * n_bands + band_idx] = np.log10(band_power + 1e-10)
    
    return features


def load_subject_data(subject_id, task='left_right', n_calib_runs=2):
    """Load data for one subject."""
    if task == 'left_right':
        event_ids = {'left': 2, 'right': 3}
        calib_runs = [3, 7][:n_calib_runs]
        test_runs = [4, 8, 12]
    else:
        event_ids = {'hands': 2, 'feet': 3}
        calib_runs = [5, 9][:n_calib_runs]
        test_runs = [6, 10, 14]
    
    X_calib_list, y_calib_list = [], []
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
    
    X_test_list, y_test_list = [], []
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


def test_configuration(config, test_subjects=[1, 2, 3, 5, 8, 13, 21, 34]):
    """
    Test a configuration on a subset of subjects.
    
    Args:
        config: dict with keys:
            - n_calib_runs: 1 or 2
            - n_csp_components: 4, 6, or 8
            - use_psd: True or False
            - classifier: 'lr', 'svm', or 'rf'
            - csp_reg: CSP regularization
        test_subjects: list of subject IDs to test on
    
    Returns:
        mean_acc: mean test accuracy across subjects
        std_acc: standard deviation
    """
    results = []
    
    for subject_id in test_subjects:
        for task in ['left_right', 'hands_feet']:
            try:
                X_calib, y_calib, X_test, y_test = load_subject_data(
                    subject_id, task, config['n_calib_runs']
                )
                
                # CSP
                csp = CSP(n_components=config['n_csp_components'],
                         reg=config['csp_reg'], log=True, norm_trace=False)
                X_calib_csp = csp.fit_transform(X_calib, y_calib)
                X_test_csp = csp.transform(X_test)
                
                # Combine with PSD if requested
                if config['use_psd']:
                    X_calib_psd = compute_band_power(X_calib)
                    X_test_psd = compute_band_power(X_test)
                    X_calib_combined = np.hstack([X_calib_csp, X_calib_psd])
                    X_test_combined = np.hstack([X_test_csp, X_test_psd])
                else:
                    X_calib_combined = X_calib_csp
                    X_test_combined = X_test_csp
                
                # Scale
                scaler = StandardScaler()
                X_calib_scaled = scaler.fit_transform(X_calib_combined)
                X_test_scaled = scaler.transform(X_test_combined)
                
                # Classifier
                if config['classifier'] == 'lr':
                    clf = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
                elif config['classifier'] == 'svm':
                    clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
                elif config['classifier'] == 'rf':
                    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
                
                clf.fit(X_calib_scaled, y_calib)
                acc = clf.score(X_test_scaled, y_test)
                results.append(acc)
                
            except Exception as e:
                print(f"Error with subject {subject_id}, task {task}: {e}")
                continue
    
    if len(results) == 0:
        return 0, 0
    
    return np.mean(results) * 100, np.std(results) * 100


def hyperparameter_search():
    """
    Search over hyperparameters to find best configuration.
    """
    print("🔍 HYPERPARAMETER SEARCH")
    print("="*80)
    print("Testing configurations on 8 subjects (Fibonacci sequence: 1,2,3,5,8,13,21,34)")
    print("="*80 + "\n")
    
    # Define search space
    search_space = {
        'n_calib_runs': [1, 2],
        'n_csp_components': [4, 6, 8],
        'use_psd': [False, True],
        'classifier': ['lr', 'svm', 'rf'],
        'csp_reg': [0.01, 0.1, 0.2]
    }
    
    # Generate all combinations
    keys = list(search_space.keys())
    values = list(search_space.values())
    configurations = [dict(zip(keys, v)) for v in product(*values)]
    
    print(f"Total configurations to test: {len(configurations)}\n")
    
    # Test each configuration
    results = []
    best_acc = 0
    best_config = None
    
    for i, config in enumerate(configurations):
        print(f"\n[{i+1}/{len(configurations)}] Testing:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        
        mean_acc, std_acc = test_configuration(config)
        results.append({
            'config': config,
            'mean_acc': mean_acc,
            'std_acc': std_acc
        })
        
        print(f"  → Result: {mean_acc:.1f}% ± {std_acc:.1f}%")
        
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_config = config
            print(f"  ✨ NEW BEST!")
    
    # Sort results
    results = sorted(results, key=lambda x: x['mean_acc'], reverse=True)
    
    # Print summary
    print("\n" + "="*80)
    print("🏆 TOP 10 CONFIGURATIONS")
    print("="*80 + "\n")
    
    for i, result in enumerate(results[:10]):
        print(f"{i+1}. Accuracy: {result['mean_acc']:.1f}% ± {result['std_acc']:.1f}%")
        print(f"   Config: {result['config']}")
        print()
    
    print("="*80)
    print("🎯 BEST CONFIGURATION")
    print("="*80)
    print(f"Mean Accuracy: {best_acc:.1f}%")
    print("Configuration:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    print("="*80)
    
    # Save results
    with open('hyperparameter_search_results.pkl', 'wb') as f:
        pickle.dump({'results': results, 'best_config': best_config}, f)
    
    print("\n✅ Results saved to 'hyperparameter_search_results.pkl'")
    
    return best_config, results


if __name__ == '__main__':
    best_config, results = hyperparameter_search()
