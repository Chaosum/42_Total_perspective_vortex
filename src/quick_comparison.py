#!/usr/bin/env python3
"""
Quick comparison of key improvements vs baseline.
Tests on 10 subjects to quickly evaluate which improvements help most.
"""

import numpy as np
import mne
from mne.decoding import CSP
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from scipy.signal import welch
import pickle
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('ERROR')


def compute_band_power(epochs_data, sfreq=160):
    """Compute band power features."""
    bands = {'mu': (8, 13), 'beta': (13, 30), 'low_gamma': (30, 45)}
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


def test_approach(name, subjects, tasks, n_calib_runs=1, use_psd=False, 
                  classifier_type='lr', use_ensemble=False):
    """Test one approach on given subjects."""
    results = []
    
    for subject_id in subjects:
        for task in tasks:
            try:
                X_calib, y_calib, X_test, y_test = load_subject_data(
                    subject_id, task, n_calib_runs
                )
                
                # CSP
                csp = CSP(n_components=6, reg=0.1, log=True, norm_trace=False)
                X_calib_csp = csp.fit_transform(X_calib, y_calib)
                X_test_csp = csp.transform(X_test)
                
                # Add PSD if requested
                if use_psd:
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
                if use_ensemble:
                    clf_lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
                    clf_svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
                    clf = VotingClassifier(
                        estimators=[('lr', clf_lr), ('svm', clf_svm)],
                        voting='soft'
                    )
                elif classifier_type == 'lr':
                    clf = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
                elif classifier_type == 'svm':
                    clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
                
                clf.fit(X_calib_scaled, y_calib)
                acc = clf.score(X_test_scaled, y_test)
                results.append(acc)
                
            except Exception as e:
                continue
    
    mean_acc = np.mean(results) * 100
    std_acc = np.std(results) * 100
    
    return mean_acc, std_acc, len(results)


def quick_comparison():
    """
    Quick comparison of key improvements.
    """
    print("⚡ QUICK COMPARISON OF IMPROVEMENTS")
    print("="*80)
    
    # Test on diverse subjects (mix of good and bad performers based on previous results)
    test_subjects = [1, 2, 5, 10, 15, 20, 30, 50, 75, 100]
    tasks = ['left_right', 'hands_feet']
    
    print(f"Testing on {len(test_subjects)} subjects × {len(tasks)} tasks = {len(test_subjects) * len(tasks)} tests")
    print(f"Subjects: {test_subjects}")
    print("="*80 + "\n")
    
    approaches = [
        {
            'name': '1. BASELINE (1 run, CSP only, LogReg)',
            'n_calib_runs': 1,
            'use_psd': False,
            'classifier_type': 'lr',
            'use_ensemble': False
        },
        {
            'name': '2. Double calibration data (2 runs)',
            'n_calib_runs': 2,
            'use_psd': False,
            'classifier_type': 'lr',
            'use_ensemble': False
        },
        {
            'name': '3. Add PSD features',
            'n_calib_runs': 1,
            'use_psd': True,
            'classifier_type': 'lr',
            'use_ensemble': False
        },
        {
            'name': '4. Use SVM instead of LogReg',
            'n_calib_runs': 1,
            'use_psd': False,
            'classifier_type': 'svm',
            'use_ensemble': False
        },
        {
            'name': '5. Combine: 2 runs + PSD',
            'n_calib_runs': 2,
            'use_psd': True,
            'classifier_type': 'lr',
            'use_ensemble': False
        },
        {
            'name': '6. Combine: 2 runs + SVM',
            'n_calib_runs': 2,
            'use_psd': False,
            'classifier_type': 'svm',
            'use_ensemble': False
        },
        {
            'name': '7. FULL: 2 runs + PSD + SVM',
            'n_calib_runs': 2,
            'use_psd': True,
            'classifier_type': 'svm',
            'use_ensemble': False
        },
        {
            'name': '8. FULL + Ensemble (LR+SVM)',
            'n_calib_runs': 2,
            'use_psd': True,
            'classifier_type': 'lr',
            'use_ensemble': True
        }
    ]
    
    results = []
    
    for i, approach in enumerate(approaches):
        print(f"\n[{i+1}/{len(approaches)}] {approach['name']}")
        print("-" * 80)
        
        mean_acc, std_acc, n_tests = test_approach(
            approach['name'],
            test_subjects,
            tasks,
            approach['n_calib_runs'],
            approach['use_psd'],
            approach['classifier_type'],
            approach['use_ensemble']
        )
        
        improvement = mean_acc - (results[0]['mean_acc'] if results else mean_acc)
        
        result = {
            'name': approach['name'],
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'n_tests': n_tests,
            'improvement': improvement
        }
        results.append(result)
        
        print(f"Mean Accuracy: {mean_acc:.1f}% ± {std_acc:.1f}%")
        if i > 0:
            print(f"Improvement over baseline: {improvement:+.1f}%")
        print(f"Tests completed: {n_tests}/{len(test_subjects) * len(tasks)}")
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY - RANKED BY PERFORMANCE")
    print("="*80 + "\n")
    
    results_sorted = sorted(results, key=lambda x: x['mean_acc'], reverse=True)
    
    print(f"{'Rank':<6} {'Accuracy':<15} {'Improvement':<15} {'Approach'}")
    print("-" * 80)
    
    for i, result in enumerate(results_sorted):
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        acc_str = f"{result['mean_acc']:.1f}%±{result['std_acc']:.1f}%"
        imp_str = f"{result['improvement']:+.1f}%" if result['improvement'] != 0 else "baseline"
        
        print(f"{rank_emoji:<6} {acc_str:<15} {imp_str:<15} {result['name']}")
    
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)
    
    best = results_sorted[0]
    print(f"\n✨ Best approach: {best['name']}")
    print(f"   Accuracy: {best['mean_acc']:.1f}% ± {best['std_acc']:.1f}%")
    print(f"   Improvement: {best['improvement']:+.1f}% over baseline")
    
    if best['mean_acc'] >= 65:
        print(f"\n✅ TARGET ACHIEVED! {best['mean_acc']:.1f}% ≥ 65%")
        print("   This approach should be used for full training on all 109 subjects.")
    else:
        print(f"\n⚠️  Target not yet reached: {best['mean_acc']:.1f}% < 65%")
        print(f"   Additional {65 - best['mean_acc']:.1f}% improvement needed.")
        print("\n   Further improvements to try:")
        print("   • Optimize CSP components per subject")
        print("   • Try different frequency bands (wider or narrower)")
        print("   • Use more sophisticated features (wavelet transform)")
        print("   • Try deep learning (EEGNet, ShallowConvNet)")
    
    # Save results
    with open('quick_comparison_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n✅ Results saved to 'quick_comparison_results.pkl'")
    
    return results


if __name__ == '__main__':
    results = quick_comparison()
