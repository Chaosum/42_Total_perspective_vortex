#!/usr/bin/env python3
"""
Test RAPIDE: Comparaison directe des 3 stratégies sur 20 sujets.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from mne.decoding import CSP
from processing import Processing

def optimize_csp(X, y):
    """Optimisation rapide CSP."""
    n_samples_per_class = np.bincount(y)
    if len(n_samples_per_class) < 2 or min(n_samples_per_class) < 3:
        return 6
    
    best_n = 6
    best_score = 0
    
    for n_comp in [4, 6, 8]:
        if n_comp > X.shape[1]:
            continue
        
        try:
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                csp = CSP(n_components=n_comp, reg=0.1, log=True, norm_trace=False)
                X_train_csp = csp.fit_transform(X_train, y_train)
                X_val_csp = csp.transform(X_val)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_csp)
                X_val_scaled = scaler.transform(X_val_csp)
                
                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                clf.fit(X_train_scaled, y_train)
                scores.append(clf.score(X_val_scaled, y_val))
            
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_n = n_comp
        except:
            continue
    
    return best_n


def test_strategy_1_baseline(subject_id, task='left_right'):
    """Stratégie 1: Baseline actuelle (3 runs calib, 3 runs test)."""
    if task == 'left_right':
        calib_runs, test_runs = [3, 7, 11], [4, 8, 12]
    else:
        calib_runs, test_runs = [5, 9, 13], [6, 10, 14]
    
    p = Processing()
    
    X_calib_all, y_calib_all = [], []
    for r in calib_runs:
        try:
            X, y = p.get_all_data(subject_id, r)
            if len(X) > 0:
                X_calib_all.extend(X)
                y_calib_all.extend(y)
        except:
            pass
    
    X_test_all, y_test_all = [], []
    for r in test_runs:
        try:
            X, y = p.get_all_data(subject_id, r)
            if len(X) > 0:
                X_test_all.extend(X)
                y_test_all.extend(y)
        except:
            pass
    
    if len(X_calib_all) < 20 or len(X_test_all) < 5:
        return None
    
    min_len = min(x.shape[1] for x in X_calib_all)
    X_calib = np.array([x[:, :min_len] for x in X_calib_all])
    y_calib = np.array(y_calib_all)
    
    min_len = min(x.shape[1] for x in X_test_all)
    X_test = np.array([x[:, :min_len] for x in X_test_all])
    y_test = np.array(y_test_all)
    
    if len(set(y_calib)) < 2 or len(set(y_test)) < 2:
        return None
    
    le = LabelEncoder()
    y_calib_enc = le.fit_transform(y_calib)
    y_test_enc = le.transform(y_test)
    
    optimal_n = optimize_csp(X_calib, y_calib_enc)
    
    csp = CSP(n_components=optimal_n, reg=0.1, log=True, norm_trace=False)
    X_calib_csp = csp.fit_transform(X_calib, y_calib_enc)
    X_test_csp = csp.transform(X_test)
    
    scaler = StandardScaler()
    X_calib_scaled = scaler.fit_transform(X_calib_csp)
    X_test_scaled = scaler.transform(X_test_csp)
    
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(X_calib_scaled, y_calib_enc)
    
    return clf.score(X_test_scaled, y_test_enc)


def test_strategy_2_loro(subject_id, task='left_right'):
    """Stratégie 2: Leave-One-Run-Out (max données)."""
    if task == 'left_right':
        all_runs = [3, 4, 7, 8, 11, 12]
        test_runs = [4, 8, 12]
    else:
        all_runs = [5, 6, 9, 10, 13, 14]
        test_runs = [6, 10, 14]
    
    p = Processing()
    scores = []
    
    for test_run in test_runs:
        train_runs = [r for r in all_runs if r != test_run]
        
        X_train_all, y_train_all = [], []
        for r in train_runs:
            try:
                X, y = p.get_all_data(subject_id, r)
                if len(X) > 0:
                    X_train_all.extend(X)
                    y_train_all.extend(y)
            except:
                pass
        
        X_test_all, y_test_all = [], []
        try:
            X, y = p.get_all_data(subject_id, test_run)
            if len(X) > 0:
                X_test_all.extend(X)
                y_test_all.extend(y)
        except:
            pass
        
        if len(X_train_all) < 20 or len(X_test_all) < 5:
            continue
        
        min_len = min(x.shape[1] for x in X_train_all)
        X_train = np.array([x[:, :min_len] for x in X_train_all])
        y_train = np.array(y_train_all)
        
        min_len = min(x.shape[1] for x in X_test_all)
        X_test = np.array([x[:, :min_len] for x in X_test_all])
        y_test = np.array(y_test_all)
        
        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            continue
        
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        
        optimal_n = optimize_csp(X_train, y_train_enc)
        
        csp = CSP(n_components=optimal_n, reg=0.1, log=True, norm_trace=False)
        X_train_csp = csp.fit_transform(X_train, y_train_enc)
        X_test_csp = csp.transform(X_test)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_csp)
        X_test_scaled = scaler.transform(X_test_csp)
        
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf.fit(X_train_scaled, y_train_enc)
        
        scores.append(clf.score(X_test_scaled, y_test_enc))
    
    return np.mean(scores) if scores else None


def main():
    print("="*80)
    print("🔬 TEST RAPIDE: Comparaison des Stratégies")
    print("="*80)
    print("\n1. BASELINE: 3 runs calib [3,7,11] → 3 runs test [4,8,12]")
    print("2. LORO: Pour chaque run test, train sur tous les autres\n")
    print("Test sur 20 sujets pour LEFT/RIGHT...\n")
    
    results = []
    
    for subject_id in range(1, 21):
        print(f"\r🧪 Sujet {subject_id}/20...", end="", flush=True)
        
        score_baseline = test_strategy_1_baseline(subject_id, 'left_right')
        score_loro = test_strategy_2_loro(subject_id, 'left_right')
        
        if score_baseline is not None and score_loro is not None:
            results.append({
                'subject_id': subject_id,
                'baseline': score_baseline,
                'loro': score_loro
            })
    
    print("\n\n" + "="*80)
    print("📊 RÉSULTATS")
    print("="*80 + "\n")
    
    if results:
        baseline_scores = [r['baseline'] * 100 for r in results]
        loro_scores = [r['loro'] * 100 for r in results]
        
        print(f"Stratégie 1 (Baseline):      {np.mean(baseline_scores):.2f}% ± {np.std(baseline_scores):.2f}%")
        print(f"Stratégie 2 (LORO):          {np.mean(loro_scores):.2f}% ± {np.std(loro_scores):.2f}%")
        
        gain = np.mean(loro_scores) - np.mean(baseline_scores)
        print(f"\n💡 Gain LORO vs Baseline:    {gain:+.2f}%")
        
        # Combien atteignent 65%?
        n_above_65_baseline = sum(1 for s in baseline_scores if s >= 65)
        n_above_65_loro = sum(1 for s in loro_scores if s >= 65)
        
        print(f"\n🎯 Sujets ≥ 65%:")
        print(f"   Baseline: {n_above_65_baseline}/{len(results)} ({100*n_above_65_baseline/len(results):.0f}%)")
        print(f"   LORO:     {n_above_65_loro}/{len(results)} ({100*n_above_65_loro/len(results):.0f}%)")
        
        # Détail par sujet
        print(f"\n📋 Détail par sujet:")
        for r in results:
            diff = (r['loro'] - r['baseline']) * 100
            symbol = "🔥" if r['loro'] >= 0.65 else "📊" if r['loro'] >= 0.60 else "⚠️"
            print(f"   S{r['subject_id']:03d}: Baseline={r['baseline']*100:5.1f}% → LORO={r['loro']*100:5.1f}% ({diff:+.1f}%) {symbol}")
        
        print("\n" + "="*80)
        print("💡 CONCLUSION")
        print("="*80)
        
        if np.mean(loro_scores) >= 65:
            print(f"\n🎉🎉🎉 OBJECTIF ATTEINT avec LORO! {np.mean(loro_scores):.2f}% ≥ 65% 🎉🎉🎉")
        elif gain > 3:
            print(f"\n✅ LORO améliore significativement (+{gain:.2f}%)")
            print(f"   Projection 109 sujets: ~{np.mean(loro_scores):.1f}%")
            if np.mean(loro_scores) > 63:
                print(f"   → TRÈS PROCHE de 65%!")
        else:
            print(f"\n📊 Amélioration modeste: +{gain:.2f}%")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
