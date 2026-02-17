#!/usr/bin/env python3
"""
Optimisation avancée CSP: n_components ET régularisation.
Test sur quelques sujets pour voir l'impact.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from mne.decoding import CSP
from processing import Processing

def optimize_csp_advanced(X_calib, y_calib, n_splits=3):
    """
    Optimise à la fois n_components ET reg (régularisation).
    
    Teste toutes les combinaisons:
    - n_components: 4, 6, 8
    - reg: 0.0, 0.1, 0.3
    
    Returns:
        best_n_components, best_reg, best_score
    """
    # Vérifier qu'on a assez d'échantillons
    n_samples_per_class = np.bincount(y_calib)
    if len(n_samples_per_class) < 2 or min(n_samples_per_class) < n_splits:
        return 6, 0.1, 0.0
    
    best_n = 6
    best_reg = 0.1
    best_score = 0
    
    # Tester toutes les combinaisons
    for n_comp in [4, 6, 8]:
        if n_comp > X_calib.shape[1]:
            continue
            
        for reg in [0.0, 0.1, 0.3]:
            scores = []
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            try:
                for train_idx, val_idx in skf.split(X_calib, y_calib):
                    X_train, X_val = X_calib[train_idx], X_calib[val_idx]
                    y_train, y_val = y_calib[train_idx], y_calib[val_idx]
                    
                    # CSP avec reg variable
                    csp = CSP(n_components=n_comp, reg=reg if reg > 0 else None, 
                             log=True, norm_trace=False)
                    X_train_csp = csp.fit_transform(X_train, y_train)
                    X_val_csp = csp.transform(X_val)
                    
                    # Scaler
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_csp)
                    X_val_scaled = scaler.transform(X_val_csp)
                    
                    # LDA
                    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                    clf.fit(X_train_scaled, y_train)
                    scores.append(clf.score(X_val_scaled, y_val))
                
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_n = n_comp
                    best_reg = reg
            except:
                continue
    
    return best_n, best_reg, best_score


def test_on_subject(subject_id, task='left_right'):
    """Test optimisation avancée sur un sujet."""
    
    if task == 'left_right':
        calib_runs = [3, 7, 11]  # 3 runs!
        test_runs = [4, 8, 12]
    else:
        calib_runs = [5, 9, 13]  # 3 runs!
        test_runs = [6, 10, 14]
    
    p = Processing()
    
    # Charger données
    X_calib_all, y_calib_all = [], []
    for run in calib_runs:
        try:
            X, y = p.get_all_data(subject_id, run)
            if len(X) > 0:
                X_calib_all.extend(X)
                y_calib_all.extend(y)
        except:
            continue
    
    X_test_all, y_test_all = [], []
    for run in test_runs:
        try:
            X, y = p.get_all_data(subject_id, run)
            if len(X) > 0:
                X_test_all.extend(X)
                y_test_all.extend(y)
        except:
            continue
    
    if len(X_calib_all) < 15 or len(X_test_all) < 5:
        return None
    
    # Uniformiser
    min_len = min(x.shape[1] for x in X_calib_all)
    X_calib = np.array([x[:, :min_len] for x in X_calib_all])
    y_calib = np.array(y_calib_all)
    
    min_len = min(x.shape[1] for x in X_test_all)
    X_test = np.array([x[:, :min_len] for x in X_test_all])
    y_test = np.array(y_test_all)
    
    if len(set(y_calib)) < 2 or len(set(y_test)) < 2:
        return None
    
    # Encoder
    le = LabelEncoder()
    y_calib_enc = le.fit_transform(y_calib)
    y_test_enc = le.transform(y_test)
    
    # Test 1: Baseline (6 comp, reg=0.1)
    csp_base = CSP(n_components=6, reg=0.1, log=True, norm_trace=False)
    X_calib_csp = csp_base.fit_transform(X_calib, y_calib_enc)
    X_test_csp = csp_base.transform(X_test)
    
    scaler = StandardScaler()
    X_calib_scaled = scaler.fit_transform(X_calib_csp)
    X_test_scaled = scaler.transform(X_test_csp)
    
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(X_calib_scaled, y_calib_enc)
    score_baseline = clf.score(X_test_scaled, y_test_enc)
    
    # Test 2: Optimisation simple (n_components seulement)
    best_n_simple, _, cv_simple = optimize_csp_advanced(X_calib, y_calib_enc, n_splits=3)
    # Forcer reg=0.1 pour comparison
    csp_simple = CSP(n_components=best_n_simple, reg=0.1, log=True, norm_trace=False)
    X_calib_csp = csp_simple.fit_transform(X_calib, y_calib_enc)
    X_test_csp = csp_simple.transform(X_test)
    
    scaler = StandardScaler()
    X_calib_scaled = scaler.fit_transform(X_calib_csp)
    X_test_scaled = scaler.transform(X_test_csp)
    
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(X_calib_scaled, y_calib_enc)
    score_simple = clf.score(X_test_scaled, y_test_enc)
    
    # Test 3: Optimisation avancée (n_components + reg)
    best_n_adv, best_reg_adv, cv_adv = optimize_csp_advanced(X_calib, y_calib_enc, n_splits=3)
    
    csp_adv = CSP(n_components=best_n_adv, 
                  reg=best_reg_adv if best_reg_adv > 0 else None,
                  log=True, norm_trace=False)
    X_calib_csp = csp_adv.fit_transform(X_calib, y_calib_enc)
    X_test_csp = csp_adv.transform(X_test)
    
    scaler = StandardScaler()
    X_calib_scaled = scaler.fit_transform(X_calib_csp)
    X_test_scaled = scaler.transform(X_test_csp)
    
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(X_calib_scaled, y_calib_enc)
    score_advanced = clf.score(X_test_scaled, y_test_enc)
    
    return {
        'subject_id': subject_id,
        'task': task,
        'n_calib': len(X_calib),
        'baseline': score_baseline,
        'simple_opt': score_simple,
        'advanced_opt': score_advanced,
        'best_n_simple': best_n_simple,
        'best_n_adv': best_n_adv,
        'best_reg_adv': best_reg_adv
    }


def main():
    print("="*80)
    print("🔬 TEST: Impact de 3 runs de calibration + Optimisation reg")
    print("="*80)
    print("Comparaison sur 10 sujets:\n")
    print("1. Baseline: 6 comp, reg=0.1")
    print("2. Opti simple: n_comp optimisé, reg=0.1")
    print("3. Opti avancée: n_comp ET reg optimisés\n")
    
    results = []
    for subject_id in range(1, 11):
        print(f"\r🧪 Test sujet {subject_id}/10...", end="", flush=True)
        
        for task in ['left_right', 'hands_feet']:
            result = test_on_subject(subject_id, task)
            if result:
                results.append(result)
    
    print("\n\n" + "="*80)
    print("📊 RÉSULTATS")
    print("="*80 + "\n")
    
    if not results:
        print("❌ Aucun résultat")
        return
    
    # Statistiques
    baseline_scores = [r['baseline'] * 100 for r in results]
    simple_scores = [r['simple_opt'] * 100 for r in results]
    advanced_scores = [r['advanced_opt'] * 100 for r in results]
    
    print("📈 Scores moyens:")
    print(f"  Baseline (6 comp, reg=0.1):        {np.mean(baseline_scores):.2f}% ± {np.std(baseline_scores):.2f}%")
    print(f"  Opti simple (n_comp):              {np.mean(simple_scores):.2f}% ± {np.std(simple_scores):.2f}%")
    print(f"  Opti avancée (n_comp + reg):       {np.mean(advanced_scores):.2f}% ± {np.std(advanced_scores):.2f}%")
    
    # Gains
    gain_simple = np.mean(simple_scores) - np.mean(baseline_scores)
    gain_advanced = np.mean(advanced_scores) - np.mean(baseline_scores)
    
    print(f"\n💡 Gains par rapport au baseline:")
    print(f"  Opti simple:   {gain_simple:+.2f}%")
    print(f"  Opti avancée:  {gain_advanced:+.2f}%")
    
    # Distribution des paramètres
    print(f"\n🔧 Paramètres optimaux sélectionnés:")
    for reg_val in [0.0, 0.1, 0.3]:
        count = sum(1 for r in results if r['best_reg_adv'] == reg_val)
        pct = 100 * count / len(results)
        print(f"  reg={reg_val}: {count}/{len(results)} ({pct:.0f}%)")
    
    print(f"\n🏆 Top 5 gains avec optimisation avancée:")
    sorted_by_gain = sorted(results, 
                           key=lambda r: r['advanced_opt'] - r['baseline'],
                           reverse=True)
    for i, r in enumerate(sorted_by_gain[:5], 1):
        gain = 100 * (r['advanced_opt'] - r['baseline'])
        print(f"  {i}. S{r['subject_id']:03d} {r['task']:12s}: "
              f"Baseline={r['baseline']*100:5.1f}% → Advanced={r['advanced_opt']*100:5.1f}% "
              f"({gain:+.1f}%) [n={r['best_n_adv']}, reg={r['best_reg_adv']}]")
    
    print("\n" + "="*80)
    print("💡 RECOMMANDATION")
    print("="*80)
    
    if gain_advanced > gain_simple + 0.5:
        print("✅ Utiliser l'optimisation avancée (n_comp + reg)")
        print(f"   Gain additionnel: +{gain_advanced - gain_simple:.2f}%")
    elif gain_simple > 0.5:
        print("⚖️  Optimisation simple suffisante")
        print("   Gain similaire avec moins de calculs")
    else:
        print("➖ Peu d'amélioration, mais toujours positif")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
