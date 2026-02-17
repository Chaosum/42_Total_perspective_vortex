#!/usr/bin/env python3
"""
Comparaison rapide LDA vs Logistic Regression pour BCI.
Test sur quelques sujets pour voir l'impact.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mne.decoding import CSP
from processing import Processing

def compare_classifiers_on_subject(subject_id, task='left_right'):
    """Compare LDA vs LogReg sur un sujet."""
    
    # Configuration
    if task == 'left_right':
        calib_runs = [3, 7]
        test_runs = [4, 8, 12]
    else:
        calib_runs = [5, 9]
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
    
    if len(X_calib_all) < 10 or len(X_test_all) < 5:
        return None
    
    # Uniformiser
    min_len_calib = min(x.shape[1] for x in X_calib_all)
    X_calib = np.array([x[:, :min_len_calib] for x in X_calib_all])
    y_calib = np.array(y_calib_all)
    
    min_len_test = min(x.shape[1] for x in X_test_all)
    X_test = np.array([x[:, :min_len_test] for x in X_test_all])
    y_test = np.array(y_test_all)
    
    if len(set(y_calib)) < 2 or len(set(y_test)) < 2:
        return None
    
    # Encoder
    le = LabelEncoder()
    y_calib_enc = le.fit_transform(y_calib)
    y_test_enc = le.transform(y_test)
    
    # CSP + Scaler
    csp = CSP(n_components=6, reg=0.1, log=True, norm_trace=False)
    X_calib_csp = csp.fit_transform(X_calib, y_calib_enc)
    X_test_csp = csp.transform(X_test)
    
    scaler = StandardScaler()
    X_calib_scaled = scaler.fit_transform(X_calib_csp)
    X_test_scaled = scaler.transform(X_test_csp)
    
    # Test 1: Logistic Regression
    clf_logreg = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf_logreg.fit(X_calib_scaled, y_calib_enc)
    score_logreg = clf_logreg.score(X_test_scaled, y_test_enc)
    
    # Test 2: LDA
    clf_lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_lda.fit(X_calib_scaled, y_calib_enc)
    score_lda = clf_lda.score(X_test_scaled, y_test_enc)
    
    # Test 3: LDA sans scaling (LDA est robuste au scaling)
    clf_lda_noscale = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf_lda_noscale.fit(X_calib_csp, y_calib_enc)
    score_lda_noscale = clf_lda_noscale.score(X_test_csp, y_test_enc)
    
    return {
        'subject_id': subject_id,
        'task': task,
        'logreg': score_logreg,
        'lda': score_lda,
        'lda_noscale': score_lda_noscale,
        'n_calib': len(X_calib),
        'n_test': len(X_test)
    }


def main():
    print("="*80)
    print("🔬 COMPARAISON: LDA vs Logistic Regression pour BCI")
    print("="*80)
    print("Test sur 20 sujets × 2 tâches = 40 comparaisons\n")
    
    results = []
    test_subjects = list(range(1, 21))  # 20 premiers sujets
    
    for subject_id in test_subjects:
        print(f"\r🧪 Test sujet {subject_id}/20...", end="", flush=True)
        
        for task in ['left_right', 'hands_feet']:
            result = compare_classifiers_on_subject(subject_id, task)
            if result:
                results.append(result)
    
    print("\n\n" + "="*80)
    print("📊 RÉSULTATS")
    print("="*80 + "\n")
    
    if not results:
        print("❌ Aucun résultat")
        return
    
    # Statistiques globales
    logreg_scores = [r['logreg'] * 100 for r in results]
    lda_scores = [r['lda'] * 100 for r in results]
    lda_noscale_scores = [r['lda_noscale'] * 100 for r in results]
    
    print("📈 Scores moyens:")
    print(f"  Logistic Regression:    {np.mean(logreg_scores):.2f}% ± {np.std(logreg_scores):.2f}%")
    print(f"  LDA (avec scaling):     {np.mean(lda_scores):.2f}% ± {np.std(lda_scores):.2f}%")
    print(f"  LDA (sans scaling):     {np.mean(lda_noscale_scores):.2f}% ± {np.std(lda_noscale_scores):.2f}%")
    
    # Comparaisons pairées
    print("\n🔍 Comparaisons pairées:")
    
    lda_wins = sum(1 for r in results if r['lda'] > r['logreg'])
    logreg_wins = sum(1 for r in results if r['logreg'] > r['lda'])
    ties = len(results) - lda_wins - logreg_wins
    
    print(f"  LDA meilleur:    {lda_wins}/{len(results)} cas ({100*lda_wins/len(results):.1f}%)")
    print(f"  LogReg meilleur: {logreg_wins}/{len(results)} cas ({100*logreg_wins/len(results):.1f}%)")
    print(f"  Égalité:         {ties}/{len(results)} cas")
    
    # Gains moyens
    gains_lda = [100 * (r['lda'] - r['logreg']) for r in results]
    print(f"\n💡 Gain moyen LDA vs LogReg: {np.mean(gains_lda):+.2f}% (médiane: {np.median(gains_lda):+.2f}%)")
    
    # Top cas où LDA aide le plus
    print("\n🏆 Top 5 gains avec LDA:")
    sorted_by_gain = sorted(results, key=lambda r: r['lda'] - r['logreg'], reverse=True)
    for i, r in enumerate(sorted_by_gain[:5], 1):
        gain = 100 * (r['lda'] - r['logreg'])
        print(f"  {i}. S{r['subject_id']:03d} {r['task']:12s}: LDA={r['lda']*100:5.1f}% vs LogReg={r['logreg']*100:5.1f}% ({gain:+.1f}%)")
    
    # Cas où LogReg est meilleur
    print("\n⚠️  Top 5 cas où LogReg est meilleur:")
    sorted_by_loss = sorted(results, key=lambda r: r['lda'] - r['logreg'])
    for i, r in enumerate(sorted_by_loss[:5], 1):
        gain = 100 * (r['lda'] - r['logreg'])
        print(f"  {i}. S{r['subject_id']:03d} {r['task']:12s}: LDA={r['lda']*100:5.1f}% vs LogReg={r['logreg']*100:5.1f}% ({gain:+.1f}%)")
    
    # Test statistique
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(lda_scores, logreg_scores)
    
    print(f"\n📊 Test t apparié:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        if np.mean(lda_scores) > np.mean(logreg_scores):
            print(f"  ✅ LDA est statistiquement MEILLEUR que LogReg (p < 0.05)")
        else:
            print(f"  ⚠️  LogReg est statistiquement meilleur que LDA (p < 0.05)")
    else:
        print(f"  ➖ Pas de différence statistiquement significative (p > 0.05)")
    
    print("\n" + "="*80)
    print("💡 RECOMMANDATION")
    print("="*80)
    
    mean_gain = np.mean(gains_lda)
    if mean_gain > 1.0 and p_value < 0.05:
        print("✅ Utiliser LDA: Gain significatif et statistiquement valide")
        print("   → CSP + LDA est le pipeline standard en BCI")
    elif mean_gain > 0.5:
        print("⚖️  LDA légèrement meilleur: Gain modeste mais positif")
        print("   → Considérer LDA pour sa simplicité (pas d'hyperparamètre C)")
    else:
        print("➖ Performances équivalentes: Garder LogReg ou essayer LDA")
    
    print("\n" + "="*80 + "\n")
    
    # Sauvegarder résultats
    import pickle
    with open('lda_vs_logreg_comparison.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("💾 Résultats sauvegardés: lda_vs_logreg_comparison.pkl\n")


if __name__ == "__main__":
    main()
