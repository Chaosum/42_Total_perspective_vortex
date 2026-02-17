#!/usr/bin/env python3
"""
Stratégie Leave-One-Run-Out: Un modèle par run de test.
Pour chaque run de test, on entraîne sur TOUS les autres runs du sujet.
Cette approche maximise les données d'entraînement.
"""

import sys
import numpy as np
import joblib
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from mne.decoding import CSP

# Add src to path if needed
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from processing import Processing

def optimize_csp_components(X_calib, y_calib, n_splits=3):
    """Optimise n_components CSP via CV."""
    from sklearn.model_selection import StratifiedKFold
    
    n_samples_per_class = np.bincount(y_calib)
    if len(n_samples_per_class) < 2 or min(n_samples_per_class) < n_splits:
        return 6, 0.0
    
    best_n = 6
    best_score = 0
    
    for n_comp in [4, 6, 8]:
        if n_comp > X_calib.shape[1]:
            continue
        
        scores = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        try:
            for train_idx, val_idx in skf.split(X_calib, y_calib):
                X_train, X_val = X_calib[train_idx], X_calib[val_idx]
                y_train, y_val = y_calib[train_idx], y_calib[val_idx]
                
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
    
    return best_n, best_score


def train_leave_one_run_out(
    subject_ids=range(1, 110),
    task_name="left_right"
):
    """
    Leave-One-Run-Out: Pour chaque run de test, entraîner sur tous les autres.
    
    Exemple pour left_right:
    - Runs disponibles: 3, 4, 7, 8, 11, 12
    - Pour tester run 4:  entraîner sur [3, 7, 8, 11, 12]
    - Pour tester run 8:  entraîner sur [3, 4, 7, 11, 12]
    - Pour tester run 12: entraîner sur [3, 4, 7, 8, 11]
    
    Cela donne 3 modèles par sujet (un par run de test).
    """
    
    # Configuration selon la tâche
    if task_name == "left_right":
        all_runs = [3, 4, 7, 8, 11, 12]  # Tous les runs left/right
        test_runs = [4, 8, 12]           # Runs de test (imagery)
    else:  # hands_feet
        all_runs = [5, 6, 9, 10, 13, 14]
        test_runs = [6, 10, 14]
    
    print("="*80)
    print("🚀 LEAVE-ONE-RUN-OUT TRAINING")
    print("="*80)
    print(f"Tâche: {task_name.upper()}")
    print(f"Sujets: {len(list(subject_ids))}")
    print(f"Runs disponibles: {all_runs}")
    print(f"Stratégie: Pour chaque run de test, entraîner sur tous les autres runs")
    print("="*80 + "\n")
    
    p = Processing()
    all_results = []
    failed = []
    
    for subject_id in subject_ids:
        print(f"\n{'='*80}")
        print(f"👤 SUJET {subject_id:03d}")
        print(f"{'='*80}")
        
        subject_results = []
        
        # Pour chaque run de test
        for test_run in test_runs:
            # Runs d'entraînement = tous sauf le run de test
            train_runs = [r for r in all_runs if r != test_run]
            
            print(f"\n  📊 Modèle pour TEST RUN {test_run}")
            print(f"     Train runs: {train_runs}")
            
            # Charger données d'entraînement (TOUS les autres runs)
            X_train_all, y_train_all = [], []
            for run_id in train_runs:
                try:
                    X, y = p.get_all_data(subject_id, run_id)
                    if len(X) > 0:
                        X_train_all.extend(X)
                        y_train_all.extend(y)
                        print(f"        Run {run_id:2d}: {len(X)} epochs")
                except:
                    continue
            
            # Charger données de test (UN SEUL run)
            X_test_all, y_test_all = [], []
            try:
                X, y = p.get_all_data(subject_id, test_run)
                if len(X) > 0:
                    X_test_all.extend(X)
                    y_test_all.extend(y)
                    print(f"        Test {test_run:2d}: {len(X)} epochs")
            except:
                pass
            
            # Vérifier données
            if len(X_train_all) < 20 or len(X_test_all) < 5:
                print(f"     ❌ Pas assez de données")
                continue
            
            # Uniformiser
            min_len_train = min(x.shape[1] for x in X_train_all)
            X_train = np.array([x[:, :min_len_train] for x in X_train_all])
            y_train = np.array(y_train_all)
            
            min_len_test = min(x.shape[1] for x in X_test_all)
            X_test = np.array([x[:, :min_len_test] for x in X_test_all])
            y_test = np.array(y_test_all)
            
            if len(set(y_train)) < 2 or len(set(y_test)) < 2:
                print(f"     ❌ Une seule classe présente")
                continue
            
            # Encoder
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_test_enc = le.transform(y_test)
            
            # Optimiser CSP
            optimal_n, cv_score = optimize_csp_components(X_train, y_train_enc, n_splits=3)
            
            # Entraîner modèle
            csp = CSP(n_components=optimal_n, reg=0.1, log=True, norm_trace=False)
            X_train_csp = csp.fit_transform(X_train, y_train_enc)
            X_test_csp = csp.transform(X_test)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_csp)
            X_test_scaled = scaler.transform(X_test_csp)
            
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            clf.fit(X_train_scaled, y_train_enc)
            
            # Évaluer
            train_score = clf.score(X_train_scaled, y_train_enc)
            test_score = clf.score(X_test_scaled, y_test_enc)
            
            print(f"     ✅ n_comp={optimal_n}, train={train_score:.3f}, test={test_score:.3f}")
            
            # Sauvegarder modèle
            model_dir = f"models/leave_one_run_out/{task_name}"
            os.makedirs(model_dir, exist_ok=True)
            
            model = {
                'csp': csp,
                'scaler': scaler,
                'clf': clf,
                'label_encoder': le,
                'subject_id': subject_id,
                'test_run': test_run,
                'train_runs': train_runs,
                'train_score': train_score,
                'test_score': test_score,
                'n_components': optimal_n,
                'cv_score': cv_score
            }
            
            model_path = os.path.join(model_dir, f"subject_{subject_id:03d}_testrun_{test_run}.pkl")
            joblib.dump(model, model_path)
            
            subject_results.append({
                'subject_id': subject_id,
                'test_run': test_run,
                'train_score': train_score,
                'test_score': test_score,
                'n_components': optimal_n,
                'n_train': len(X_train),
                'n_test': len(X_test)
            })
        
        if subject_results:
            # Moyenne pour ce sujet
            mean_test = np.mean([r['test_score'] for r in subject_results])
            print(f"\n  📊 Sujet {subject_id:03d} mean test: {mean_test:.3f}")
            all_results.extend(subject_results)
        else:
            failed.append(subject_id)
    
    # Statistiques globales
    print("\n\n" + "="*80)
    print("📊 RÉSULTATS GLOBAUX - LEAVE-ONE-RUN-OUT")
    print("="*80)
    print(f"Tâche: {task_name.upper()}\n")
    
    if all_results:
        test_scores = [r['test_score'] for r in all_results]
        train_scores = [r['train_score'] for r in all_results]
        
        print(f"Modèles réussis: {len(all_results)} ({len(all_results)//3} sujets × 3 runs)")
        print(f"Sujets échoués: {len(failed)}")
        
        print(f"\n🎯 Scores de test:")
        print(f"   Mean: {np.mean(test_scores):.4f} ({np.mean(test_scores)*100:.2f}%)")
        print(f"   Std:  {np.std(test_scores):.4f}")
        print(f"   Min:  {np.min(test_scores):.4f}")
        print(f"   Max:  {np.max(test_scores):.4f}")
        
        # Comparaison par run de test
        print(f"\n📊 Performance par run de test:")
        for test_run in test_runs:
            scores_run = [r['test_score'] for r in all_results if r['test_run'] == test_run]
            if scores_run:
                print(f"   Run {test_run:2d}: {np.mean(scores_run)*100:.2f}% ± {np.std(scores_run)*100:.2f}%")
        
        print(f"\n💾 Modèles sauvegardés dans: models/leave_one_run_out/{task_name}/")
        print("="*80)
        
        return {
            'results': all_results,
            'failed_subjects': failed,
            'mean_test_score': np.mean(test_scores),
            'std_test_score': np.std(test_scores)
        }
    else:
        print("❌ Aucun résultat")
        return None


if __name__ == "__main__":
    import pickle
    
    print("\n" + "🎯"*40)
    print("STRATÉGIE LEAVE-ONE-RUN-OUT")
    print("Pour chaque run de test, entraîner sur TOUS les autres runs")
    print("Maximise les données d'entraînement!")
    print("🎯"*40 + "\n")
    
    all_results = {}
    
    # Tâche 1: LEFT vs RIGHT
    print("\n" + "="*80)
    print("TÂCHE 1: LEFT vs RIGHT")
    print("="*80)
    results_lr = train_leave_one_run_out(
        subject_ids=range(1, 110),
        task_name="left_right"
    )
    all_results['left_right'] = results_lr
    
    # Tâche 2: HANDS vs FEET
    print("\n" + "="*80)
    print("TÂCHE 2: HANDS vs FEET")
    print("="*80)
    results_hf = train_leave_one_run_out(
        subject_ids=range(1, 110),
        task_name="hands_feet"
    )
    all_results['hands_feet'] = results_hf
    
    # Résumé global
    print("\n\n" + "="*80)
    print("🏆 RÉSUMÉ GLOBAL - TOUTES TÂCHES")
    print("="*80)
    
    if results_lr and results_hf:
        all_test_scores = (
            [r['test_score'] * 100 for r in results_lr['results']] +
            [r['test_score'] * 100 for r in results_hf['results']]
        )
        
        global_mean = np.mean(all_test_scores)
        global_std = np.std(all_test_scores)
        
        print(f"\n✅ LEFT vs RIGHT:  {results_lr['mean_test_score']*100:.2f}% ± {results_lr['std_test_score']*100:.2f}%")
        print(f"✅ HANDS vs FEET:  {results_hf['mean_test_score']*100:.2f}% ± {results_hf['std_test_score']*100:.2f}%")
        print(f"\n🏆 MOYENNE GLOBALE: {global_mean:.2f}% ± {global_std:.2f}%")
        
        if global_mean >= 65.0:
            print(f"\n🎉🎉🎉 OBJECTIF ATTEINT! Score ≥ 65% 🎉🎉🎉")
        else:
            gap = 65.0 - global_mean
            print(f"\n📈 Écart à combler: {gap:.2f}%")
        
        # Sauvegarder
        with open('results_leave_one_run_out.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        
        print(f"\n💾 Résultats sauvegardés: results_leave_one_run_out.pkl")
    
    print("\n" + "="*80 + "\n")
