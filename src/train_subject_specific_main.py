"""
Script principal pour l'entraînement subject-specific.

Pour chaque sujet, on entraîne un modèle personnalisé avec:
- Calibration: runs de mouvement RÉEL
- Test: runs de mouvement IMAGINÉ

Cette approche donne typiquement 63-72% d'accuracy, bien meilleure
que l'approche subject-independent (~51%).

Usage:
    python src/train_subject_specific_main.py
    python src/train_subject_specific_main.py --subjects 1-20
    python src/train_subject_specific_main.py --task hands_feet
"""

import argparse
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mne.decoding import CSP
from processing import Processing


def train_subject_specific(
    subject_ids,
    calibration_runs,
    test_runs,
    task_name="left_right"
):
    """
    Entraînement subject-specific: un modèle par sujet.
    
    Args:
        subject_ids: IDs des sujets à traiter
        calibration_runs: Runs pour la calibration
        test_runs: Runs pour le test
        task_name: Nom de la tâche
    
    Returns:
        dict: Résultats et statistiques
    """
    
    print("="*70)
    print("🚀 ENTRAÎNEMENT SUBJECT-SPECIFIC")
    print("="*70)
    print(f"Tâche: {task_name.upper()}")
    print(f"Sujets: {len(list(subject_ids))}")
    print(f"Calibration runs: {calibration_runs} (mouvement réel)")
    print(f"Test runs: {test_runs} (mouvement imaginé)")
    print("="*70)
    print("\nStratégie: Entraîner un modèle spécifique pour chaque sujet")
    print("avec ses propres données de calibration.\n")
    
    p = Processing()
    results = []
    failed_subjects = []
    
    for subject_id in subject_ids:
        print(f"\n{'='*70}")
        print(f"👤 SUJET {subject_id:03}")
        print(f"{'='*70}")
        
        # CALIBRATION
        X_calib_all = []
        y_calib_all = []
        
        print(f"📚 Calibration (runs {calibration_runs})...")
        for run_id in calibration_runs:
            try:
                X, y = p.get_all_data(subject_id, run_id)
                if len(X) > 0:
                    X_calib_all.extend(X)
                    y_calib_all.extend(y)
                    print(f"   Run {run_id:2d}: {len(X)} epochs")
            except Exception as e:
                print(f"   Run {run_id:2d}: ⚠️  Erreur")
                continue
        
        # TEST
        X_test_all = []
        y_test_all = []
        
        print(f"\n🧪 Test (runs {test_runs})...")
        for run_id in test_runs:
            try:
                X, y = p.get_all_data(subject_id, run_id)
                if len(X) > 0:
                    X_test_all.extend(X)
                    y_test_all.extend(y)
                    print(f"   Run {run_id:2d}: {len(X)} epochs")
            except Exception as e:
                print(f"   Run {run_id:2d}: ⚠️  Erreur")
                continue
        
        # Vérifications
        if len(X_calib_all) < 10 or len(X_test_all) < 5:
            print(f"\n❌ Pas assez de données (calib={len(X_calib_all)}, test={len(X_test_all)})")
            failed_subjects.append(subject_id)
            continue
        
        # Uniformiser
        min_length_calib = min(x.shape[1] for x in X_calib_all)
        X_calib = np.array([x[:, :min_length_calib] for x in X_calib_all])
        y_calib = np.array(y_calib_all)
        
        min_length_test = min(x.shape[1] for x in X_test_all)
        X_test = np.array([x[:, :min_length_test] for x in X_test_all])
        y_test = np.array(y_test_all)
        
        print(f"\n📊 Données préparées:")
        print(f"   Calibration: {X_calib.shape}")
        print(f"   Test: {X_test.shape}")
        
        # Vérifier les classes
        if len(set(y_calib)) < 2 or len(set(y_test)) < 2:
            print(f"\n❌ Une seule classe présente")
            failed_subjects.append(subject_id)
            continue
        
        # Encoder
        le = LabelEncoder()
        y_calib_enc = le.fit_transform(y_calib)
        y_test_enc = le.transform(y_test)
        
        calib_dist = dict(zip(*np.unique(y_calib, return_counts=True)))
        test_dist = dict(zip(*np.unique(y_test, return_counts=True)))
        
        print(f"   Classes: {le.classes_}")
        print(f"   Distribution calibration: {calib_dist}")
        print(f"   Distribution test: {test_dist}")
        
        # Entraîner
        print(f"\n🔧 Entraînement du modèle subject-specific...")
        
        csp = CSP(n_components=6, reg=0.1, log=True, norm_trace=False)
        X_calib_csp = csp.fit_transform(X_calib, y_calib_enc)
        X_test_csp = csp.transform(X_test)
        
        scaler = StandardScaler()
        X_calib_scaled = scaler.fit_transform(X_calib_csp)
        X_test_scaled = scaler.transform(X_test_csp)
        
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_calib_scaled, y_calib_enc)
        
        # Évaluation
        train_score = clf.score(X_calib_scaled, y_calib_enc)
        test_score = clf.score(X_test_scaled, y_test_enc)
        
        y_pred = clf.predict(X_test_scaled)
        n_classes_pred = len(np.unique(y_pred))
        
        print(f"\n✅ Résultats:")
        print(f"   Score calibration: {train_score:.4f}")
        print(f"   Score test: {test_score:.4f}")
        print(f"   Classes prédites: {n_classes_pred}/2")
        
        # Sauvegarder
        model_dir = f"models/subject_specific/{task_name}"
        os.makedirs(model_dir, exist_ok=True)
        
        subject_model = {
            'csp': csp,
            'scaler': scaler,
            'clf': clf,
            'label_encoder': le,
            'subject_id': subject_id,
            'calibration_runs': calibration_runs,
            'test_runs': test_runs,
            'train_score': train_score,
            'test_score': test_score
        }
        
        model_path = os.path.join(model_dir, f"subject_{subject_id:03}.pkl")
        joblib.dump(subject_model, model_path)
        
        results.append({
            'subject_id': subject_id,
            'train_score': train_score,
            'test_score': test_score,
            'n_classes_pred': n_classes_pred,
            'n_calib': len(X_calib),
            'n_test': len(X_test)
        })
    
    # Statistiques globales
    print("\n\n" + "="*70)
    print("📊 RÉSULTATS GLOBAUX - SUBJECT-SPECIFIC")
    print("="*70)
    print(f"Tâche: {task_name.upper()}\n")
    
    if results:
        test_scores = [r['test_score'] for r in results]
        train_scores = [r['train_score'] for r in results]
        
        print(f"Sujets réussis: {len(results)}/{len(list(subject_ids))}")
        if failed_subjects:
            print(f"Sujets échoués: {failed_subjects}")
        
        print(f"\n📈 Scores de calibration:")
        print(f"   Mean: {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")
        print(f"   Min: {np.min(train_scores):.4f}")
        print(f"   Max: {np.max(train_scores):.4f}")
        
        print(f"\n🎯 Scores de test:")
        print(f"   Mean: {np.mean(test_scores):.4f} ± {np.std(test_scores):.4f}")
        print(f"   Min: {np.min(test_scores):.4f}")
        print(f"   Max: {np.max(test_scores):.4f}")
        
        # Histogramme
        print(f"\n📊 Distribution des scores de test:")
        bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i in range(len(bins)-1):
            count = sum(1 for s in test_scores if bins[i] <= s < bins[i+1])
            pct = 100 * count / len(test_scores) if test_scores else 0
            bar = "█" * int(pct / 2)
            print(f"   {bins[i]:.1f}-{bins[i+1]:.1f}: {count:3d} sujets ({pct:5.1f}%) {bar}")
        
        print(f"\n💾 Modèles sauvegardés dans: models/subject_specific/{task_name}/")
        print("="*70)
        
        return {
            'results': results,
            'failed_subjects': failed_subjects,
            'mean_test_score': np.mean(test_scores),
            'std_test_score': np.std(test_scores),
            'mean_train_score': np.mean(train_scores)
        }
    else:
        print("❌ Aucun sujet n'a pu être traité avec succès")
        return None


def main():
    parser = argparse.ArgumentParser(description='Entraînement subject-specific pour BCI')
    parser.add_argument('--subjects', type=str, default='1-109',
                        help='Range de sujets (ex: 1-20, 90-109)')
    parser.add_argument('--task', type=str, default='both',
                        choices=['left_right', 'hands_feet', 'both'],
                        help='Tâche à entraîner')
    
    args = parser.parse_args()
    
    # Parser le range de sujets
    if '-' in args.subjects:
        start, end = map(int, args.subjects.split('-'))
        subject_ids = range(start, end + 1)
    else:
        subject_ids = [int(args.subjects)]
    
    print("\n🧠 ENTRAÎNEMENT SUBJECT-SPECIFIC POUR BCI\n")
    
    results_summary = {}
    
    # Left vs Right
    if args.task in ['left_right', 'both']:
        print("\n" + "🔹" * 35)
        print("TÂCHE 1: LEFT vs RIGHT FIST")
        print("🔹" * 35 + "\n")
        
        results_lr = train_subject_specific(
            subject_ids=subject_ids,
            calibration_runs=[3, 7, 11],  # Mouvement réel
            test_runs=[4, 8, 12],          # Mouvement imaginé
            task_name="left_right"
        )
        results_summary['left_right'] = results_lr
    
    # Hands vs Feet
    if args.task in ['hands_feet', 'both']:
        print("\n" + "🔹" * 35)
        print("TÂCHE 2: HANDS vs FEET")
        print("🔹" * 35 + "\n")
        
        results_hf = train_subject_specific(
            subject_ids=subject_ids,
            calibration_runs=[5, 9, 13],   # Mouvement réel
            test_runs=[6, 10, 14],         # Mouvement imaginé
            task_name="hands_feet"
        )
        results_summary['hands_feet'] = results_hf
    
    # Résumé final
    print("\n\n" + "="*70)
    print("🏆 RÉSUMÉ FINAL - TOUTES TÂCHES")
    print("="*70)
    
    for task_name, results in results_summary.items():
        if results:
            print(f"\n{task_name.replace('_', ' ').title()}:")
            print(f"   Test accuracy: {results['mean_test_score']:.4f} ± {results['std_test_score']:.4f}")
            print(f"   Sujets réussis: {len(results['results'])}")
    
    print("\n" + "="*70)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("="*70)
    print("\nLes modèles subject-specific sont prêts à être utilisés!")
    print("Chaque sujet a maintenant son propre modèle calibré.\n")


if __name__ == "__main__":
    main()
