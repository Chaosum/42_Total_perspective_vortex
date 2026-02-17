#!/usr/bin/env python3
"""
Entraînement subject-specific avec optimisation CSP.
Lance l'entraînement pour les 109 sujets avec les deux tâches.
Optimise automatiquement le nombre de composantes CSP (4, 6 ou 8) pour chaque sujet.
"""

import sys
import pickle
from train_incremental import train_subject_specific

def main():
    print("\n" + "="*80)
    print("🚀 ENTRAÎNEMENT SUBJECT-SPECIFIC AVEC OPTIMISATION CSP")
    print("="*80)
    print("Stratégie: Optimiser n_components CSP pour chaque sujet via CV")
    print("Tâches: LEFT vs RIGHT + HANDS vs FEET")
    print("Sujets: 109 (tous les sujets disponibles)")
    print("="*80 + "\n")
    
    all_results = {}
    
    # Tâche 1: LEFT vs RIGHT
    print("\n" + "🎯"*40)
    print("TÂCHE 1: LEFT vs RIGHT")
    print("🎯"*40 + "\n")
    
    results_lr = train_subject_specific(
        subject_ids=range(1, 110),
        calibration_runs=[3, 7, 11],  # 3 runs de calibration (PLUS DE DONNÉES!)
        test_runs=[4, 8, 12],         # 3 runs de test (mouvement imaginé)
        task_name="left_right"
    )
    
    all_results['left_right'] = results_lr
    
    # Tâche 2: HANDS vs FEET
    print("\n" + "🎯"*40)
    print("TÂCHE 2: HANDS vs FEET")
    print("🎯"*40 + "\n")
    
    results_hf = train_subject_specific(
        subject_ids=range(1, 110),
        calibration_runs=[5, 9, 13],  # 3 runs de calibration (PLUS DE DONNÉES!)
        test_runs=[6, 10, 14],        # 3 runs de test (mouvement imaginé)
        task_name="hands_feet"
    )
    
    all_results['hands_feet'] = results_hf
    
    # Résumé global
    print("\n\n" + "="*80)
    print("📊 RÉSUMÉ GLOBAL - TOUTES TÂCHES")
    print("="*80)
    
    if results_lr and results_hf:
        print(f"\n✅ LEFT vs RIGHT:")
        print(f"   Mean test accuracy: {results_lr['mean_test_score']*100:.2f}% ± {results_lr['std_test_score']*100:.2f}%")
        print(f"   Sujets réussis: {len(results_lr['results'])}")
        
        print(f"\n✅ HANDS vs FEET:")
        print(f"   Mean test accuracy: {results_hf['mean_test_score']*100:.2f}% ± {results_hf['std_test_score']*100:.2f}%")
        print(f"   Sujets réussis: {len(results_hf['results'])}")
        
        # Calculer moyenne globale
        all_test_scores = (
            [r['test_score'] for r in results_lr['results']] +
            [r['test_score'] for r in results_hf['results']]
        )
        import numpy as np
        global_mean = np.mean(all_test_scores) * 100
        global_std = np.std(all_test_scores) * 100
        
        print(f"\n🏆 MOYENNE GLOBALE (toutes tâches):")
        print(f"   {global_mean:.2f}% ± {global_std:.2f}%")
        
        # Vérifier si on atteint 65%
        if global_mean >= 65.0:
            print(f"\n🎉🎉🎉 OBJECTIF ATTEINT! Score ≥ 65% 🎉🎉🎉")
        elif global_mean >= 60.0:
            print(f"\n🔥 Très proche! Score ≥ 60%")
            print(f"   Besoin de +{65.0 - global_mean:.2f}% pour atteindre 65%")
        else:
            print(f"\n⚠️  Score actuel: {global_mean:.2f}%")
            print(f"   Besoin de +{65.0 - global_mean:.2f}% pour atteindre 65%")
        
        # Sauvegarder tous les résultats
        with open('results_subject_specific_optimized.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        
        print(f"\n💾 Résultats sauvegardés dans: results_subject_specific_optimized.pkl")
        print(f"💾 Modèles sauvegardés dans: models/subject_specific/")
        
    print("\n" + "="*80)
    print("✅ ENTRAÎNEMENT TERMINÉ")
    print("="*80 + "\n")
    
    return all_results


if __name__ == "__main__":
    results = main()
