#!/usr/bin/env python3
"""
Script d'analyse des résultats d'entraînement subject-specific avec optimisation CSP.
Lit les modèles sauvegardés et affiche des statistiques détaillées.
"""

import pickle
import glob
import numpy as np
from pathlib import Path

def analyze_models():
    """Analyse les modèles subject-specific sauvegardés."""
    
    print("="*80)
    print("📊 ANALYSE DES MODÈLES SUBJECT-SPECIFIC OPTIMISÉS")
    print("="*80 + "\n")
    
    results = {'left_right': [], 'hands_feet': []}
    
    for task in ['left_right', 'hands_feet']:
        task_name = "LEFT vs RIGHT" if task == 'left_right' else "HANDS vs FEET"
        print(f"\n{'='*80}")
        print(f"📁 Analyse: {task_name}")
        print(f"{'='*80}")
        
        model_files = sorted(glob.glob(f'models/subject_specific/{task}/*.pkl'))
        
        if not model_files:
            print(f"⚠️  Aucun modèle trouvé pour {task}")
            continue
        
        print(f"✅ {len(model_files)} modèles trouvés\n")
        
        test_scores = []
        train_scores = []
        n_components_list = []
        cv_scores = []
        subjects = []
        
        for model_path in model_files:
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                subject_id = model.get('subject_id', 0)
                test_score = model.get('test_score', 0)
                train_score = model.get('train_score', 0)
                n_comp = model.get('n_components', 6)
                cv_score = model.get('cv_score', 0)
                
                test_scores.append(test_score)
                train_scores.append(train_score)
                n_components_list.append(n_comp)
                cv_scores.append(cv_score)
                subjects.append(subject_id)
                
                results[task].append({
                    'subject_id': subject_id,
                    'test_score': test_score,
                    'train_score': train_score,
                    'n_components': n_comp,
                    'cv_score': cv_score
                })
                
            except Exception as e:
                print(f"⚠️  Erreur lors du chargement de {model_path}: {e}")
                continue
        
        if not test_scores:
            print(f"⚠️  Aucune donnée valide pour {task}")
            continue
        
        # Convertir en arrays numpy
        test_scores = np.array(test_scores) * 100
        train_scores = np.array(train_scores) * 100
        n_components_list = np.array(n_components_list)
        cv_scores = np.array(cv_scores) * 100
        
        # Statistiques de performance
        print(f"📈 SCORES DE TEST (mouvement imaginé):")
        print(f"   Moyenne:    {test_scores.mean():.2f}% ± {test_scores.std():.2f}%")
        print(f"   Médiane:    {np.median(test_scores):.2f}%")
        print(f"   Min:        {test_scores.min():.2f}%")
        print(f"   Max:        {test_scores.max():.2f}%")
        print(f"   Q1 (25%):   {np.percentile(test_scores, 25):.2f}%")
        print(f"   Q3 (75%):   {np.percentile(test_scores, 75):.2f}%")
        
        print(f"\n📊 SCORES DE CALIBRATION (mouvement réel):")
        print(f"   Moyenne:    {train_scores.mean():.2f}% ± {train_scores.std():.2f}%")
        
        print(f"\n🔧 COMPOSANTES CSP OPTIMALES:")
        for nc in [4, 6, 8]:
            count = (n_components_list == nc).sum()
            pct = 100 * count / len(n_components_list)
            bar = "█" * int(pct / 3)
            print(f"   {nc} composantes: {count:3d} sujets ({pct:5.1f}%) {bar}")
        
        print(f"\n🎯 DISTRIBUTION DES PERFORMANCES:")
        bins = [(0, 50), (50, 60), (60, 65), (65, 70), (70, 80), (80, 100)]
        for low, high in bins:
            count = ((test_scores >= low) & (test_scores < high)).sum()
            pct = 100 * count / len(test_scores)
            bar = "█" * int(pct / 2)
            print(f"   {low:3d}-{high:3d}%: {count:3d} sujets ({pct:5.1f}%) {bar}")
        
        print(f"\n🏆 TOP 10 SUJETS:")
        top_indices = np.argsort(test_scores)[-10:][::-1]
        for i, idx in enumerate(top_indices, 1):
            subj = subjects[idx]
            score = test_scores[idx]
            n_comp = n_components_list[idx]
            print(f"   {i:2d}. Sujet {subj:03d}: {score:5.2f}% (n_comp={n_comp})")
        
        print(f"\n⚠️  BOTTOM 10 SUJETS:")
        bottom_indices = np.argsort(test_scores)[:10]
        for i, idx in enumerate(bottom_indices, 1):
            subj = subjects[idx]
            score = test_scores[idx]
            n_comp = n_components_list[idx]
            print(f"   {i:2d}. Sujet {subj:03d}: {score:5.2f}% (n_comp={n_comp})")
    
    # Résumé global
    if results['left_right'] and results['hands_feet']:
        print(f"\n\n{'='*80}")
        print("🏆 RÉSUMÉ GLOBAL - TOUTES TÂCHES")
        print("="*80)
        
        all_test_scores = (
            [r['test_score'] * 100 for r in results['left_right']] +
            [r['test_score'] * 100 for r in results['hands_feet']]
        )
        
        mean_global = np.mean(all_test_scores)
        std_global = np.std(all_test_scores)
        
        print(f"\n📊 Performance globale:")
        print(f"   Moyenne: {mean_global:.2f}% ± {std_global:.2f}%")
        print(f"   N total: {len(all_test_scores)} modèles")
        
        # Objectif 65%
        n_above_65 = sum(1 for s in all_test_scores if s >= 65)
        pct_above_65 = 100 * n_above_65 / len(all_test_scores)
        
        print(f"\n🎯 Objectif 65%:")
        print(f"   Sujets ≥ 65%: {n_above_65}/{len(all_test_scores)} ({pct_above_65:.1f}%)")
        
        if mean_global >= 65.0:
            print(f"\n🎉🎉🎉 OBJECTIF ATTEINT! Moyenne ≥ 65% 🎉🎉🎉")
        else:
            gap = 65.0 - mean_global
            print(f"\n📈 Progression:")
            print(f"   Écart à combler: +{gap:.2f}%")
            print(f"   Pourcentage de l'objectif: {100 * mean_global / 65.0:.1f}%")
        
        # Sauvegarder l'analyse
        analysis_results = {
            'left_right': results['left_right'],
            'hands_feet': results['hands_feet'],
            'global_mean': mean_global,
            'global_std': std_global,
            'n_above_65': n_above_65
        }
        
        with open('analysis_subject_specific_optimized.pkl', 'wb') as f:
            pickle.dump(analysis_results, f)
        
        print(f"\n💾 Analyse sauvegardée dans: analysis_subject_specific_optimized.pkl")
    
    print("\n" + "="*80)
    print("✅ ANALYSE TERMINÉE")
    print("="*80 + "\n")


if __name__ == "__main__":
    analyze_models()
