#!/usr/bin/env python3
"""
Visualisation avancée des résultats de l'optimisation CSP.
Crée des graphiques pour analyser l'impact de l'optimisation.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def visualize_csp_optimization():
    """Crée des visualisations des résultats de l'optimisation CSP."""
    
    print("📊 Génération des visualisations...")
    
    # Charger les modèles
    results = {'left_right': [], 'hands_feet': []}
    
    for task in ['left_right', 'hands_feet']:
        model_files = sorted(glob.glob(f'models/subject_specific/{task}/*.pkl'))
        
        for model_path in model_files:
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                results[task].append({
                    'subject_id': model.get('subject_id', 0),
                    'test_score': model.get('test_score', 0) * 100,
                    'train_score': model.get('train_score', 0) * 100,
                    'n_components': model.get('n_components', 6),
                    'cv_score': model.get('cv_score', 0) * 100
                })
            except:
                continue
    
    if not results['left_right'] or not results['hands_feet']:
        print("⚠️  Pas assez de données pour visualiser")
        return
    
    # Créer les figures
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Distribution des composantes CSP
    ax1 = plt.subplot(2, 3, 1)
    for task, color in [('left_right', 'blue'), ('hands_feet', 'red')]:
        n_comps = [r['n_components'] for r in results[task]]
        counts = [n_comps.count(n) for n in [4, 6, 8]]
        ax1.bar([4, 6, 8], counts, alpha=0.5, label=task.replace('_', ' ').title(), color=color, width=0.4)
    ax1.set_xlabel('Nombre de composantes CSP')
    ax1.set_ylabel('Nombre de sujets')
    ax1.set_title('Distribution des composantes CSP optimales')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Scores de test par nombre de composantes
    ax2 = plt.subplot(2, 3, 2)
    all_data = results['left_right'] + results['hands_feet']
    for n in [4, 6, 8]:
        scores = [r['test_score'] for r in all_data if r['n_components'] == n]
        if scores:
            ax2.boxplot([scores], positions=[n], widths=0.5)
    ax2.set_xlabel('Nombre de composantes CSP')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Performance par nombre de composantes')
    ax2.axhline(y=65, color='r', linestyle='--', label='Objectif 65%')
    ax2.axhline(y=50, color='gray', linestyle='--', label='Chance')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xticks([4, 6, 8])
    
    # 3. Histogramme des performances
    ax3 = plt.subplot(2, 3, 3)
    all_scores = [r['test_score'] for r in all_data]
    ax3.hist(all_scores, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(x=np.mean(all_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_scores):.1f}%')
    ax3.axvline(x=65, color='orange', linestyle='--', linewidth=2, label='Objectif: 65%')
    ax3.set_xlabel('Test Accuracy (%)')
    ax3.set_ylabel('Nombre de sujets')
    ax3.set_title('Distribution des performances')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Corrélation CV score vs Test score
    ax4 = plt.subplot(2, 3, 4)
    cv_scores = [r['cv_score'] for r in all_data]
    test_scores = [r['test_score'] for r in all_data]
    n_comps = [r['n_components'] for r in all_data]
    
    colors = ['blue' if n == 4 else 'green' if n == 6 else 'red' for n in n_comps]
    ax4.scatter(cv_scores, test_scores, c=colors, alpha=0.5)
    ax4.set_xlabel('CV Score (%)')
    ax4.set_ylabel('Test Score (%)')
    ax4.set_title('Corrélation CV vs Test')
    ax4.plot([0, 100], [0, 100], 'k--', alpha=0.3)
    ax4.grid(alpha=0.3)
    
    # Légende pour les couleurs
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='4 composantes'),
        Patch(facecolor='green', label='6 composantes'),
        Patch(facecolor='red', label='8 composantes')
    ]
    ax4.legend(handles=legend_elements, loc='lower right')
    
    # 5. Comparaison par tâche
    ax5 = plt.subplot(2, 3, 5)
    lr_scores = [r['test_score'] for r in results['left_right']]
    hf_scores = [r['test_score'] for r in results['hands_feet']]
    
    ax5.boxplot([lr_scores, hf_scores], labels=['Left/Right', 'Hands/Feet'])
    ax5.set_ylabel('Test Accuracy (%)')
    ax5.set_title('Performance par tâche')
    ax5.axhline(y=65, color='r', linestyle='--', label='Objectif 65%')
    ax5.axhline(y=50, color='gray', linestyle='--', label='Chance')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. Overfitting analysis (Train vs Test)
    ax6 = plt.subplot(2, 3, 6)
    train_scores = [r['train_score'] for r in all_data]
    test_scores = [r['test_score'] for r in all_data]
    
    ax6.scatter(train_scores, test_scores, alpha=0.5, c=colors)
    ax6.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Perfect generalization')
    ax6.set_xlabel('Train Score (%)')
    ax6.set_ylabel('Test Score (%)')
    ax6.set_title('Généralisation (Train vs Test)')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('csp_optimization_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Graphique sauvegardé: csp_optimization_analysis.png")
    
    # Statistiques textuelles
    print("\n" + "="*80)
    print("📊 STATISTIQUES DÉTAILLÉES")
    print("="*80)
    
    print("\n🔧 Impact du nombre de composantes:")
    for n in [4, 6, 8]:
        scores = [r['test_score'] for r in all_data if r['n_components'] == n]
        if scores:
            print(f"  {n} composantes: {np.mean(scores):.1f}% ± {np.std(scores):.1f}% (n={len(scores)})")
    
    print("\n📈 Comparaison des tâches:")
    for task in ['left_right', 'hands_feet']:
        scores = [r['test_score'] for r in results[task]]
        print(f"  {task.replace('_', ' ').title()}: {np.mean(scores):.1f}% ± {np.std(scores):.1f}%")
    
    print("\n🎯 Sujets au-dessus de 65%:")
    above_65 = sum(1 for r in all_data if r['test_score'] >= 65)
    print(f"  {above_65}/{len(all_data)} sujets ({100*above_65/len(all_data):.1f}%)")
    
    print("\n🏆 Top 10 sujets:")
    sorted_results = sorted(all_data, key=lambda x: x['test_score'], reverse=True)
    for i, r in enumerate(sorted_results[:10], 1):
        print(f"  {i:2d}. Sujet {r['subject_id']:03d}: {r['test_score']:5.1f}% (n_comp={r['n_components']})")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        visualize_csp_optimization()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("Note: matplotlib est nécessaire pour la visualisation")
        print("Installez avec: pip install matplotlib")
