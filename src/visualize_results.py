"""
Script V.1.2: Visualiser les résultats de la pipeline BCI

Usage:
    python src/visualize_results.py
    
Permet de:
1. Charger les résultats du training (per-subject ou global)
2. Afficher les statistiques detaillées (mean, std, distribution)
3. Créer des graphes de comparaison (train vs test, tâches, sujets)
4. Analyser la performance par tâche et par sujet
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from mybci import run_per_subject
from utils import experiments


class ResultsVisualizer:
    """Visualise les résultats du pipeline BCI"""
    
    def __init__(self):
        self.results = {}  # {exp_name: [results by subject]}
        self.global_results = {}
        
    def run_experiments(self, save_json=True):
        """Exécute toutes les expériences et collecte les résultats"""
        print("\n" + "="*70)
        print("EXECUTING BCI PIPELINE - V.1.2 Treatment Pipeline")
        print("="*70)
        
        for exp_id in range(1, 7):
            exp_name = experiments[exp_id-1]['name']
            print(f"\n[{exp_id}/6] Running: {exp_name}...")
            
            # Capture results from run_per_subject
            # Note: This is a simplified approach - in production you'd modify
            # run_per_subject to return results instead of printing
            try:
                results = self._run_experiment_and_capture(exp_id)
                self.results[exp_name] = results
                print(f"    ✓ {len(results)} subjects processed")
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        if save_json:
            self._save_results_to_json()
        
        return self.results
    
    def _run_experiment_and_capture(self, exp_id: int) -> List[Dict]:
        """
        Exécute une expérience et retourne les résultats.
        """
        try:
            # Appel direct à run_per_subject qui retourne maintenant les résultats
            results = run_per_subject(exp_id)
            return results if results else []
        except Exception as e:
            print(f"    ❌ Erreur lors de l'exécution: {e}")
            return []
    
    def load_results_from_file(self, filepath: str) -> bool:
        """Charge les résultats depuis un fichier JSON"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.results = data
            print(f"✓ Résultats chargés depuis {filepath}")
            return True
        except FileNotFoundError:
            print(f"❌ Fichier non trouvé: {filepath}")
            return False
    
    def _save_results_to_json(self):
        """Sauvegarde les résultats en JSON"""
        output_file = "bci_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Résultats sauvegardés: {output_file}")
    
    def print_statistics(self, exp_name: str, results: List[Dict]):
        """Affiche les statistiques détaillées d'une expérience"""
        
        if not results:
            print(f"  ⚠️  Pas de résultats pour {exp_name}")
            return
        
        test_scores = [r.get('test', 0.5) for r in results]
        train_scores = [r.get('train', 0.5) for r in results]
        cv_scores = [r.get('cv', 0.5) for r in results]
        
        print(f"\n{'='*70}")
        print(f"📊 STATISTIQUES - {exp_name.replace('_', ' ').upper()}")
        print(f"{'='*70}")
        
        print(f"\n🎯 Test Accuracy:")
        print(f"   Mean:   {np.mean(test_scores):.4f} ({np.mean(test_scores)*100:.1f}%)")
        print(f"   Std:    {np.std(test_scores):.4f}")
        print(f"   Median: {np.median(test_scores):.4f}")
        print(f"   Min:    {np.min(test_scores):.4f} (Subject {results[np.argmin(test_scores)]['subject']})")
        print(f"   Max:    {np.max(test_scores):.4f} (Subject {results[np.argmax(test_scores)]['subject']})")
        
        print(f"\n📈 Training Accuracy:")
        print(f"   Mean:   {np.mean(train_scores):.4f}")
        print(f"   Std:    {np.std(train_scores):.4f}")
        
        print(f"\n✔️  Cross-Validation Accuracy:")
        print(f"   Mean:   {np.mean(cv_scores):.4f}")
        print(f"   Std:    {np.std(cv_scores):.4f}")
        
        # Statistical test vs 50% chance
        t_stat, p_value = stats.ttest_1samp(test_scores, 0.5)
        print(f"\n🔬 Significance (vs 50% chance):")
        print(f"   t-stat: {t_stat:.4f}")
        print(f"   p-value: {p_value:.2e}")
        if p_value < 0.001:
            print(f"   ✅ Highly significant (p < 0.001)")
        elif p_value < 0.05:
            print(f"   ✅ Significant (p < 0.05)")
        else:
            print(f"   ❌ Not significant")
        
        # Performance categories
        print(f"\n📊 Performance Distribution:")
        categories = [
            ("Outstanding (>80%)", lambda s: s > 0.8),
            ("Very Good (70-80%)", lambda s: 0.7 <= s <= 0.8),
            ("Good (60-70%)", lambda s: 0.6 <= s < 0.7),
            ("Above Chance (50-60%)", lambda s: 0.5 <= s < 0.6),
            ("Below Chance (<50%)", lambda s: s < 0.5)
        ]
        
        for label, condition in categories:
            count = sum(1 for s in test_scores if condition(s))
            pct = 100 * count / len(test_scores) if test_scores else 0
            bar = "█" * int(pct / 2)
            print(f"   {label:25s}: {count:3d} ({pct:5.1f}%) {bar}")
        
        # Top 5 subjects
        sorted_results = sorted(results, key=lambda r: r.get('test', 0), reverse=True)
        print(f"\n🏆 Top 5 Subjects:")
        for i, r in enumerate(sorted_results[:5], 1):
            print(f"   {i}. Subject {r['subject']:3d}: "
                  f"Test={r.get('test', 0):.3f}, Train={r.get('train', 0):.3f}, CV={r.get('cv', 0):.3f}")
    
    def plot_accuracy_distribution(self, exp_name: str, results: List[Dict]):
        """Crée des graphes de distribution d'accuracy"""
        
        if not results:
            return
        
        test_scores = [r.get('test', 0.5) for r in results]
        train_scores = [r.get('train', 0.5) for r in results]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(test_scores, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Chance (50%)')
        axes[0].axvline(np.mean(test_scores), color='green', linestyle='-', linewidth=2, 
                       label=f'Mean ({np.mean(test_scores):.1%})')
        axes[0].axvline(np.median(test_scores), color='orange', linestyle=':', linewidth=2,
                       label=f'Median ({np.median(test_scores):.1%})')
        axes[0].set_xlabel('Test Accuracy')
        axes[0].set_ylabel('Number of Subjects')
        axes[0].set_title(f'{exp_name.replace("_", " ").title()}\nTest Accuracy Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, 1)
        
        # Train vs Test scatter
        axes[1].scatter(train_scores, test_scores, alpha=0.6, s=80, color='steelblue', edgecolor='black')
        axes[1].plot([0, 1], [0, 1], 'g--', linewidth=2, label='Perfect generalization')
        axes[1].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
        axes[1].set_xlabel('Train Accuracy')
        axes[1].set_ylabel('Test Accuracy')
        axes[1].set_title(f'{exp_name.replace("_", " ").title()}\nTrain vs Test')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0.5, 1.0)
        axes[1].set_ylim(0, 1.0)
        
        plt.tight_layout()
        filename = f'results_{exp_name}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Graphique sauvegardé: {filename}")
        plt.close()
    
    def plot_comparison_all_experiments(self):
        """Compare toutes les expériences"""
        
        if not self.results:
            print("❌ Aucun résultat à afficher")
            return
        
        exp_names = list(self.results.keys())
        means = []
        stds = []
        
        for exp_name in exp_names:
            results = self.results[exp_name]
            test_scores = [r.get('test', 0.5) for r in results]
            means.append(np.mean(test_scores))
            stds.append(np.std(test_scores))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(exp_names))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Chance (50%)')
        ax.axhline(0.65, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target (65%)')
        
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title('Comparison of BCI Pipeline Performance Across Tasks', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([name.replace('_', '\n') for name in exp_names], fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.02, f'{mean:.1%}', ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results_comparison_all_experiments.png', dpi=150, bbox_inches='tight')
        print("✓ Graphique comparatif sauvegardé: results_comparison_all_experiments.png")
        plt.close()
    
    def visualize_all(self):
        """Lance la visualisation complète"""
        
        print("\n" + "="*70)
        print("📊 BCI RESULTS VISUALIZATION")
        print("="*70)
        
        if not self.results:
            print("\n⚠️  Aucun résultat chargé. Exécution des expériences...")
            self.run_experiments()
        
        # Afficher stats pour chaque expérience
        for exp_name, results in self.results.items():
            print(f"\n[*] Traitement: {exp_name}")
            self.print_statistics(exp_name, results)
            self.plot_accuracy_distribution(exp_name, results)
        
        # Graphe comparatif
        print(f"\n[*] Création du graphe comparatif...")
        self.plot_comparison_all_experiments()
        
        print("\n" + "="*70)
        print("✅ VISUALISATION TERMINÉE")
        print("="*70)
        print("\nGraphes générés:")
        for exp_name in self.results.keys():
            print(f"  - results_{exp_name}.png")
        print("  - results_comparison_all_experiments.png")
        print("\nPour examiner en détail, ouvrez les fichiers PNG")


def create_demo_results():
    """Crée des résultats de démo pour tester la visualisation"""
    demo_results = {}
    
    for exp_id, exp in enumerate(experiments, 1):
        exp_name = exp['name']
        # Simuler des résultats
        np.random.seed(42)
        n_subjects = 109
        
        results = []
        # Distribution: moyenne 65%, mais variable
        mean_acc = 0.63 + np.random.rand() * 0.08  # 63-71%
        
        for subj_id in range(1, n_subjects + 1):
            # Sample from beta distribution for realistic accuracy distribution
            test_acc = np.random.beta(3, 3) * 0.4 + 0.5  # Biased vers 0.5-0.9
            train_acc = test_acc + np.random.randn() * 0.1
            cv_acc = (test_acc + train_acc) / 2 + np.random.randn() * 0.05
            
            results.append({
                'subject': subj_id,
                'train': np.clip(train_acc, 0.5, 1.0),
                'test': np.clip(test_acc, 0, 1.0),
                'cv': np.clip(cv_acc, 0, 1.0)
            })
        
        demo_results[exp_name] = results
    
    return demo_results


if __name__ == "__main__":
    import sys
    
    # Create visualizer
    viz = ResultsVisualizer()
    
    # Check if demo mode requested
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        print("[*] Mode DEMO - Loading simulated results...")
        viz.results = create_demo_results()
    else:
        print("[*] Mode REEL - Running actual BCI pipeline...")
        print("    Cela peut prendre plusieurs minutes...\n")
        viz.run_experiments(save_json=True)
    
    # Run visualization
    viz.visualize_all()
    
    print("\n💡 Tips:")
    print("  - Utiliser --demo pour tester avec données simulées")
    print("  - Utiliser visualize_raw.py pour explorer les données brutes")
    print("  - Les résultats sont sauvegardés en JSON pour réutilisation")

