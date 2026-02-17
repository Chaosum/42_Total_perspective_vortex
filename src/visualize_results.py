"""
Script pour visualiser les résultats des modèles subject-specific.

Usage:
    python src/visualize_results.py
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt


def load_all_results(task_name="left_right"):
    """Charge tous les résultats d'une tâche."""
    
    model_dir = f"models/subject_specific/{task_name}"
    
    if not os.path.exists(model_dir):
        print(f"❌ Répertoire non trouvé: {model_dir}")
        return None
    
    results = []
    
    for filename in sorted(os.listdir(model_dir)):
        if filename.endswith('.pkl'):
            filepath = os.path.join(model_dir, filename)
            model = joblib.load(filepath)
            results.append({
                'subject_id': model['subject_id'],
                'train_score': model['train_score'],
                'test_score': model['test_score'],
                'calibration_runs': model['calibration_runs'],
                'test_runs': model['test_runs']
            })
    
    return results


def plot_accuracy_distribution(results, task_name):
    """Trace la distribution des accuracies."""
    
    test_scores = [r['test_score'] for r in results]
    train_scores = [r['train_score'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(test_scores, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(0.5, color='red', linestyle='--', label='Chance (50%)')
    axes[0].axvline(np.mean(test_scores), color='green', linestyle='-', linewidth=2, 
                   label=f'Mean ({np.mean(test_scores):.1%})')
    axes[0].set_xlabel('Test Accuracy')
    axes[0].set_ylabel('Number of Subjects')
    axes[0].set_title(f'{task_name.replace("_", " ").title()}\nTest Accuracy Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Train vs Test scatter
    axes[1].scatter(train_scores, test_scores, alpha=0.6, s=50)
    axes[1].plot([0, 1], [0, 1], 'r--', label='Perfect generalization')
    axes[1].axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='Chance')
    axes[1].set_xlabel('Calibration Accuracy')
    axes[1].set_ylabel('Test Accuracy')
    axes[1].set_title(f'{task_name.replace("_", " ").title()}\nCalibration vs Test')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0.5, 1.0)
    axes[1].set_ylim(0.2, 1.0)
    
    plt.tight_layout()
    plt.savefig(f'results_{task_name}.png', dpi=150)
    print(f"✅ Graphique sauvegardé: results_{task_name}.png")
    
    return fig


def print_statistics(results, task_name):
    """Affiche les statistiques détaillées."""
    
    test_scores = [r['test_score'] for r in results]
    train_scores = [r['train_score'] for r in results]
    
    print("\n" + "="*70)
    print(f"📊 STATISTIQUES - {task_name.replace('_', ' ').upper()}")
    print("="*70)
    
    print(f"\n🎯 Test Accuracy:")
    print(f"   Mean:   {np.mean(test_scores):.4f} ({np.mean(test_scores)*100:.1f}%)")
    print(f"   Std:    {np.std(test_scores):.4f}")
    print(f"   Median: {np.median(test_scores):.4f}")
    print(f"   Min:    {np.min(test_scores):.4f}")
    print(f"   Max:    {np.max(test_scores):.4f}")
    
    print(f"\n📈 Calibration Accuracy:")
    print(f"   Mean:   {np.mean(train_scores):.4f} ({np.mean(train_scores)*100:.1f}%)")
    print(f"   Std:    {np.std(train_scores):.4f}")
    
    print(f"\n📊 Performance Categories:")
    categories = [
        ("Outstanding (>80%)", lambda s: s > 0.8),
        ("Very Good (70-80%)", lambda s: 0.7 <= s <= 0.8),
        ("Good (60-70%)", lambda s: 0.6 <= s < 0.7),
        ("Above Chance (50-60%)", lambda s: 0.5 <= s < 0.6),
        ("Below Chance (<50%)", lambda s: s < 0.5)
    ]
    
    for label, condition in categories:
        count = sum(1 for s in test_scores if condition(s))
        pct = 100 * count / len(test_scores)
        bar = "█" * int(pct / 2)
        print(f"   {label:25s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    # Top 10 subjects
    sorted_results = sorted(results, key=lambda r: r['test_score'], reverse=True)
    
    print(f"\n🏆 Top 10 Subjects:")
    for i, r in enumerate(sorted_results[:10], 1):
        print(f"   {i:2d}. Subject {r['subject_id']:03d}: "
              f"Train={r['train_score']:.3f}, Test={r['test_score']:.3f}")
    
    # Bottom 10 subjects
    print(f"\n⚠️  Bottom 10 Subjects:")
    for i, r in enumerate(sorted_results[-10:][::-1], 1):
        print(f"   {i:2d}. Subject {r['subject_id']:03d}: "
              f"Train={r['train_score']:.3f}, Test={r['test_score']:.3f}")
    
    # Statistical significance
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(test_scores, 0.5)
    
    print(f"\n🔬 Statistical Significance (vs 50% chance):")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value: {p_value:.2e}")
    if p_value < 0.001:
        print(f"   ✅ Highly significant (p < 0.001)")
    elif p_value < 0.05:
        print(f"   ✅ Significant (p < 0.05)")
    else:
        print(f"   ❌ Not significant")


def compare_tasks():
    """Compare les deux tâches."""
    
    print("\n" + "="*70)
    print("🆚 COMPARAISON DES TÂCHES")
    print("="*70)
    
    results_lr = load_all_results("left_right")
    results_hf = load_all_results("hands_feet")
    
    if results_lr and results_hf:
        scores_lr = [r['test_score'] for r in results_lr]
        scores_hf = [r['test_score'] for r in results_hf]
        
        print(f"\nLeft vs Right:")
        print(f"   Mean: {np.mean(scores_lr):.4f} ± {np.std(scores_lr):.4f}")
        print(f"   Range: [{np.min(scores_lr):.3f}, {np.max(scores_lr):.3f}]")
        
        print(f"\nHands vs Feet:")
        print(f"   Mean: {np.mean(scores_hf):.4f} ± {np.std(scores_hf):.4f}")
        print(f"   Range: [{np.min(scores_hf):.3f}, {np.max(scores_hf):.3f}]")
        
        # Test de différence
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(scores_lr, scores_hf)
        
        diff = np.mean(scores_lr) - np.mean(scores_hf)
        print(f"\nDifférence: {diff:+.4f} ({diff*100:+.1f}%)")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            winner = "Left vs Right" if diff > 0 else "Hands vs Feet"
            print(f"✅ {winner} est significativement meilleur")
        else:
            print(f"⚖️  Pas de différence significative entre les tâches")


def main():
    """Fonction principale."""
    
    print("="*70)
    print("📊 VISUALISATION DES RÉSULTATS BCI")
    print("="*70)
    
    # Left vs Right
    results_lr = load_all_results("left_right")
    if results_lr:
        print(f"\n✅ Chargé {len(results_lr)} sujets (Left vs Right)")
        print_statistics(results_lr, "left_right")
        plot_accuracy_distribution(results_lr, "left_right")
    
    # Hands vs Feet
    results_hf = load_all_results("hands_feet")
    if results_hf:
        print(f"\n✅ Chargé {len(results_hf)} sujets (Hands vs Feet)")
        print_statistics(results_hf, "hands_feet")
        plot_accuracy_distribution(results_hf, "hands_feet")
    
    # Comparaison
    if results_lr and results_hf:
        compare_tasks()
    
    print("\n" + "="*70)
    print("✅ ANALYSE TERMINÉE")
    print("="*70)
    print("\nGraphiques générés:")
    print("   - results_left_right.png")
    print("   - results_hands_feet.png")


if __name__ == "__main__":
    main()
