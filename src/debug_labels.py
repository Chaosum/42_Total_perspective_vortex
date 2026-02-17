"""
Script de diagnostic pour comprendre le problème de performance.
On va vérifier:
1. Distribution des labels
2. Que les bonnes runs sont utilisées
3. Que les labels correspondent bien aux runs
4. La structure des données après CSP
"""

import numpy as np
from processing import Processing
from global_variable import useful_runs
from sklearn.preprocessing import LabelEncoder
from mne.decoding import CSP

def check_labels_distribution():
    """Vérifie la distribution des labels pour différentes configurations."""
    
    p = Processing()
    
    print("="*70)
    print("🔍 DIAGNOSTIC DES LABELS")
    print("="*70)
    
    # Test 1: Vérifier les runs Left/Right (3, 4, 7, 8, 11, 12)
    print("\n1️⃣  Configuration LEFT vs RIGHT")
    print("   Runs utilisées: 3, 4, 7, 8, 11, 12")
    print("   Classes attendues: ['left_fist', 'right_fist']")
    print("-"*70)
    
    left_right_runs = [3, 4, 7, 8, 11, 12]
    
    # Charger quelques sujets pour vérifier
    for subject_id in [1, 2, 3]:
        print(f"\n   Sujet {subject_id}:")
        all_labels = []
        
        for run_id in left_right_runs:
            try:
                X, y = p.get_all_data(subject_id, run_id)
                if len(y) > 0:
                    unique, counts = np.unique(y, return_counts=True)
                    print(f"      Run {run_id:2d} ({useful_runs[run_id]['modal']:8s}): {len(y):2d} epochs → {dict(zip(unique, counts))}")
                    all_labels.extend(y)
            except Exception as e:
                print(f"      Run {run_id:2d}: ⚠️  {e}")
        
        if all_labels:
            unique, counts = np.unique(all_labels, return_counts=True)
            print(f"   📊 TOTAL: {dict(zip(unique, counts))}")
    
    # Test 2: Vérifier les runs Hands/Feet (5, 6, 9, 10, 13, 14)
    print("\n\n2️⃣  Configuration HANDS vs FEET")
    print("   Runs utilisées: 5, 6, 9, 10, 13, 14")
    print("   Classes attendues: ['both_fists', 'both_feet']")
    print("-"*70)
    
    hands_feet_runs = [5, 6, 9, 10, 13, 14]
    
    for subject_id in [1, 2, 3]:
        print(f"\n   Sujet {subject_id}:")
        all_labels = []
        
        for run_id in hands_feet_runs:
            try:
                X, y = p.get_all_data(subject_id, run_id)
                if len(y) > 0:
                    unique, counts = np.unique(y, return_counts=True)
                    print(f"      Run {run_id:2d} ({useful_runs[run_id]['modal']:8s}): {len(y):2d} epochs → {dict(zip(unique, counts))}")
                    all_labels.extend(y)
            except Exception as e:
                print(f"      Run {run_id:2d}: ⚠️  {e}")
        
        if all_labels:
            unique, counts = np.unique(all_labels, return_counts=True)
            print(f"   📊 TOTAL: {dict(zip(unique, counts))}")


def check_csp_output():
    """Vérifie que CSP produit des features discriminantes."""
    
    print("\n\n" + "="*70)
    print("🔍 DIAGNOSTIC CSP")
    print("="*70)
    
    p = Processing()
    
    # Charger un petit échantillon
    print("\n   Chargement d'un échantillon pour LEFT vs RIGHT...")
    X_all = []
    y_all = []
    
    for subject_id in [1, 2]:
        for run_id in [3, 4]:  # Une run real + une imagery
            X, y = p.get_all_data(subject_id, run_id)
            if len(X) > 0:
                X_all.extend(X)
                y_all.extend(y)
    
    # Uniformiser
    min_length = min(x.shape[1] for x in X_all)
    X_uniform = np.array([x[:, :min_length] for x in X_all])
    y_uniform = np.array(y_all)
    
    print(f"   Shape après uniformisation: {X_uniform.shape}")
    print(f"   Labels: {np.unique(y_uniform, return_counts=True)}")
    
    # Encoder les labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_uniform)
    print(f"   Classes: {le.classes_}")
    print(f"   Encoded: {np.unique(y_encoded, return_counts=True)}")
    
    # Appliquer CSP
    print("\n   Application de CSP...")
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    X_csp = csp.fit_transform(X_uniform, y_encoded)
    
    print(f"   Shape après CSP: {X_csp.shape}")
    
    # Vérifier la séparabilité des classes
    print("\n   Statistiques par classe:")
    for cls_name, cls_idx in zip(le.classes_, range(len(le.classes_))):
        mask = y_encoded == cls_idx
        X_cls = X_csp[mask]
        print(f"      {cls_name:12s}: mean={X_cls.mean(axis=0)[:3]}, std={X_cls.std(axis=0)[:3]}")
    
    # Calculer la distance entre les centroïdes
    mean_0 = X_csp[y_encoded == 0].mean(axis=0)
    mean_1 = X_csp[y_encoded == 1].mean(axis=0)
    distance = np.linalg.norm(mean_0 - mean_1)
    print(f"\n   Distance entre centroïdes: {distance:.4f}")
    
    # Vérifier si les features sont trop similaires (indicateur de problème)
    if distance < 0.5:
        print("   ⚠️  PROBLÈME: Distance très faible entre les classes!")
        print("       Les features CSP ne semblent pas discriminantes.")
    else:
        print("   ✅ Distance raisonnable entre les classes.")


def check_label_consistency():
    """Vérifie que les labels correspondent bien aux runs."""
    
    print("\n\n" + "="*70)
    print("🔍 VÉRIFICATION COHÉRENCE LABELS <-> RUNS")
    print("="*70)
    
    print("\nMapping attendu (global_variable.py):")
    for run_id, info in useful_runs.items():
        print(f"   Run {run_id:2d}: T1='{info['T1']}', T2='{info['T2']}' ({info['modal']})")
    
    p = Processing()
    
    print("\n\nVérification sur quelques epochs...")
    for subject_id in [1]:
        for run_id in [3, 5]:
            print(f"\n   Sujet {subject_id}, Run {run_id}:")
            print(f"   Attendu: T1='{useful_runs[run_id]['T1']}', T2='{useful_runs[run_id]['T2']}'")
            
            try:
                X, y = p.get_all_data(subject_id, run_id)
                if len(y) > 0:
                    unique = np.unique(y)
                    print(f"   Obtenu: {unique}")
                    
                    # Vérifier que les labels sont bien ceux attendus
                    expected = [useful_runs[run_id]['T1'], useful_runs[run_id]['T2']]
                    if set(unique) == set(expected):
                        print("   ✅ Labels corrects!")
                    else:
                        print(f"   ❌ PROBLÈME: Labels incorrects!")
                        print(f"      Attendu: {expected}")
                        print(f"      Obtenu: {list(unique)}")
            except Exception as e:
                print(f"   ⚠️  Erreur: {e}")


if __name__ == "__main__":
    check_labels_distribution()
    check_label_consistency()
    check_csp_output()
    
    print("\n\n" + "="*70)
    print("🏁 FIN DU DIAGNOSTIC")
    print("="*70)
