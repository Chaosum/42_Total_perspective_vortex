"""
Script de prédiction utilisant les modèles subject-specific.

Usage:
    python src/predict_subject_specific.py --subject 1 --task left_right
    python src/predict_subject_specific.py --subject 90 --task hands_feet --run 6
"""

import argparse
import numpy as np
import joblib
import os
from processing import Processing


def load_subject_model(subject_id, task_name="left_right"):
    """Charge le modèle d'un sujet spécifique."""
    
    model_path = f"models/subject_specific/{task_name}/subject_{subject_id:03}.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}\n"
                                f"Entraînez d'abord avec train_subject_specific_main.py")
    
    model = joblib.load(model_path)
    return model


def predict_subject(subject_id, run_id, task_name="left_right"):
    """
    Fait des prédictions pour un sujet sur une run donnée.
    
    Args:
        subject_id: ID du sujet
        run_id: ID de la run
        task_name: Nom de la tâche
    
    Returns:
        dict: Prédictions et métriques
    """
    
    print("="*70)
    print(f"🔮 PRÉDICTION SUBJECT-SPECIFIC")
    print("="*70)
    print(f"Sujet: {subject_id:03}")
    print(f"Run: {run_id}")
    print(f"Tâche: {task_name}")
    print("="*70)
    
    # Charger le modèle du sujet
    print(f"\n📦 Chargement du modèle du sujet {subject_id:03}...")
    model = load_subject_model(subject_id, task_name)
    
    csp = model['csp']
    scaler = model['scaler']
    clf = model['clf']
    le = model['label_encoder']
    
    print(f"✅ Modèle chargé")
    print(f"   Classes: {le.classes_}")
    print(f"   Entraîné avec runs: {model['calibration_runs']}")
    print(f"   Score calibration: {model['train_score']:.4f}")
    
    # Charger les données
    print(f"\n📊 Chargement des données...")
    p = Processing()
    X, y = p.get_all_data(subject_id, run_id)
    
    if len(X) == 0:
        print("❌ Aucune donnée disponible")
        return None
    
    print(f"✅ {len(X)} epochs chargés")
    
    # Uniformiser
    min_length = min(x.shape[1] for x in X)
    X_uniform = np.array([x[:, :min_length] for x in X])
    
    # Transformer et prédire
    print(f"\n🔧 Transformation et prédiction...")
    X_csp = csp.transform(X_uniform)
    X_scaled = scaler.transform(X_csp)
    
    y_pred = clf.predict(X_scaled)
    y_pred_labels = le.inverse_transform(y_pred)
    
    y_proba = clf.predict_proba(X_scaled)
    
    # Métriques si on a les vraies labels
    y_true = np.array(y)
    y_true_enc = le.transform(y_true)
    
    accuracy = np.mean(y_pred == y_true_enc)
    
    print(f"✅ Prédictions effectuées")
    
    # Afficher les résultats
    print(f"\n📊 RÉSULTATS:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Classes prédites: {np.unique(y_pred_labels)}")
    
    # Distribution
    unique_pred, counts_pred = np.unique(y_pred_labels, return_counts=True)
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    
    print(f"\n   Distribution prédite:")
    for label, count in zip(unique_pred, counts_pred):
        pct = 100 * count / len(y_pred_labels)
        print(f"      {label}: {count} ({pct:.1f}%)")
    
    print(f"\n   Distribution réelle:")
    for label, count in zip(unique_true, counts_true):
        pct = 100 * count / len(y_true)
        print(f"      {label}: {count} ({pct:.1f}%)")
    
    # Confiance moyenne
    mean_confidence = np.max(y_proba, axis=1).mean()
    print(f"\n   Confiance moyenne: {mean_confidence:.4f}")
    
    # Epoch par epoch
    if len(X) <= 20:
        print(f"\n📝 Détail par epoch:")
        for i in range(len(X)):
            pred_label = y_pred_labels[i]
            true_label = y_true[i]
            confidence = np.max(y_proba[i])
            correct = "✓" if pred_label == true_label else "✗"
            print(f"      Epoch {i+1:2d}: {pred_label:12s} (conf={confidence:.2f}) "
                  f"vs {true_label:12s} {correct}")
    
    return {
        'predictions': y_pred_labels,
        'true_labels': y_true,
        'probabilities': y_proba,
        'accuracy': accuracy,
        'mean_confidence': mean_confidence
    }


def evaluate_subject_all_runs(subject_id, task_name="left_right"):
    """Évalue un sujet sur toutes ses runs de test."""
    
    print("="*70)
    print(f"🧪 ÉVALUATION COMPLÈTE - SUJET {subject_id:03}")
    print("="*70)
    
    # Charger le modèle
    model = load_subject_model(subject_id, task_name)
    test_runs = model['test_runs']
    
    print(f"Tâche: {task_name}")
    print(f"Runs de test: {test_runs}\n")
    
    scores = []
    
    for run_id in test_runs:
        result = predict_subject(subject_id, run_id, task_name)
        if result:
            scores.append(result['accuracy'])
            print()
    
    if scores:
        print("="*70)
        print(f"📊 RÉSUMÉ - SUJET {subject_id:03}")
        print("="*70)
        print(f"Mean accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
        print(f"Runs testées: {len(scores)}/{len(test_runs)}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Prédiction avec modèles subject-specific')
    parser.add_argument('--subject', type=int, required=True,
                        help='ID du sujet')
    parser.add_argument('--task', type=str, default='left_right',
                        choices=['left_right', 'hands_feet'],
                        help='Tâche')
    parser.add_argument('--run', type=int, default=None,
                        help='ID de la run (si None, teste toutes les runs)')
    parser.add_argument('--all', action='store_true',
                        help='Évaluer toutes les runs de test')
    
    args = parser.parse_args()
    
    if args.all or args.run is None:
        # Évaluer toutes les runs de test
        evaluate_subject_all_runs(args.subject, args.task)
    else:
        # Prédire sur une run spécifique
        result = predict_subject(args.subject, args.run, args.task)


if __name__ == "__main__":
    main()
