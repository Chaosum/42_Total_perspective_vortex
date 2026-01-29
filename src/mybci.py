"""Script principal BCI : entraînement et évaluation avec le pipeline complet.

Deux modes principaux :
- sans argument    → évaluation globale sur les 6 types d'expériences définis à partir des runs EEGMMI ;
- avec arguments   → `python mybci.py <subject> <run> train` (pour lancer l'entraînement de base).

Remarque importante :
- dans une version avancée, on entraînera un modèle dédié par type et on les sauvera
    sous des noms du style `full_pipeline_type{type_id}.pkl`.
- pour l'instant, ce script illustre la structure d'évaluation par type en
    utilisant un unique fichier `full_pipeline.pkl` sauvegardé par `train.py`.
"""

import sys
import numpy as np
import joblib

from processing import Processing
from global_variable import useful_runs
from train import train


# Définition des 6 types d'expériences à partir des runs de l'EEGMMI
# cf. description PhysioNet :
# - Task 1 : gauche/droite, mouvement réel des mains    → runs 3, 7, 11
# - Task 2 : gauche/droite, mouvement imaginaire mains  → runs 4, 8, 12
# - Task 3 : mains vs pieds, mouvement réel             → runs 5, 9, 13
# - Task 4 : mains vs pieds, mouvement imaginaire       → runs 6, 10, 14
#
# Nous définissons 6 "types" comme demandé dans le sujet :
#   0 : Task 1 et répliques
#   1 : Task 2 et répliques
#   2 : Task 3 et répliques
#   3 : Task 4 et répliques
#   4 : Toutes les tâches avec mouvement imaginaire (Task 2 + Task 4)
#   5 : Toutes les tâches avec mouvement réel       (Task 1 + Task 3)

EXPERIMENT_TYPES: dict[int, dict] = {
    0: {"name": "task1_real_left_right", "runs": [3, 7, 11]},
    1: {"name": "task2_imagery_left_right", "runs": [4, 8, 12]},
    2: {"name": "task3_real_hands_vs_feet", "runs": [5, 9, 13]},
    3: {"name": "task4_imagery_hands_vs_feet", "runs": [6, 10, 14]},
    4: {"name": "all_imagery_tasks", "runs": [4, 8, 12, 6, 10, 14]},
    5: {"name": "all_real_tasks", "runs": [3, 7, 11, 5, 9, 13]},
}


def eval_subject_run(subject_id: int, run_id: int, full_pipeline) -> float | None:
    """Évalue la précision du pipeline complet sur un sujet + run donnés.

    On récupère les epochs via Processing, puis on appelle simplement
    `full_pipeline.score(X, y)`.
    """

    subject_str = f"S{subject_id:03}"
    p = Processing()
    data = p.setup_data(subject_id, run_id)

    X_total = []
    y_total = []
    for run in data[subject_str]:
        X, y = data[subject_str][run]
        X_total.extend(X)
        y_total.extend(y)

    if len(set(y_total)) < 2:
        # Si on n'a qu'une seule classe, on ne peut pas calculer une accuracy pertinente
        return None

    X_total = np.array(X_total)
    y_total = np.array(y_total)

    acc = full_pipeline.score(X_total, y_total)
    return float(acc)


def main_no_args():
    """Mode sans argument : évaluation globale sur les 6 types d'expérience.

    Pour l'instant, on charge un unique `full_pipeline.pkl` et on l'évalue
    sur chaque type défini dans EXPERIMENT_TYPES. À terme, on pourra
    utiliser un pipeline dédié par type (ex: `full_pipeline_type{i}.pkl`).
    """

    try:
        full_pipeline = joblib.load("full_pipeline.pkl")
    except FileNotFoundError:
        print("❌ full_pipeline.pkl introuvable. Lance d'abord 'python src/train.py <subject> <run>'.")
        sys.exit(1)

    type_accuracies: dict[int, float] = {}

    # Boucle sur les 6 types définis plus haut
    for type_id, info in EXPERIMENT_TYPES.items():
        type_name = info["name"]
        runs = info["runs"]

        print(f"\n=== Type {type_id} – {type_name} ===")

        scores_for_type = []

        # Pour chaque run de ce type et pour chaque sujet, on évalue le pipeline.
        for run_id in runs:
            run_scores = []
            for subject_id in range(1, 110):
                acc = eval_subject_run(subject_id, run_id, full_pipeline)
                if acc is not None:
                    run_scores.append(acc)
                    print(f"type {type_id}, run {run_id}, subject {subject_id:03}: accuracy = {acc:.4f}")

            if run_scores:
                mean_run_acc = float(np.mean(run_scores))
                scores_for_type.extend(run_scores)
                print(f"Mean accuracy for run {run_id}: {mean_run_acc:.4f}")

        if scores_for_type:
            mean_type_acc = float(np.mean(scores_for_type))
            type_accuracies[type_id] = mean_type_acc
            print(f"\n📊 Mean accuracy for type {type_id} ({type_name}): {mean_type_acc:.4f}\n")
        else:
            print(f"Aucun score calculé pour le type {type_id} (peut-être trop de runs mono-classe).")

    if not type_accuracies:
        print("Aucun score global calculé (tous les types vides ?).")
        return

    print("📊 Mean accuracy over all types:")
    for type_id, acc in type_accuracies.items():
        print(f"Type {type_id} ({EXPERIMENT_TYPES[type_id]['name']}): accuracy = {acc:.4f}")

    final_mean = float(np.mean(list(type_accuracies.values())))
    print(f"\n🔚 Final mean accuracy over all types: {final_mean:.4f}")


def main_with_args(argv: list[str]):
    """Mode avec arguments : entraînement ou (futur) mode prédiction.

    Usage : python mybci.py <subject_num> <run_num> <train>
    """

    if len(argv) != 4:
        print("Usage: python mybci.py <subject_num> <run_num> <train>")
        sys.exit(1)

    subject = int(argv[1])
    run = int(argv[2])
    mode = argv[3]

    if not (1 <= subject <= 109):
        print("❌ Subject must be between 1 and 109 (S001 to S109)")
        sys.exit(1)

    if run not in useful_runs:
        print(f"❌ Run must be one of: {list(useful_runs.keys())}")
        sys.exit(1)

    if mode == "train":
        # On appelle simplement la fonction train du nouveau fichier `train.py`.
        train(subject, run)
    else:
        print("❌ Mode must be 'train'")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main_no_args()
    else:
        main_with_args(sys.argv)

