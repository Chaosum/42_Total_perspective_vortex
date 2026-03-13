"""Script principal BCI — interface en ligne de commande.

Utilisation :
    python mybci.py train  --subject <N>  --task <left_right|hands_feet>
    python mybci.py predict --subject <N> --task <left_right|hands_feet> [--run <R>]

Pipeline : Leave-One-Run-Out (CSP + StandardScaler + LDA)
  - left_right  : runs 3,4,7,8,11,12  — classer left_fist vs right_fist
  - hands_feet  : runs 5,6,9,10,13,14 — classer both_fists vs both_feet
"""

import argparse
import os
import sys

import joblib
import numpy as np

# Accès aux modules du dossier src/ quel que soit le répertoire de lancement
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from processing import Processing
from train_leave_one_run_out import train_leave_one_run_out

# Runs de test par tâche (imagery uniquement)
TEST_RUNS = {
    "left_right":  [4, 8, 12],
    "hands_feet":  [6, 10, 14],
}

# Chemin racine des modèles (relatif au répertoire de lancement)
def _model_path(subject_id: int, task: str, test_run: int) -> str:
    base = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "leave_one_run_out", task
    )
    return os.path.join(base, f"subject_{subject_id:03d}_testrun_{test_run}.pkl")


# ──────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────

def cmd_train(args):
    """Entraîne le pipeline LORO pour un sujet et une tâche donnés."""
    print(f"\nTraining subject {args.subject:03d}, task: {args.task}")
    train_leave_one_run_out(
        subject_ids=[args.subject],
        task_name=args.task,
    )


# ──────────────────────────────────────────────
# PREDICT
# ──────────────────────────────────────────────

def _predict_run(subject_id: int, task: str, run_id: int) -> float | None:
    """Prédit sur un run de test et retourne l'accuracy."""
    model_file = _model_path(subject_id, task, run_id)
    if not os.path.exists(model_file):
        print(f"Model not found: {model_file}")
        print(f"Train first: python mybci.py train --subject {subject_id} --task {task}")
        return None

    model = joblib.load(model_file)
    csp    = model["csp"]
    scaler = model["scaler"]
    clf    = model["clf"]
    le     = model["label_encoder"]

    p = Processing()
    data = p.setup_data(subject_id, run_id)
    subject_str = f"S{subject_id:03d}"

    X_list, y_list = data[subject_str][run_id]
    if len(X_list) == 0:
        print(f"No data for run {run_id}")
        return None

    X = np.array(X_list)
    y_true_labels = np.array(y_list)

    # Transformation
    X_csp    = csp.transform(X)
    X_scaled = scaler.transform(X_csp)

    y_pred_enc = clf.predict(X_scaled)
    y_pred_labels = le.inverse_transform(y_pred_enc)
    y_true_enc    = le.transform(y_true_labels)

    accuracy = float(np.mean(y_pred_enc == y_true_enc))
    y_proba  = clf.predict_proba(X_scaled)

    print(f"\nRun {run_id:2d} - {len(X)} epochs")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Classes: {list(le.classes_)}")

    print(f"\n{'Epoch':>5}  {'Predicted':>14}  {'True':>14}  {'Conf':>6}  OK")
    print(f"{'─'*5}  {'─'*14}  {'─'*14}  {'─'*6}  ──")
    for i in range(len(X)):
        conf = float(np.max(y_proba[i]))
        ok   = "Y" if y_pred_labels[i] == y_true_labels[i] else "N"
        print(f"{i+1:>5}  {y_pred_labels[i]:>14}  {y_true_labels[i]:>14}  {conf:>6.2f}  {ok}")

    return accuracy


def cmd_predict(args):
    """Prédit sur un ou tous les runs de test d'un sujet."""
    print(f"\nPrediction - subject {args.subject:03d}, task: {args.task}")

    if args.run is not None:
        _predict_run(args.subject, args.task, args.run)
    else:
        scores = []
        for run_id in TEST_RUNS[args.task]:
            acc = _predict_run(args.subject, args.task, run_id)
            if acc is not None:
                scores.append(acc)

        if scores:
            print(f"\n{'-'*50}")
            print(f"Subject {args.subject:03d} - {args.task}")
            print(f"Mean accuracy: {np.mean(scores)*100:.2f}%")
            print(f"Runs tested: {len(scores)}/{len(TEST_RUNS[args.task])}")
            print(f"{'-'*50}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BCI — pipeline Leave-One-Run-Out (CSP + LDA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemples :\n"
            "  python mybci.py train   --subject 1 --task left_right\n"
            "  python mybci.py predict --subject 1 --task left_right\n"
            "  python mybci.py predict --subject 1 --task left_right --run 4\n"
        ),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    p_train = sub.add_parser("train", help="Entraîner le modèle LORO pour un sujet")
    p_train.add_argument("--subject", type=int, required=True,
                         help="Numéro du sujet (1–109)")
    p_train.add_argument("--task", type=str, required=True,
                         choices=["left_right", "hands_feet"],
                         help="Tâche à entraîner")

    # --- predict ---
    p_pred = sub.add_parser("predict", help="Prédire sur les runs de test d'un sujet")
    p_pred.add_argument("--subject", type=int, required=True,
                        help="Numéro du sujet (1–109)")
    p_pred.add_argument("--task", type=str, required=True,
                        choices=["left_right", "hands_feet"],
                        help="Tâche")
    p_pred.add_argument("--run", type=int, default=None,
                        help="Run spécifique (optionnel, sinon toutes les runs de test)")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)


if __name__ == "__main__":
    main()

