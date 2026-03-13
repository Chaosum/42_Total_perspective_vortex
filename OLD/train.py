"""Entraînement d'un pipeline sklearn complet (CSP + wavelet + PCA + scaler + régression logistique).

Objectifs :
- être le plus simple et lisible possible ;
- utiliser une vraie validation croisée sur TOUT le pipeline ;
- sauvegarder un unique objet `full_pipeline.pkl` réutilisable pour la prédiction.
"""

from collections import Counter
import joblib
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from mne.decoding import CSP

from processing import Processing
from global_variable import useful_runs


def build_full_pipeline() -> Pipeline:
    """Construit le pipeline complet classique : CSP + PCA + scaler + régression logistique.

    Étapes :
    - CSP (MNE) extrait des features spatiales à partir des signaux EEG bruts ;
    - PCA (sklearn) réduit encore la dimension des features ;
    - StandardScaler met toutes les features sur la même échelle ;
    - LogisticRegression fait la classification finale.

    Ce pipeline prend en entrée X de shape (n_epochs, n_channels, n_times)
    et renvoie des prédictions de classe.
    """

    full_pipeline = Pipeline([
        # CSP : transforme les epochs brutes en features de variance projetées
        ("csp", CSP(n_components=8, log=True, norm_trace=False)),

        # Normalisation des features CSP
        ("scaler", StandardScaler()),

        # Classifieur linéaire classique
        ("clf", LogisticRegression(max_iter=1000))
    ])

    return full_pipeline


def train(subject_id=None, run_id=None):
    """Entraîne le pipeline complet sur les données d'un sujet / run.

    Étapes :
    1) récupération des données brutes (epochs EEG) via Processing ;
    2) encodage des labels (LabelEncoder) ;
    3) construction du pipeline complet ;
    4) validation croisée sur tout le pipeline ;
    5) entraînement final sur toutes les données ;
    6) sauvegarde du pipeline et de l'encoder.
    """

    p = Processing()
    X_total, y_total = p.get_all_data(subject_id, run_id)

    # Encodage des labels texte -> entiers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_total)

    print("🧾 Répartition des classes :", Counter(y_encoded))

    # Construction du pipeline complet
    full_pipeline = build_full_pipeline()

    # ==============================
    # 1) Validation croisée complète
    # ==============================
    print("📈 Validation croisée sur tout le pipeline (features + scaler + clf)...")
    scores = cross_val_score(full_pipeline, X_total, y_encoded, cv=5)
    print("  Scores CV :", np.round(scores, 4))
    print("  Moyenne   :", round(float(np.mean(scores)), 4))

    # ==============================
    # 2) Entraînement final
    # ==============================
    print("🎯 Entraînement final du pipeline sur toutes les données...")
    full_pipeline.fit(X_total, y_encoded)

    # ==============================
    # 3) Sauvegarde
    # ==============================
    print("💾 Sauvegarde du pipeline complet et du label encoder...")
    joblib.dump(full_pipeline, "full_pipeline.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    print("✅ Sauvegarde terminée.")


def train_all(
    subject_ids: range | list[int] = range(1, 20),
    run_ids: list[int] | None = None,
):
    """Entraîne un pipeline global sur plusieurs sujets et runs.

    Cette fonction construit un gros jeu de données X_total, y_total en
    concaténant les epochs de plusieurs sujets et plusieurs runs, puis :

    - effectue une validation croisée 5-fold sur tout ce dataset ;
    - entraîne le pipeline final sur toutes les données ;
    - sauvegarde `full_pipeline.pkl` et `label_encoder.pkl`.

    Paramètres par défaut :
    - subject_ids : sujets 1 à 19 (range(1, 20)) ;
    - run_ids : tous les runs considérés comme "utiles" dans useful_runs.
    """

    p = Processing()

    if run_ids is None:
        # On prend par défaut tous les runs marqués utiles
        run_ids = list(useful_runs.keys())

    X_all: list = []
    y_all: list = []

    print("🧾 Construction du dataset global pour train_all()...")

    for subject_id in subject_ids:
        for run_id in run_ids:
            try:
                X_subj_run, y_subj_run = p.get_all_data(subject_id, run_id)
            except Exception as e:
                # Si un sujet/run pose problème (fichier manquant, etc.), on le saute.
                print(f"⚠️  Impossible de charger subject={subject_id}, run={run_id} : {e}")
                continue

            if len(X_subj_run) == 0:
                continue

            print(
                f"  Ajout subject={subject_id}, run={run_id} : "
                f"{len(X_subj_run)} epochs"
            )

            X_all.extend(X_subj_run)
            y_all.extend(y_subj_run)

    if not X_all:
        print("❌ Aucun epoch chargé dans train_all(). Vérifie les sujets/runs.")
        return

    X_total = np.array(X_all)
    y_total = np.array(y_all)

    print(
        f"✅ Dataset global : {X_total.shape[0]} epochs, "
        f"shape par epoch = {X_total.shape[1:]}"
    )

    # Encodage des labels texte -> entiers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_total)

    print("🧾 Répartition des classes (global) :", Counter(y_encoded))

    # Construction du pipeline complet
    full_pipeline = build_full_pipeline()

    # 1) Validation croisée complète
    print("📈 Validation croisée globale sur tout le dataset (features + scaler + clf)...")
    scores = cross_val_score(full_pipeline, X_total, y_encoded, cv=5)
    print("  Scores CV :", np.round(scores, 4))
    print("  Moyenne   :", round(float(np.mean(scores)), 4))

    # 2) Entraînement final
    print("🎯 Entraînement final du pipeline global sur toutes les données...")
    full_pipeline.fit(X_total, y_encoded)

    # 3) Sauvegarde
    print("💾 Sauvegarde du pipeline global et du label encoder...")
    joblib.dump(full_pipeline, "full_pipeline.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    print("✅ Sauvegarde globale terminée.")


if __name__ == "__main__":
    import sys

    # Deux modes de lancement :
    # - sans argument : train_all() sur un ensemble de sujets/runs par défaut ;
    # - avec 2 arguments : train(subject, run) pour un entraînement ciblé.

    if len(sys.argv) == 1:
        # Entraînement global
        train_all()
    elif len(sys.argv) == 3:
        try:
            subject = int(sys.argv[1])
            run = int(sys.argv[2])
        except ValueError:
            print("Both subject and run must be integers")
            sys.exit(1)

        train(subject, run)
    else:
        print("Usage: python train.py [<subject_num> <run_num>]")
        sys.exit(1)

