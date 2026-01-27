# TODO List — projet Total Perspective Vortex

## Priorité haute

2) Refondre `train.py` (conformité au sujet)
   - Construire un `Pipeline` sklearn complet :
     WaveletTransformer -> MyPCA / MyCSP -> StandardScaler -> Classifier (LogisticRegression recommandé)
   - Utiliser `cross_val_score` (cv stratifié) sur la pipeline entière et afficher mean ± std.
   - Sauvegarder la pipeline complète et `LabelEncoder` (joblib.dump).

3) Evaluation robuste
   - Remplacer split 80/20 par validation croisée stratifiée (k-fold) pour petits jeux de données.
   - Produire rapports par sujet/run et moyenne sur les 6 expériences demandées.

## Priorité moyenne

4) Nettoyer `processing.py`
   - Fusionner la double définition de `get_all_data`.
   - Rendre `BASE_PATH` configurable (paramètre/variable d'environnement).
   - Améliorer gestion d'erreurs si EDF manquent.

5) Normalisation & stabilité numérique
   - Appliquer `StandardScaler` ou log-scaling aux features wavelet avant PCA.
   - Vérifier variances nulles ou très faibles et corriger si nécessaire.

6) Classifieur
   - Option rapide : utiliser `sklearn.linear_model.LogisticRegression` dans la pipeline.
   - Option bonus : améliorer `MyLogisticRegression` (vectorisation, régularisation, critère d'arrêt).

## Priorité basse / infra

7) Predict (streaming <2s)
   - Refactoriser `predict.py` pour charger la pipeline complète et mesurer la latence par epoch (<2s).

8) Diagnostics & tests
   - Conserver `inspect_train_debug.py` et ajouter tests unitaires (shapes, non-NaN, variance non-nulle) et un test d'intégration pipeline+CV.

9) Docs & README
   - Rédiger `README.md` (setup venv, install requirements, commandes d'entraînement/prédiction, format des données, critères d'évaluation).

10) Nettoyage repo
   - Supprimer ou utiliser `utils.py`, organiser sauvegardes (`models/`), corriger imports/typos.

11) Validation finale multi-sujets
   - Lancer les 6 expériences pour plusieurs sujets, collecter les moyennes par expérience et vérifier l'objectif >= 60%.

## Notes rapides

- Avec très peu d'échantillons (ex. 15 epochs), utiliser CV plutôt que split 80/20 — un test de 3 échantillons n'est pas significatif.
- Inclure scaler dans le pipeline pour éviter data leakage lors de cross-validation.

## Prochaine action proposée

- Option A (recommandé) : Je génère `requirements.txt` et refonds `train.py` pour pipeline+CV puis j'exécute un test sur `4 14` et sur quelques sujets.
- Option B : Je crée seulement `requirements.txt`.

Marque la tâche que tu veux que je lance en premier et je m'en occupe.
