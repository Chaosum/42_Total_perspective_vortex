
import sys
import os
import numpy as np
import joblib
# from processing import Processing
# from utils import cross_val_score_pipeline

# Définition des expériences (run, classes)
EXPERIMENTS = [
    {"id": 0, "run": 3, "classes": ("left_fist", "right_fist")},
    {"id": 1, "run": 4, "classes": ("left_fist", "right_fist")},
    {"id": 2, "run": 5, "classes": ("both_fists", "both_feet")},
    {"id": 3, "run": 6, "classes": ("both_fists", "both_feet")},
    {"id": 4, "run": 7, "classes": ("left_fist", "right_fist")},
    {"id": 5, "run": 8, "classes": ("left_foot", "right_foot")},
]

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def train(subject, experiment):
    """
    Entraîne un pipeline sklearn pour un sujet et une expérience (catégorie).
    """
    from processing import Processing
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    import joblib
    import os

    # Associer expérience à run et classes
    exp = EXPERIMENTS[experiment]
    run = exp["run"]
    print(f"[TRAIN] Sujet {subject:03d} | Expérience {experiment} | Run {run}")

    # Charger les données
    p = Processing()
    X, y = p.get_all_data(subject, run)
    if len(X) == 0:
        print("Aucune donnée trouvée pour ce sujet/run.")
        return

    # Convertir y en numérique : T1 -> 0, T2 -> 1
    y_numeric = np.array([0 if label == 'T1' else 1 for label in y])

    # Aplatir epochs pour PCA (n_epochs, n_channels * n_times)
    X_flat = X.reshape((X.shape[0], -1))

    # Pipeline avec tes classes personnalisées : CSP + PCA + LogisticRegression
    # CSP attend X en (n_epochs, n_channels, n_times), pas aplati
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'OLD', 'myImplementation'))
    from MyCSP import MyCSP
    from MyPCA import MyPCA
    from MyLogisticRegression import MyLogisticRegression

    # Aplatir après CSP pour PCA
    X_csp = X  # Garder la forme 3D pour CSP
    n_components_pca = min(10, X.shape[0], X.shape[1] * X.shape[2])  # Après aplatissement
    pipe = Pipeline([
        ("csp", MyCSP(n_components=4)),  # CSP pour extraire 4 composantes spatiales
        ("pca", MyPCA(n_components=n_components_pca)),  # PCA personnalisée sur les features CSP
        ("clf", MyLogisticRegression(max_iter=1000, lambda_=0.01))  # LogisticRegression personnalisée
    ])

    # Cross-validation (ajuster cv si peu de données)
    cv_folds = min(5, len(y_numeric))
    if cv_folds >= 2:
        try:
            scores = cross_val_score(pipe, X_csp, y_numeric, cv=cv_folds)
            print(f"cross_val_score: {scores.mean():.4f}")
        except ValueError:
            # Si CV impossible (pas assez d'échantillons par classe), skip
            print("Pas assez d'échantillons par classe pour cross-validation, skip.")
    else:
        print("Pas assez de données pour cross-validation, skip.")

    # Entraînement final
    pipe.fit(X_csp, y_numeric)

    # Sauvegarde avec métadonnées
    os.makedirs("models", exist_ok=True)
    model_path = f"models/subject_{subject:03d}_exp{experiment}.pkl"
    cv_score = scores.mean() if 'scores' in locals() and len(scores) > 0 else None
    model_data = {
        'pipeline': pipe,
        'cv_score': cv_score,
        'subject': subject,
        'experiment': experiment
    }
    joblib.dump(model_data, model_path)
    print(f"Modèle sauvegardé : {model_path}")

def predict(subject, run, experiment_id, return_acc=False):
    """
    Charge le modèle, prédit sur chaque epoch, affiche la vérité et la prédiction.
    Si return_acc=True, retourne l'accuracy (float), sinon None.
    """
    from processing import Processing
    import joblib
    import numpy as np
    import os

    exp = EXPERIMENTS[experiment_id]
    run = exp["run"]
    print(f"[PREDICT] Sujet {subject:03d} | Expérience {experiment_id} | Run {run}")

    model_path = f"models/subject_{subject:03d}_exp{experiment_id}.pkl"
    if not os.path.exists(model_path):
        print(f"Modèle non trouvé : {model_path}")
        return None if return_acc else None
    model_data = joblib.load(model_path)
    pipe = model_data['pipeline']

    p = Processing()
    X, y = p.get_all_data(subject, run)
    if len(X) == 0:
        print("Aucune donnée trouvée pour ce sujet/run.")
        return None if return_acc else None

    # Convertir y en numérique pour la prédiction
    y_numeric = np.array([0 if label == 'T1' else 1 for label in y])

    # Baseline : accuracy de la classe majoritaire
    baseline = max(np.sum(y == 'T1'), np.sum(y == 'T2')) / len(y)
    print(f"Baseline accuracy (majority class): {baseline:.4f}")

    # Utiliser X en 3D pour le pipeline CSP
    X_input = X  # (n_epochs, n_channels, n_times)
    y_pred_numeric = pipe.predict(X_input)

    # Convertir y_pred_numeric en labels pour affichage
    y_pred = ['T1' if pred == 0 else 'T2' for pred in y_pred_numeric]

    print("epoch nb: [prediction] [truth] equal?")
    correct = 0
    for i in range(len(y)):
        equal = y_pred[i] == y[i]
        print(f"epoch {i:02d}: [{y_pred[i]}] [{y[i]}] {str(equal)}")
        if equal:
            correct += 1
    acc = correct / len(y)
    print(f"Accuracy: {acc:.4f}")
    # Sauvegarde du score de test
    model_data['test_score'] = acc
    joblib.dump(model_data, model_path)
    if return_acc:
        return acc
    return None

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        # Mode global : cross-val sur tous les sujets et expériences
        all_accs = []
        for exp in EXPERIMENTS:
            accs = []
            for subject in range(1, 110):
                train(subject, exp["id"])
                acc = predict(subject, exp["run"], exp["id"], return_acc=True)
                if acc is not None:
                    accs.append(acc)
                    all_accs.append(acc)
            print(f"experiment {exp['id']}: accuracy = {np.mean(accs) if accs else 0:.4f}")
        print(f"Global mean accuracy: {np.mean(all_accs) if all_accs else 0:.4f}")
    elif len(args) == 2:
        subject = int(args[0])
        experiment_id = int(args[1])
        exp = EXPERIMENTS[experiment_id]
        run = exp["run"]
        train(subject, experiment_id)
        predict(subject, run, experiment_id)
    else:
        print("Usage: python mybci.py [subject run]")

if __name__ == "__main__":
    main()

