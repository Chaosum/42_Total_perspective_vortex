from collections import Counter 
import numpy as np
import joblib
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from MyLogisticRegression import MyLogisticRegression
from processing import Processing
from waveletsTransformer import WaveletTransformer
from MyCSP import MyCSP
from MyPCA import MyPCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

def train(subject_id = None, run_id=None):

    p = Processing()
    X_total, y_total = p.get_all_data(subject_id, run_id) # X = shape: (n_samples, n_channels, n_times) Y = shape: (n_samples,)
    label_encoder = LabelEncoder()
    y_total = label_encoder.fit_transform(y_total)

    print("🧾 Répartition des classes :", Counter(y_total))
    # -----------------------
    # Étape 2 — Construire les pipelines
    # -----------------------

    wavelet_pipeline = Pipeline([
        ("wavelet", WaveletTransformer()), # fit().transform() sur chaque epoch
        ("pca", MyPCA(n_components=10)) # fit().transform() sur les features extraites et prend les 10 meilleures
    ])

    csp_pipeline = Pipeline([
        ("csp", MyCSP(n_components=4))
    ])

    # -----------------------
    # Étape 3 — Feature extraction
    # -----------------------

    print("🧠 Extraction des features CSP...")
    X_csp = csp_pipeline.fit_transform(X_total, y_total)

    print("📊 Extraction des features wavelet + PCA...")
    X_wavelet_pca = wavelet_pipeline.fit_transform(X_total, y_total)

    # Concaténation des features
    X_combined = np.hstack([X_csp, X_wavelet_pca])

    # -----------------------
    # Étape 4 — Split explicite TrainVal/Test
    # -----------------------
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_combined, y_total, test_size=0.2, random_state=42, stratify=y_total
    )

    # -----------------------
    # Étape 5 — Cross-validation sur X_trainval
    # -----------------------
    print("📈 Évaluation par validation croisée...")

    # normalisation des données
    scaler = StandardScaler()
    X_trainval = scaler.fit_transform(X_trainval)
    X_test = scaler.transform(X_test)

    clf_my =  MyLogisticRegression(learning_rate=0.01, max_iter=1000)
    clf_my.fit(X_trainval, y_trainval)
    scores_my = cross_val_score(clf_my, X_trainval, y_trainval, cv=5)
    print("🧪 Mon LogisticRegression:")
    print("  Scores CV :", np.round(scores_my, 4))
    print("  Moyenne   :", round(np.mean(scores_my), 4))

    # -----------------------
    # Étape 6 — Entraînement final + test sur X_test
    # -----------------------
    print("🎯 Entraînement final sur trainval et test sur X_test...")
    clf_final =  MyLogisticRegression(learning_rate=0.01, max_iter=1000)
    clf_final.fit(X_trainval, y_trainval)
    y_pred = clf_final.predict(X_test)

    acc_test = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy finale sur X_test : {acc_test:.4f}")


    # 🔍 Diagnostic de répartition des prédictions
    print("🧾 Répartition des labels dans y_test :", Counter(y_test))
    print("📌 Répartition des prédictions :", Counter(y_pred))

    # 📄 Rapport de classification (sans warning)
    print("🧾 Rapport de classification :")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    print("📊 Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    # -----------------------
    # Étape 7 — Sauvegarde
    # -----------------------
    print("💾 Sauvegarde des modèles entraînés...")
    joblib.dump(wavelet_pipeline, "wavelet_pipeline.pkl")
    joblib.dump(csp_pipeline, "csp_pipeline.pkl")
    joblib.dump(clf_final, "classifier.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    print("✅ Modèles sauvegardés avec succès.")
