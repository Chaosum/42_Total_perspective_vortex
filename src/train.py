import os
import numpy as np
import joblib

from MyLogisticRegression import MyLogisticRegression
from processing import Processing
from waveletsTransformer import WaveletTransformer
from MyCSP import MyCSP
from MyPCA import MyPCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# -----------------------
# Étape 1 — Charger les données
# -----------------------
print("🔄 Chargement des données EEG...")
subject_id = os.getenv("BCI_SUBJECT")
run_id = int(os.getenv("BCI_RUN"))

p = Processing()
X_total, y_total = p.get_all_data() # X = shape: (n_samples, n_channels, n_times) Y = shape: (n_samples,)

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

# Séparer 80% pour l'entraînement, 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_total, test_size=0.2, random_state=42
)

# -----------------------
# Étape 4 — Évaluation (cross-validation)
# -----------------------
print("📈 Évaluation par validation croisée...")
clf = MyLogisticRegression(learning_rate=0.5, max_iter=1000)
scores = cross_val_score(clf, X_combined, y_total, cv=5)
print("✅ Score cross-val moyen :", np.mean(scores))

# -----------------------
# Étape 5 — Entraînement final et sauvegarde
# -----------------------
print("💾 Entraînement final et sauvegarde des modèles...")
clf.fit(X_combined, y_total)

joblib.dump(wavelet_pipeline, "wavelet_pipeline.pkl")
joblib.dump(csp_pipeline, "csp_pipeline.pkl")
joblib.dump(clf, "classifier.pkl")
print("✅ Modèles sauvegardés avec succès.")
