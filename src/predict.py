import numpy as np
import joblib
import time

from processing import Processing

# -----------------------
# Étape 1 — Charger les modèles entraînés
# -----------------------
wavelet_pipeline = joblib.load("wavelet_pipeline.pkl")
csp_pipeline = joblib.load("csp_pipeline.pkl")
clf = joblib.load("classifier.pkl")

# -----------------------
# Étape 2 — Charger les données EEG
# -----------------------
print("🔄 Chargement des données EEG...")
p = Processing()
subject = p.setup_data()

X_total = []
y_total = []

for subject_id in subject:
    for run_id in subject[subject_id]:
        X, y = subject[subject_id][run_id]
        X_total.extend(X)
        y_total.extend(y)

X_total = np.array(X_total)
y_total = np.array(y_total)

# -----------------------
# Étape 3 — Boucle de prédiction sur chaque epoch
# -----------------------

print("🧠 Prédictions (streaming simulé)")
for i, epoch in enumerate(X_total):
    start = time.time()

    # Ajouter une dimension batch (1, n_channels, n_times)
    epoch = epoch[np.newaxis, :, :]

    # Transformer
    X_csp = csp_pipeline.transform(epoch)
    X_wavelet = wavelet_pipeline.transform(epoch)
    X_combined = np.hstack([X_csp, X_wavelet])

    # Prédiction
    y_pred = clf.predict(X_combined)

    elapsed = time.time() - start
    print(f"Epoch {i:02d} | Prediction: {y_pred[0]} | Truth: {y_total[i]} | Time: {elapsed:.3f}s")

    # Optionnel : assert temps < 2s
    if elapsed > 2:
        print("⚠️ Prédiction trop lente !")
