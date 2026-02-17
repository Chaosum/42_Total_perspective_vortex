"""
Entraînement avec des features plus génériques: Power Spectral Density (PSD)
au lieu de CSP. Les PSD features sont plus transférables entre sujets.
"""

import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import signal
from scipy.integrate import simps
from processing import Processing
from global_variable import useful_runs
import joblib


def extract_band_powers(epoch, sfreq=160.0):
    """
    Extrait les puissances dans différentes bandes de fréquences.
    
    Bandes EEG:
    - Delta: 0.5-4 Hz
    - Theta: 4-8 Hz
    - Alpha: 8-13 Hz
    - Beta: 13-30 Hz
    - Gamma: 30-50 Hz
    
    Pour BCI motor imagery, Beta (13-30 Hz) et Mu (8-13 Hz) sont les plus importants.
    """
    
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    features = []
    
    # Pour chaque canal
    for ch_idx in range(epoch.shape[0]):
        ch_data = epoch[ch_idx, :]
        
        # Calculer le spectre de puissance
        freqs, psd = signal.welch(ch_data, sfreq, nperseg=min(256, len(ch_data)))
        
        # Calculer la puissance dans chaque bande
        for band_name, (low_freq, high_freq) in bands.items():
            # Trouver les indices correspondant à la bande
            idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
            
            # Intégrer la puissance dans la bande
            band_power = simps(psd[idx_band], freqs[idx_band])
            features.append(band_power)
    
    return np.array(features)


def train_with_psd_features(
    train_subject_ids=range(1, 90),
    test_subject_ids=range(90, 110),
    run_ids=None,
    task_name="left_right"
):
    """
    Entraînement avec features PSD au lieu de CSP.
    
    Args:
        train_subject_ids: IDs des sujets pour l'entraînement
        test_subject_ids: IDs des sujets pour le test
        run_ids: Liste des runs à utiliser
        task_name: "left_right" ou "hands_feet"
    """
    
    if run_ids is None:
        if task_name == "left_right":
            run_ids = [3, 4, 7, 8, 11, 12]
        else:
            run_ids = [5, 6, 9, 10, 13, 14]
    
    print("="*70)
    print(f"🚀 ENTRAÎNEMENT AVEC PSD FEATURES - {task_name.upper()}")
    print("="*70)
    print(f"Sujets train: {len(list(train_subject_ids))}")
    print(f"Sujets test:  {len(list(test_subject_ids))}")
    print(f"Runs:         {run_ids}\n")
    
    p = Processing()
    
    # Étape 1: Collecter un échantillon pour fitter le scaler
    print("📊 Échantillonnage pour fitter le scaler...")
    X_sample_raw = []
    y_sample = []
    
    sample_subjects = list(train_subject_ids)[::10][:10]  # 10 sujets échantillonnés
    
    for subject_id in sample_subjects:
        for run_id in run_ids[:2]:  # 2 runs par sujet
            try:
                X, y = p.get_all_data(subject_id, run_id)
                if len(X) > 0:
                    # Prendre juste quelques epochs
                    n_samples = min(10, len(X))
                    indices = np.random.choice(len(X), n_samples, replace=False)
                    X_sample_raw.extend([X[i] for i in indices])
                    y_sample.extend([y[i] for i in indices])
            except:
                continue
    
    print(f"✅ {len(X_sample_raw)} epochs échantillonnés")
    
    # Extraire les features PSD
    print("🔧 Extraction des features PSD...")
    X_sample_psd = np.array([extract_band_powers(epoch) for epoch in X_sample_raw])
    y_sample = np.array(y_sample)
    
    print(f"   Shape PSD features: {X_sample_psd.shape}")
    
    # Label encoding
    le = LabelEncoder()
    y_sample_enc = le.fit_transform(y_sample)
    classes = le.classes_
    print(f"   Classes: {classes}\n")
    
    # Fitter le scaler
    scaler = StandardScaler()
    scaler.fit(X_sample_psd)
    
    # Initialiser le classifieur
    clf = SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=0.0001,
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        learning_rate='optimal',
        early_stopping=False
    )
    
    # Premier fit avec l'échantillon
    X_sample_scaled = scaler.transform(X_sample_psd)
    clf.partial_fit(X_sample_scaled, y_sample_enc, classes=np.arange(len(classes)))
    print(f"✅ Classifieur initialisé\n")
    
    # Étape 2: Entraînement incrémental
    print("📚 Entraînement incrémental...")
    train_subjects_list = list(train_subject_ids)
    batch_size = 5
    n_batches = (len(train_subjects_list) + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(train_subjects_list))
        batch_subjects = train_subjects_list[batch_start:batch_end]
        
        # Charger le batch
        X_batch_raw = []
        y_batch = []
        
        for subject_id in batch_subjects:
            for run_id in run_ids:
                try:
                    X, y = p.get_all_data(subject_id, run_id)
                    if len(X) > 0:
                        X_batch_raw.extend(X)
                        y_batch.extend(y)
                except:
                    continue
        
        if len(X_batch_raw) == 0:
            continue
        
        # Extraire features PSD
        X_batch_psd = np.array([extract_band_powers(epoch) for epoch in X_batch_raw])
        y_batch_enc = le.transform(y_batch)
        
        # Scaler et entraîner
        X_batch_scaled = scaler.transform(X_batch_psd)
        clf.partial_fit(X_batch_scaled, y_batch_enc)
        
        batch_score = clf.score(X_batch_scaled, y_batch_enc)
        print(f"   Batch {batch_idx+1}/{n_batches} "
              f"(S{batch_subjects[0]:03}-S{batch_subjects[-1]:03}): "
              f"{len(X_batch_raw)} epochs, score={batch_score:.4f}")
    
    # Étape 3: Évaluation sur test set
    print(f"\n🧪 Évaluation sur test set...")
    test_scores = []
    
    for subject_id in list(test_subject_ids):
        scores_subj = []
        for run_id in run_ids:
            try:
                X_test, y_test = p.get_all_data(subject_id, run_id)
                if len(X_test) > 0 and len(set(y_test)) > 1:
                    # Extraire PSD features
                    X_test_psd = np.array([extract_band_powers(epoch) for epoch in X_test])
                    y_test_enc = le.transform(y_test)
                    X_test_scaled = scaler.transform(X_test_psd)
                    
                    score = clf.score(X_test_scaled, y_test_enc)
                    scores_subj.append(score)
            except:
                continue
        
        if scores_subj:
            mean_score = np.mean(scores_subj)
            test_scores.append(mean_score)
            print(f"   S{subject_id:03}: {mean_score:.4f}")
    
    # Résultat final
    if test_scores:
        final_mean = np.mean(test_scores)
        final_std = np.std(test_scores)
        print(f"\n{'='*70}")
        print(f"🏆 RÉSULTAT FINAL - {task_name.upper()}")
        print(f"{'='*70}")
        print(f"Mean test accuracy: {final_mean:.4f} ± {final_std:.4f}")
        print(f"N = {len(test_scores)} test subjects")
        print(f"{'='*70}\n")
        
        # Sauvegarder
        model_name = f"psd_pipeline_{task_name}.pkl"
        le_name = f"psd_label_encoder_{task_name}.pkl"
        
        model_data = {
            'scaler': scaler,
            'clf': clf,
            'label_encoder': le
        }
        
        joblib.dump(model_data, model_name)
        print(f"💾 Modèle sauvegardé: {model_name}\n")
        
        return final_mean
    else:
        print("❌ Aucun score calculé")
        return None


if __name__ == "__main__":
    print("🔬 Test avec Power Spectral Density features\n")
    print("Ces features sont plus transférables entre sujets que CSP.\n")
    
    # Left vs Right
    score_lr = train_with_psd_features(
        train_subject_ids=range(1, 90),
        test_subject_ids=range(90, 110),
        task_name="left_right"
    )
    
    # Hands vs Feet
    score_hf = train_with_psd_features(
        train_subject_ids=range(1, 90),
        test_subject_ids=range(90, 110),
        task_name="hands_feet"
    )
    
    print("\n" + "="*70)
    print("📊 RÉSUMÉ FINAL")
    print("="*70)
    if score_lr:
        print(f"Left vs Right:  {score_lr:.4f}")
    if score_hf:
        print(f"Hands vs Feet:  {score_hf:.4f}")
    print("="*70)
