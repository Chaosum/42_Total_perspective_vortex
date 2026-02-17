"""
Entraînement incrémental PAR GROUPE DE TÂCHES.

Au lieu de mélanger toutes les tâches, on entraîne un modèle par groupe homogène :
- Groupe "left_right" : Task 1 + Task 2 (left_fist vs right_fist)
- Groupe "hands_feet" : Task 3 + Task 4 (both_fists vs both_feet)

Chaque modèle est binaire et plus facile à apprendre.

Usage:
    python src/train_grouped_incremental.py
"""

import sys
import numpy as np
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from mne.decoding import CSP

from processing import Processing
from global_variable import useful_runs


class PreFittedTransformer:
    """Wrapper pour un transformer déjà fitté, compatible avec pickle."""
    
    def __init__(self, transformer):
        self.transformer = transformer
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return self.transformer.transform(X)
    
    def get_params(self, deep=True):
        return {"transformer": self.transformer}
    
    def set_params(self, **params):
        if "transformer" in params:
            self.transformer = params["transformer"]
        return self


# Définition des groupes de tâches homogènes
TASK_GROUPS = {
    "left_right": {
        "name": "Left vs Right Hands",
        "runs": [3, 4, 7, 8, 11, 12],  # Task 1 (real) + Task 2 (imagery)
        "classes": ["left_fist", "right_fist"],
        "description": "Real + Imagery Left/Right hand movements"
    },
    "hands_feet": {
        "name": "Hands vs Feet",
        "runs": [5, 6, 9, 10, 13, 14],  # Task 3 (real) + Task 4 (imagery)
        "classes": ["both_fists", "both_feet"],
        "description": "Real + Imagery Hands vs Feet movements"
    }
}


def build_incremental_pipeline():
    """Construit un pipeline avec SGDClassifier pour entraînement incrémental."""
    
    csp = CSP(n_components=8, reg=None, log=True, norm_trace=False)
    scaler = StandardScaler()
    
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
    
    return csp, scaler, clf


def sample_subjects_for_fitting(subject_ids, run_ids, n_samples_per_run=30):
    """Charge un échantillon de données pour pré-fitter CSP et Scaler."""
    print("📊 Échantillonnage de données pour pré-fitting CSP et Scaler...")
    
    X_sample = []
    y_sample = []
    p = Processing()
    
    # Prendre quelques sujets bien espacés
    sample_subjects = subject_ids[::len(subject_ids)//10 + 1][:10]
    
    for subject_id in sample_subjects:
        for run_id in run_ids:
            try:
                X_subj_run, y_subj_run = p.get_all_data(subject_id, run_id)
                
                if len(X_subj_run) > 0:
                    if len(X_subj_run) > n_samples_per_run:
                        indices = np.random.choice(len(X_subj_run), n_samples_per_run, replace=False)
                        X_subj_run = [X_subj_run[i] for i in indices]
                        y_subj_run = [y_subj_run[i] for i in indices]
                    
                    X_sample.extend(X_subj_run)
                    y_sample.extend(y_subj_run)
                    
            except Exception as e:
                continue
    
    print(f"✅ Échantillon collecté: {len(X_sample)} epochs de {len(sample_subjects)} sujets")
    
    # Uniformiser les longueurs temporelles
    if len(X_sample) > 0:
        min_length = min(x.shape[1] for x in X_sample)
        X_sample = np.array([x[:, :min_length] for x in X_sample])
    else:
        X_sample = np.array(X_sample)
    
    return X_sample, np.array(y_sample)


def train_group_incremental(
    group_name,
    group_info,
    train_subject_ids=range(1, 90),
    test_subject_ids=range(90, 110),
    n_epochs=5,
    batch_size=5
):
    """
    Entraîne un modèle pour un groupe de tâches spécifique.
    
    Args:
        group_name: Nom du groupe (ex: "left_right")
        group_info: Infos du groupe (runs, classes, etc.)
        train_subject_ids: IDs des sujets pour l'entraînement
        test_subject_ids: IDs des sujets pour le test
        n_epochs: Nombre de passes sur les données
        batch_size: Nombre de sujets par batch
    """
    
    run_ids = group_info["runs"]
    expected_classes = group_info["classes"]
    
    print("\n" + "="*70)
    print(f"🚀 ENTRAÎNEMENT GROUPE: {group_info['name']}")
    print("="*70)
    print(f"Description:  {group_info['description']}")
    print(f"Runs:         {run_ids}")
    print(f"Classes:      {expected_classes}")
    print(f"Sujets train: {len(list(train_subject_ids))}")
    print(f"Sujets test:  {len(list(test_subject_ids))}")
    print(f"Epochs:       {n_epochs}")
    print(f"Batch size:   {batch_size} sujets\n")
    
    # Étape 1: Échantillonner des données pour pré-fitter CSP et Scaler
    X_sample, y_sample = sample_subjects_for_fitting(
        list(train_subject_ids), 
        run_ids, 
        n_samples_per_run=30
    )
    
    # Filtrer pour ne garder que les classes attendues
    mask = np.isin(y_sample, expected_classes)
    X_sample = X_sample[mask]
    y_sample = y_sample[mask]
    
    if len(X_sample) == 0:
        print(f"❌ Aucune donnée pour le groupe {group_name}")
        return None
    
    # Label encoding
    label_encoder = LabelEncoder()
    label_encoder.fit(expected_classes)  # Forcer les classes attendues
    y_sample_encoded = label_encoder.transform(y_sample)
    
    print(f"Classes détectées: {label_encoder.classes_}")
    print(f"Nombre d'epochs d'entraînement: {len(X_sample)}\n")
    
    # Étape 2: Pré-fitter CSP et Scaler
    print("🔧 Pré-fitting CSP et StandardScaler...")
    csp, scaler, clf = build_incremental_pipeline()
    
    X_sample_csp = csp.fit_transform(X_sample, y_sample_encoded)
    X_sample_scaled = scaler.fit_transform(X_sample_csp)
    
    print(f"✅ CSP fitted: {X_sample_csp.shape}")
    print(f"✅ Scaler fitted\n")
    
    # Étape 3: Initialiser le classifieur
    print("🎯 Initialisation du classifieur...")
    clf.partial_fit(X_sample_scaled, y_sample_encoded, classes=np.arange(len(expected_classes)))
    print("✅ Classifieur initialisé\n")
    
    # Étape 4: Entraînement incrémental
    p = Processing()
    train_subjects_list = list(train_subject_ids)
    
    for epoch in range(n_epochs):
        print(f"{'='*70}")
        print(f"📚 EPOCH {epoch + 1}/{n_epochs}")
        print(f"{'='*70}")
        
        np.random.shuffle(train_subjects_list)
        n_batches = (len(train_subjects_list) + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(train_subjects_list))
            batch_subjects = train_subjects_list[batch_start:batch_end]
            
            X_batch = []
            y_batch = []
            
            for subject_id in batch_subjects:
                for run_id in run_ids:
                    try:
                        X_subj_run, y_subj_run = p.get_all_data(subject_id, run_id)
                        if len(X_subj_run) > 0:
                            # Filtrer pour ne garder que les classes du groupe
                            mask = np.isin(y_subj_run, expected_classes)
                            if mask.any():
                                X_filtered = [X_subj_run[i] for i in range(len(X_subj_run)) if mask[i]]
                                y_filtered = [y_subj_run[i] for i in range(len(y_subj_run)) if mask[i]]
                                X_batch.extend(X_filtered)
                                y_batch.extend(y_filtered)
                    except Exception:
                        continue
            
            if len(X_batch) == 0:
                continue
            
            # Uniformiser et transformer
            min_length = min(x.shape[1] for x in X_batch)
            X_batch_uniform = np.array([x[:, :min_length] for x in X_batch])
            
            y_batch = np.array(y_batch)
            y_batch_encoded = label_encoder.transform(y_batch)
            
            X_batch_csp = csp.transform(X_batch_uniform)
            X_batch_scaled = scaler.transform(X_batch_csp)
            
            clf.partial_fit(X_batch_scaled, y_batch_encoded)
            
            batch_score = clf.score(X_batch_scaled, y_batch_encoded)
            
            print(f"  Batch {batch_idx+1}/{n_batches} "
                  f"(sujets {batch_subjects[0]}-{batch_subjects[-1]}): "
                  f"{len(X_batch)} epochs, score={batch_score:.4f}")
        
        # Évaluation sur test set
        print(f"\n🧪 Évaluation sur test set...")
        test_scores = []
        
        for subject_id in list(test_subject_ids)[:10]:
            for run_id in run_ids[:2]:
                try:
                    X_test, y_test = p.get_all_data(subject_id, run_id)
                    if len(X_test) > 0:
                        # Filtrer pour les classes du groupe
                        mask = np.isin(y_test, expected_classes)
                        if mask.any() and len(set(np.array(y_test)[mask])) > 1:
                            X_filtered = [X_test[i] for i in range(len(X_test)) if mask[i]]
                            y_filtered = [y_test[i] for i in range(len(y_test)) if mask[i]]
                            
                            min_length = min(x.shape[1] for x in X_filtered)
                            X_test_uniform = np.array([x[:, :min_length] for x in X_filtered])
                            
                            y_test_encoded = label_encoder.transform(y_filtered)
                            X_test_csp = csp.transform(X_test_uniform)
                            X_test_scaled = scaler.transform(X_test_csp)
                            score = clf.score(X_test_scaled, y_test_encoded)
                            test_scores.append(score)
                except Exception:
                    continue
        
        if test_scores:
            mean_test_score = np.mean(test_scores)
            print(f"📊 Mean test accuracy: {mean_test_score:.4f}\n")
    
    # Étape 5: Sauvegarder le modèle
    print("💾 Sauvegarde du modèle...")
    
    full_pipeline = Pipeline([
        ('csp', PreFittedTransformer(csp)),
        ('scaler', PreFittedTransformer(scaler)),
        ('clf', clf)
    ])
    
    model_filename = f"full_pipeline_{group_name}.pkl"
    encoder_filename = f"label_encoder_{group_name}.pkl"
    
    joblib.dump(full_pipeline, model_filename)
    joblib.dump(label_encoder, encoder_filename)
    
    print(f"✅ Modèle sauvegardé: {model_filename}, {encoder_filename}")
    
    # Évaluation finale complète
    print("\n" + "="*70)
    print(f"🎯 ÉVALUATION FINALE - {group_info['name']}")
    print("="*70)
    
    final_test_scores = []
    for subject_id in test_subject_ids:
        for run_id in run_ids:
            try:
                X_test, y_test = p.get_all_data(subject_id, run_id)
                if len(X_test) > 0:
                    mask = np.isin(y_test, expected_classes)
                    if mask.any() and len(set(np.array(y_test)[mask])) > 1:
                        X_filtered = [X_test[i] for i in range(len(X_test)) if mask[i]]
                        y_filtered = [y_test[i] for i in range(len(y_test)) if mask[i]]
                        
                        min_length = min(x.shape[1] for x in X_filtered)
                        X_test_uniform = np.array([x[:, :min_length] for x in X_filtered])
                        
                        y_test_encoded = label_encoder.transform(y_filtered)
                        X_test_csp = csp.transform(X_test_uniform)
                        X_test_scaled = scaler.transform(X_test_csp)
                        score = clf.score(X_test_scaled, y_test_encoded)
                        final_test_scores.append(score)
            except Exception:
                continue
    
    if final_test_scores:
        final_mean = np.mean(final_test_scores)
        final_std = np.std(final_test_scores)
        print(f"\n🏆 RÉSULTAT FINAL - {group_info['name']}")
        print(f"Mean test accuracy: {final_mean:.4f} ± {final_std:.4f}")
        print(f"N = {len(final_test_scores)} evaluations")
        print("="*70 + "\n")
        
        return final_mean
    else:
        print(f"❌ Aucun score de test pour {group_name}")
        return None


def main():
    """Entraîne tous les groupes de tâches."""
    
    results = {}
    
    for group_name, group_info in TASK_GROUPS.items():
        result = train_group_incremental(
            group_name=group_name,
            group_info=group_info,
            train_subject_ids=range(1, 90),
            test_subject_ids=range(90, 110),
            n_epochs=5,
            batch_size=5
        )
        
        if result is not None:
            results[group_name] = result
    
    # Résumé final
    print("\n" + "="*70)
    print("📊 RÉSUMÉ GLOBAL")
    print("="*70)
    
    for group_name, accuracy in results.items():
        group_info = TASK_GROUPS[group_name]
        print(f"{group_info['name']:25s}: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    if results:
        mean_accuracy = np.mean(list(results.values()))
        print(f"\n{'Mean accuracy (all groups)':25s}: {mean_accuracy:.4f} ({mean_accuracy*100:.1f}%)")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()
