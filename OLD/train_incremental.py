"""
Entraînement incrémental pour éviter de saturer la mémoire.

Au lieu de charger tous les sujets × runs d'un coup, on :
1. Pré-fit CSP + Scaler sur un échantillon représentatif
2. Entraîne le classifieur de façon incrémentale avec partial_fit()
3. Fait plusieurs passes (epochs) sur les données

Usage:
    python src/train_incremental.py
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


def optimize_csp_components(X_calib, y_calib, n_splits=3):
    """
    Trouve le nombre optimal de composantes CSP via validation croisée.
    
    Teste n_components = 4, 6, 8 et retourne le meilleur.
    Cette optimisation peut améliorer de 1-2% l'accuracy en s'adaptant
    aux patterns cérébraux spécifiques de chaque sujet.
    
    Args:
        X_calib: Données de calibration (n_trials, n_channels, n_times)
        y_calib: Labels de calibration
        n_splits: Nombre de folds pour la validation croisée
    
    Returns:
        best_n_components: Nombre optimal de composantes (4, 6, ou 8)
        best_score: Score CV avec les composantes optimales
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    
    # Vérifier qu'on a assez d'échantillons pour le CV
    n_samples_per_class = np.bincount(y_calib)
    if len(n_samples_per_class) < 2 or min(n_samples_per_class) < n_splits:
        # Pas assez d'échantillons, utiliser valeur par défaut
        return 6, 0.0
    
    best_n = 6
    best_score = 0
    
    # Tester différents nombres de composantes
    for n_comp in [4, 6, 8]:
        if n_comp > X_calib.shape[1]:  # Plus de composantes que de canaux
            continue
        
        scores = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        try:
            for train_idx, val_idx in skf.split(X_calib, y_calib):
                X_train, X_val = X_calib[train_idx], X_calib[val_idx]
                y_train, y_val = y_calib[train_idx], y_calib[val_idx]
                
                # CSP
                csp = CSP(n_components=n_comp, reg=0.1, log=True, norm_trace=False)
                X_train_csp = csp.fit_transform(X_train, y_train)
                X_val_csp = csp.transform(X_val)
                
                # Scaler
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_csp)
                X_val_scaled = scaler.transform(X_val_csp)
                
                # Classifier - LDA est meilleur que LogReg pour BCI
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                clf.fit(X_train_scaled, y_train)
                scores.append(clf.score(X_val_scaled, y_val))
            
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_n = n_comp
        except:
            continue
    
    return best_n, best_score


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


def build_incremental_pipeline():
    """Construit un pipeline avec SGDClassifier pour entraînement incrémental."""
    
    csp = CSP(n_components=8, reg=None, log=True, norm_trace=False)
    scaler = StandardScaler()
    
    # SGDClassifier supporte partial_fit pour l'apprentissage incrémental
    clf = SGDClassifier(
        loss='log_loss',  # équivalent logistic regression
        penalty='l2',
        alpha=0.0001,
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        learning_rate='optimal',
        early_stopping=False
    )
    
    # Note: On ne peut pas utiliser Pipeline.partial_fit directement,
    # donc on va gérer manuellement les étapes
    return csp, scaler, clf


def sample_subjects_for_fitting(subject_ids, run_ids, n_samples_per_run=50):
    """
    Charge un échantillon de données pour pré-fitter CSP et Scaler.
    
    On prend quelques epochs de plusieurs sujets pour avoir une bonne
    représentation de la variance globale.
    """
    print("📊 Échantillonnage de données pour pré-fitting CSP et Scaler...")
    
    X_sample = []
    y_sample = []
    p = Processing()
    
    # Prendre quelques sujets bien espacés
    sample_subjects = subject_ids[::len(subject_ids)//10 + 1][:10]  # ~10 sujets
    
    for subject_id in sample_subjects:
        for run_id in run_ids:
            try:
                X_subj_run, y_subj_run = p.get_all_data(subject_id, run_id)
                
                if len(X_subj_run) > 0:
                    # Prendre un échantillon si trop d'epochs
                    if len(X_subj_run) > n_samples_per_run:
                        indices = np.random.choice(len(X_subj_run), n_samples_per_run, replace=False)
                        X_subj_run = [X_subj_run[i] for i in indices]
                        y_subj_run = [y_subj_run[i] for i in indices]
                    
                    X_sample.extend(X_subj_run)
                    y_sample.extend(y_subj_run)
                    
            except Exception as e:
                print(f"⚠️  Erreur lors du chargement S{subject_id:03} run {run_id}: {e}")
                continue
    
    print(f"✅ Échantillon collecté: {len(X_sample)} epochs de {len(sample_subjects)} sujets")
    
    # Uniformiser les longueurs temporelles
    if len(X_sample) > 0:
        min_length = min(x.shape[1] for x in X_sample)
        X_sample = np.array([x[:, :min_length] for x in X_sample])
    else:
        X_sample = np.array(X_sample)
    
    return X_sample, np.array(y_sample)


def train_incremental(
    train_subject_ids=range(1, 90),
    test_subject_ids=range(90, 110),
    run_ids=None,
    n_epochs=3,
    batch_size=5
):
    """
    Entraînement incrémental sur plusieurs sujets.
    
    Args:
        train_subject_ids: IDs des sujets pour l'entraînement
        test_subject_ids: IDs des sujets pour le test (never-learned)
        run_ids: Liste des runs à utiliser (None = tous les useful_runs)
        n_epochs: Nombre de passes sur les données d'entraînement
        batch_size: Nombre de sujets à traiter ensemble dans partial_fit
    """
    
    if run_ids is None:
        run_ids = list(useful_runs.keys())
    
    print("="*70)
    print("🚀 ENTRAÎNEMENT INCRÉMENTAL")
    print("="*70)
    print(f"Sujets train: {len(list(train_subject_ids))} (ex: {list(train_subject_ids)[:3]}...)")
    print(f"Sujets test:  {len(list(test_subject_ids))} (ex: {list(test_subject_ids)[:3]}...)")
    print(f"Runs:         {run_ids}")
    print(f"Epochs:       {n_epochs}")
    print(f"Batch size:   {batch_size} sujets\n")
    
    # Étape 1: Échantillonner des données pour pré-fitter CSP et Scaler
    X_sample, y_sample = sample_subjects_for_fitting(
        list(train_subject_ids), 
        run_ids, 
        n_samples_per_run=30
    )
    
    # Label encoding
    label_encoder = LabelEncoder()
    y_sample_encoded = label_encoder.fit_transform(y_sample)
    classes = label_encoder.classes_
    
    print(f"Classes détectées: {classes}\n")
    
    # Étape 2: Pré-fitter CSP et Scaler
    print("🔧 Pré-fitting CSP et StandardScaler...")
    csp, scaler, clf = build_incremental_pipeline()
    
    X_sample_csp = csp.fit_transform(X_sample, y_sample_encoded)
    X_sample_scaled = scaler.fit_transform(X_sample_csp)
    
    print(f"✅ CSP fitted: {X_sample_csp.shape}")
    print(f"✅ Scaler fitted\n")
    
    # Étape 3: Initialiser le classifieur avec le premier batch
    print("🎯 Initialisation du classifieur...")
    clf.partial_fit(X_sample_scaled, y_sample_encoded, classes=np.arange(len(classes)))
    print("✅ Classifieur initialisé\n")
    
    # Étape 4: Entraînement incrémental par epochs
    p = Processing()
    train_subjects_list = list(train_subject_ids)
    
    for epoch in range(n_epochs):
        print(f"{'='*70}")
        print(f"📚 EPOCH {epoch + 1}/{n_epochs}")
        print(f"{'='*70}")
        
        # Mélanger l'ordre des sujets à chaque epoch
        np.random.shuffle(train_subjects_list)
        
        n_batches = (len(train_subjects_list) + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(train_subjects_list))
            batch_subjects = train_subjects_list[batch_start:batch_end]
            
            # Charger les données du batch
            X_batch = []
            y_batch = []
            
            for subject_id in batch_subjects:
                for run_id in run_ids:
                    try:
                        X_subj_run, y_subj_run = p.get_all_data(subject_id, run_id)
                        if len(X_subj_run) > 0:
                            X_batch.extend(X_subj_run)
                            y_batch.extend(y_subj_run)
                    except Exception as e:
                        continue
            
            if len(X_batch) == 0:
                continue
            
            # Transformer et entraîner
            # Problème: les epochs peuvent avoir des longueurs temporelles différentes
            # Solution: trouver la longueur minimale et tronquer tous les epochs
            min_length = min(x.shape[1] for x in X_batch)
            X_batch_uniform = np.array([x[:, :min_length] for x in X_batch])
            
            y_batch = np.array(y_batch)
            y_batch_encoded = label_encoder.transform(y_batch)
            
            X_batch_csp = csp.transform(X_batch_uniform)
            X_batch_scaled = scaler.transform(X_batch_csp)
            
            clf.partial_fit(X_batch_scaled, y_batch_encoded)
            
            # Score sur le batch pour monitoring
            batch_score = clf.score(X_batch_scaled, y_batch_encoded)
            
            print(f"  Batch {batch_idx+1}/{n_batches} "
                  f"(sujets {batch_subjects[0]}-{batch_subjects[-1]}): "
                  f"{len(X_batch)} epochs, score={batch_score:.4f}")
        
        # Évaluation sur test set après chaque epoch
        print(f"\n🧪 Évaluation sur test set (sujets {list(test_subject_ids)[0]}-{list(test_subject_ids)[-1]})...")
        test_scores = []
        
        for subject_id in list(test_subject_ids)[:10]:  # Tester sur 10 sujets pour vitesse
            for run_id in run_ids[:2]:  # Tester sur 2 runs pour vitesse
                try:
                    X_test, y_test = p.get_all_data(subject_id, run_id)
                    if len(X_test) > 0 and len(set(y_test)) > 1:
                        # Uniformiser les longueurs
                        min_length = min(x.shape[1] for x in X_test)
                        X_test_uniform = np.array([x[:, :min_length] for x in X_test])
                        
                        y_test_encoded = label_encoder.transform(y_test)
                        X_test_csp = csp.transform(X_test_uniform)
                        X_test_scaled = scaler.transform(X_test_csp)
                        score = clf.score(X_test_scaled, y_test_encoded)
                        test_scores.append(score)
                except Exception:
                    continue
        
        if test_scores:
            mean_test_score = np.mean(test_scores)
            print(f"📊 Mean test accuracy: {mean_test_score:.4f}\n")
    
    # Étape 5: Sauvegarder le modèle complet
    print("💾 Sauvegarde du modèle...")
    
    # Créer un pipeline sklearn complet pour la compatibilité
    full_pipeline = Pipeline([
        ('csp', PreFittedTransformer(csp)),
        ('scaler', PreFittedTransformer(scaler)),
        ('clf', clf)
    ])
    
    joblib.dump(full_pipeline, "full_pipeline.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    
    print("✅ Modèle sauvegardé: full_pipeline.pkl, label_encoder.pkl")
    
    # Évaluation finale complète sur test set
    print("\n" + "="*70)
    print("🎯 ÉVALUATION FINALE sur test set complet")
    print("="*70)
    
    final_test_scores = []
    for subject_id in test_subject_ids:
        for run_id in run_ids:
            try:
                X_test, y_test = p.get_all_data(subject_id, run_id)
                if len(X_test) > 0 and len(set(y_test)) > 1:
                    # Uniformiser les longueurs
                    min_length = min(x.shape[1] for x in X_test)
                    X_test_uniform = np.array([x[:, :min_length] for x in X_test])
                    
                    y_test_encoded = label_encoder.transform(y_test)
                    X_test_csp = csp.transform(X_test_uniform)
                    X_test_scaled = scaler.transform(X_test_csp)
                    score = clf.score(X_test_scaled, y_test_encoded)
                    final_test_scores.append(score)
                    print(f"S{subject_id:03} run {run_id}: {score:.4f}")
            except Exception as e:
                continue
    
    if final_test_scores:
        final_mean = np.mean(final_test_scores)
        final_std = np.std(final_test_scores)
        print(f"\n{'='*70}")
        print(f"🏆 RÉSULTAT FINAL")
        print(f"{'='*70}")
        print(f"Mean test accuracy: {final_mean:.4f} ± {final_std:.4f}")
        print(f"N = {len(final_test_scores)} evaluations")
        print(f"{'='*70}\n")
        
        return final_mean
    else:
        print("❌ Aucun score de test calculé")
        return None


def train_subject_specific(
    subject_ids=range(1, 110),
    calibration_runs=None,
    test_runs=None,
    task_name="left_right"
):
    """
    Entraînement subject-specific: un modèle par sujet.
    
    Stratégie: Pour chaque sujet, utiliser certaines runs pour calibration
    et d'autres runs pour test. Ceci simule un scénario réaliste où un
    utilisateur fait une session de calibration puis utilise le BCI.
    
    Args:
        subject_ids: IDs des sujets à traiter
        calibration_runs: Runs pour la calibration (ex: [3, 7, 11] = real)
        test_runs: Runs pour le test (ex: [4, 8, 12] = imagery)
        task_name: "left_right" ou "hands_feet"
    
    Returns:
        dict: Résultats par sujet et statistiques globales
    """
    
    # Configuration par défaut selon la tâche
    if calibration_runs is None or test_runs is None:
        if task_name == "left_right":
            calibration_runs = [3, 7, 11]  # Mouvement réel
            test_runs = [4, 8, 12]         # Mouvement imaginé
        else:  # hands_feet
            calibration_runs = [5, 9, 13]  # Mouvement réel
            test_runs = [6, 10, 14]        # Mouvement imaginé
    
    print("="*70)
    print("🚀 ENTRAÎNEMENT SUBJECT-SPECIFIC")
    print("="*70)
    print(f"Tâche: {task_name.upper()}")
    print(f"Sujets: {len(list(subject_ids))}")
    print(f"Calibration runs: {calibration_runs} (mouvement réel)")
    print(f"Test runs: {test_runs} (mouvement imaginé)")
    print("="*70)
    print("\nStratégie: Entraîner un modèle spécifique pour chaque sujet")
    print("avec ses propres données de calibration.\n")
    
    p = Processing()
    results = []
    failed_subjects = []
    
    for subject_id in subject_ids:
        print(f"\n{'='*70}")
        print(f"👤 SUJET {subject_id:03}")
        print(f"{'='*70}")
        
        # CALIBRATION: Charger les runs de calibration
        X_calib_all = []
        y_calib_all = []
        
        print(f"📚 Calibration (runs {calibration_runs})...")
        for run_id in calibration_runs:
            try:
                X, y = p.get_all_data(subject_id, run_id)
                if len(X) > 0:
                    X_calib_all.extend(X)
                    y_calib_all.extend(y)
                    print(f"   Run {run_id:2d}: {len(X)} epochs")
            except Exception as e:
                print(f"   Run {run_id:2d}: ⚠️  {e}")
                continue
        
        # TEST: Charger les runs de test
        X_test_all = []
        y_test_all = []
        
        print(f"\n🧪 Test (runs {test_runs})...")
        for run_id in test_runs:
            try:
                X, y = p.get_all_data(subject_id, run_id)
                if len(X) > 0:
                    X_test_all.extend(X)
                    y_test_all.extend(y)
                    print(f"   Run {run_id:2d}: {len(X)} epochs")
            except Exception as e:
                print(f"   Run {run_id:2d}: ⚠️  {e}")
                continue
        
        # Vérifier qu'on a assez de données
        if len(X_calib_all) < 10 or len(X_test_all) < 5:
            print(f"\n❌ Pas assez de données (calib={len(X_calib_all)}, test={len(X_test_all)})")
            failed_subjects.append(subject_id)
            continue
        
        # Uniformiser les longueurs temporelles
        min_length_calib = min(x.shape[1] for x in X_calib_all)
        X_calib = np.array([x[:, :min_length_calib] for x in X_calib_all])
        y_calib = np.array(y_calib_all)
        
        min_length_test = min(x.shape[1] for x in X_test_all)
        X_test = np.array([x[:, :min_length_test] for x in X_test_all])
        y_test = np.array(y_test_all)
        
        print(f"\n📊 Données préparées:")
        print(f"   Calibration: {X_calib.shape}")
        print(f"   Test: {X_test.shape}")
        
        # Vérifier qu'on a les deux classes
        if len(set(y_calib)) < 2 or len(set(y_test)) < 2:
            print(f"\n❌ Une seule classe présente")
            failed_subjects.append(subject_id)
            continue
        
        # Encoder les labels
        le = LabelEncoder()
        y_calib_enc = le.fit_transform(y_calib)
        y_test_enc = le.transform(y_test)
        
        print(f"   Classes: {le.classes_}")
        print(f"   Distribution calibration: {dict(zip(*np.unique(y_calib, return_counts=True)))}")
        print(f"   Distribution test: {dict(zip(*np.unique(y_test, return_counts=True)))}")
        
        # Entraîner le modèle spécifique à ce sujet
        print(f"\n🔧 Entraînement du modèle subject-specific...")
        
        # Optimiser le nombre de composantes CSP pour ce sujet
        print(f"   Optimisation CSP (test n_components = 4, 6, 8)...")
        optimal_n, cv_score = optimize_csp_components(X_calib, y_calib_enc, n_splits=3)
        print(f"   ✅ Optimal: n_components={optimal_n} (CV score={cv_score:.4f})")
        
        # CSP spécifique au sujet avec nombre optimal de composantes
        csp = CSP(n_components=optimal_n, reg=0.1, log=True, norm_trace=False)
        X_calib_csp = csp.fit_transform(X_calib, y_calib_enc)
        X_test_csp = csp.transform(X_test)
        
        # Scaler spécifique au sujet
        scaler = StandardScaler()
        X_calib_scaled = scaler.fit_transform(X_calib_csp)
        X_test_scaled = scaler.transform(X_test_csp)
        
        # Classifier - LDA (standard BCI) au lieu de LogisticRegression
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf.fit(X_calib_scaled, y_calib_enc)
        
        # Évaluation
        train_score = clf.score(X_calib_scaled, y_calib_enc)
        test_score = clf.score(X_test_scaled, y_test_enc)
        
        # Vérifier les prédictions
        y_pred = clf.predict(X_test_scaled)
        n_classes_pred = len(np.unique(y_pred))
        
        print(f"\n✅ Résultats:")
        print(f"   Score calibration: {train_score:.4f}")
        print(f"   Score test: {test_score:.4f}")
        print(f"   Classes prédites: {n_classes_pred}/2")
        
        # Sauvegarder le modèle du sujet
        model_dir = f"models/subject_specific/{task_name}"
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        subject_model = {
            'csp': csp,
            'scaler': scaler,
            'clf': clf,
            'label_encoder': le,
            'subject_id': subject_id,
            'calibration_runs': calibration_runs,
            'test_runs': test_runs,
            'train_score': train_score,
            'test_score': test_score,
            'n_components': optimal_n,
            'cv_score': cv_score
        }
        
        model_path = os.path.join(model_dir, f"subject_{subject_id:03}.pkl")
        joblib.dump(subject_model, model_path)
        
        results.append({
            'subject_id': subject_id,
            'train_score': train_score,
            'test_score': test_score,
            'n_classes_pred': n_classes_pred,
            'n_calib': len(X_calib),
            'n_test': len(X_test),
            'n_components': optimal_n,
            'cv_score': cv_score
        })
    
    # Statistiques globales
    print("\n\n" + "="*70)
    print("📊 RÉSULTATS GLOBAUX - SUBJECT-SPECIFIC")
    print("="*70)
    print(f"Tâche: {task_name.upper()}\n")
    
    if results:
        test_scores = [r['test_score'] for r in results]
        train_scores = [r['train_score'] for r in results]
        
        print(f"Sujets réussis: {len(results)}/{len(list(subject_ids))}")
        print(f"Sujets échoués: {len(failed_subjects)}")
        
        print(f"\n📈 Scores de calibration:")
        print(f"   Mean: {np.mean(train_scores):.4f} ± {np.std(train_scores):.4f}")
        print(f"   Min: {np.min(train_scores):.4f}")
        print(f"   Max: {np.max(train_scores):.4f}")
        
        print(f"\n🎯 Scores de test:")
        print(f"   Mean: {np.mean(test_scores):.4f} ± {np.std(test_scores):.4f}")
        print(f"   Min: {np.min(test_scores):.4f}")
        print(f"   Max: {np.max(test_scores):.4f}")
        
        # Histogramme des performances
        print(f"\n📊 Distribution des scores de test:")
        bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i in range(len(bins)-1):
            count = sum(1 for s in test_scores if bins[i] <= s < bins[i+1])
            pct = 100 * count / len(test_scores)
            bar = "█" * int(pct / 2)
            print(f"   {bins[i]:.1f}-{bins[i+1]:.1f}: {count:3d} sujets ({pct:5.1f}%) {bar}")
        
        # Distribution des composantes CSP optimales
        n_components_list = [r['n_components'] for r in results]
        print(f"\n🔧 Distribution des composantes CSP optimales:")
        for nc in [4, 6, 8]:
            count = sum(1 for n in n_components_list if n == nc)
            pct = 100 * count / len(n_components_list)
            bar = "█" * int(pct / 3)
            print(f"   {nc} composantes: {count:3d} sujets ({pct:5.1f}%) {bar}")
        
        print(f"\n💾 Modèles sauvegardés dans: models/subject_specific/{task_name}/")
        print("="*70)
        
        return {
            'results': results,
            'failed_subjects': failed_subjects,
            'mean_test_score': np.mean(test_scores),
            'std_test_score': np.std(test_scores),
            'mean_train_score': np.mean(train_scores)
        }
    else:
        print("❌ Aucun sujet n'a pu être traité avec succès")
        return None


if __name__ == "__main__":
    # Configuration par défaut
    # Train sur sujets 1-89, test sur 90-109 (never-learned)
    train_incremental(
        train_subject_ids=range(1, 90),
        test_subject_ids=range(90, 110),
        run_ids=None,  # tous les runs de useful_runs
        n_epochs=3,
        batch_size=5
    )
