"""
Test avec différentes configurations pour améliorer la généralisation.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from mne.decoding import CSP
from processing import Processing

def test_with_more_training_subjects():
    """Tester avec plus de sujets d'entraînement."""
    
    print("="*70)
    print("🧪 TEST: Plus de sujets d'entraînement (LEFT vs RIGHT)")
    print("="*70)
    
    p = Processing()
    runs = [3, 4, 7, 8]  # Plus de runs
    
    # TRAIN: sujets 1-10
    print(f"\n📚 Chargement données d'entraînement (sujets 1-10, runs {runs})...")
    X_train_all = []
    y_train_all = []
    
    for subject_id in range(1, 11):
        for run_id in runs:
            try:
                X, y = p.get_all_data(subject_id, run_id)
                if len(X) > 0:
                    X_train_all.extend(X)
                    y_train_all.extend(y)
            except:
                continue
    
    # Uniformiser
    min_length = min(x.shape[1] for x in X_train_all)
    X_train = np.array([x[:, :min_length] for x in X_train_all])
    y_train = np.array(y_train_all)
    
    print(f"Shape train: {X_train.shape}")
    print(f"Labels train: {np.unique(y_train, return_counts=True)}")
    
    # Encoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    
    # CSP avec régularisation
    print("\n🔧 Entraînement du modèle (CSP avec régularisation)...")
    csp = CSP(n_components=6, reg=0.1, log=True, norm_trace=False, cov_est='concat')
    X_train_csp = csp.fit_transform(X_train, y_train_enc)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_csp)
    
    # Logistic avec régularisation plus forte
    clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train_enc)
    
    train_score = clf.score(X_train_scaled, y_train_enc)
    print(f"✅ Score train: {train_score:.4f}")
    
    # TEST: sujets 90, 91, 92
    print("\n🧪 Test sur sujets jamais vus (90-92)...")
    X_test_all = []
    y_test_all = []
    
    for subject_id in [90, 91, 92]:
        for run_id in runs:
            try:
                X, y = p.get_all_data(subject_id, run_id)
                if len(X) > 0:
                    X_test_all.extend(X)
                    y_test_all.extend(y)
            except:
                continue
    
    if len(X_test_all) > 0:
        min_length = min(x.shape[1] for x in X_test_all)
        X_test = np.array([x[:, :min_length] for x in X_test_all])
        y_test = np.array(y_test_all)
        
        print(f"Shape test: {X_test.shape}")
        print(f"Labels test: {np.unique(y_test, return_counts=True)}")
        
        y_test_enc = le.transform(y_test)
        X_test_csp = csp.transform(X_test)
        X_test_scaled = scaler.transform(X_test_csp)
        
        test_score = clf.score(X_test_scaled, y_test_enc)
        print(f"📊 Score test: {test_score:.4f}")
        
        y_pred = clf.predict(X_test_scaled)
        print(f"Distribution prédictions: {np.unique(y_pred, return_counts=True)}")
        print(f"Distribution vraies: {np.unique(y_test_enc, return_counts=True)}")
        
        # Vérifier les probabilités de prédiction
        y_proba = clf.predict_proba(X_test_scaled)
        mean_proba = y_proba.mean(axis=0)
        print(f"Probabilités moyennes: {mean_proba}")
        
        if len(np.unique(y_pred)) == 1:
            print("⚠️  Modèle prédit toujours la même classe")
        else:
            print("✅ Modèle prédit les deux classes")


def test_subject_independent_approach():
    """
    Approche subject-independent: tester avec validation croisée
    sur les sujets d'entraînement pour voir si le problème persiste.
    """
    
    print("\n\n" + "="*70)
    print("🧪 TEST: Validation croisée inter-sujets (LEFT vs RIGHT)")
    print("="*70)
    
    p = Processing()
    runs = [3, 4]
    
    # Charger plusieurs sujets
    subjects = list(range(1, 11))
    
    print(f"\n📚 Chargement de {len(subjects)} sujets...")
    subject_data = {}
    
    for subject_id in subjects:
        X_subj = []
        y_subj = []
        for run_id in runs:
            try:
                X, y = p.get_all_data(subject_id, run_id)
                if len(X) > 0:
                    X_subj.extend(X)
                    y_subj.extend(y)
            except:
                continue
        
        if len(X_subj) > 0:
            min_length = min(x.shape[1] for x in X_subj)
            subject_data[subject_id] = (
                np.array([x[:, :min_length] for x in X_subj]),
                np.array(y_subj)
            )
    
    print(f"✅ {len(subject_data)} sujets chargés")
    
    # Leave-one-subject-out cross-validation
    print("\n🔧 Leave-One-Subject-Out validation...")
    
    scores = []
    for test_subj_id in list(subject_data.keys())[:5]:  # Tester sur 5 sujets
        # Train sur tous sauf test_subj_id
        X_train_all = []
        y_train_all = []
        
        for subj_id, (X, y) in subject_data.items():
            if subj_id != test_subj_id:
                X_train_all.extend(X)
                y_train_all.extend(y)
        
        X_train = np.array(X_train_all)
        y_train = np.array(y_train_all)
        
        # Test sur test_subj_id
        X_test, y_test = subject_data[test_subj_id]
        
        # Encoder
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        
        # CSP + Scaler + Classifier
        csp = CSP(n_components=6, reg=0.1, log=True, norm_trace=False)
        X_train_csp = csp.fit_transform(X_train, y_train_enc)
        X_test_csp = csp.transform(X_test)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_csp)
        X_test_scaled = scaler.transform(X_test_csp)
        
        clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train_enc)
        
        score = clf.score(X_test_scaled, y_test_enc)
        scores.append(score)
        
        y_pred = clf.predict(X_test_scaled)
        n_classes_pred = len(np.unique(y_pred))
        
        print(f"   Sujet {test_subj_id:2d}: score={score:.3f}, classes prédites={n_classes_pred}")
    
    print(f"\n📊 Mean score: {np.mean(scores):.4f} ± {np.std(scores):.4f}")


if __name__ == "__main__":
    test_with_more_training_subjects()
    test_subject_independent_approach()
    
    print("\n" + "="*70)
    print("🏁 FIN DES TESTS")
    print("="*70)
