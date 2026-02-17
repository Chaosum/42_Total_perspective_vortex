"""
Test simple pour vérifier que l'entraînement fonctionne.
On va entraîner un petit modèle sur quelques sujets et tester sur d'autres.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mne.decoding import CSP
from processing import Processing

def simple_train_test():
    """Test simple: train sur 3 sujets, test sur 2 autres."""
    
    print("="*70)
    print("🧪 TEST SIMPLE: LEFT vs RIGHT")
    print("="*70)
    
    p = Processing()
    
    # Runs pour left/right
    runs = [3, 4]  # Un real, un imagery
    
    # TRAIN: sujets 1, 2, 3
    print("\n📚 Chargement données d'entraînement...")
    X_train_all = []
    y_train_all = []
    
    for subject_id in [1, 2, 3]:
        for run_id in runs:
            X, y = p.get_all_data(subject_id, run_id)
            if len(X) > 0:
                X_train_all.extend(X)
                y_train_all.extend(y)
    
    # Uniformiser
    min_length = min(x.shape[1] for x in X_train_all)
    X_train = np.array([x[:, :min_length] for x in X_train_all])
    y_train = np.array(y_train_all)
    
    print(f"Shape train: {X_train.shape}")
    print(f"Labels train: {np.unique(y_train, return_counts=True)}")
    
    # Encoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    print(f"Classes: {le.classes_}")
    
    # CSP + Scaler + LogReg
    print("\n🔧 Entraînement du modèle...")
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    X_train_csp = csp.fit_transform(X_train, y_train_enc)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_csp)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train_enc)
    
    train_score = clf.score(X_train_scaled, y_train_enc)
    print(f"✅ Score train: {train_score:.4f}")
    
    # TEST: sujets 90, 91
    print("\n🧪 Test sur sujets jamais vus (90, 91)...")
    X_test_all = []
    y_test_all = []
    
    for subject_id in [90, 91]:
        for run_id in runs:
            try:
                X, y = p.get_all_data(subject_id, run_id)
                if len(X) > 0:
                    X_test_all.extend(X)
                    y_test_all.extend(y)
            except Exception as e:
                print(f"   ⚠️  S{subject_id} run {run_id}: {e}")
    
    if len(X_test_all) > 0:
        # Uniformiser
        min_length = min(x.shape[1] for x in X_test_all)
        X_test = np.array([x[:, :min_length] for x in X_test_all])
        y_test = np.array(y_test_all)
        
        print(f"Shape test: {X_test.shape}")
        print(f"Labels test: {np.unique(y_test, return_counts=True)}")
        
        # Transformer
        y_test_enc = le.transform(y_test)
        X_test_csp = csp.transform(X_test)
        X_test_scaled = scaler.transform(X_test_csp)
        
        test_score = clf.score(X_test_scaled, y_test_enc)
        print(f"📊 Score test: {test_score:.4f}")
        
        # Vérifier les prédictions
        y_pred = clf.predict(X_test_scaled)
        print(f"\nDistribution prédictions: {np.unique(y_pred, return_counts=True)}")
        print(f"Distribution vraies: {np.unique(y_test_enc, return_counts=True)}")
        
        # Vérifier si le modèle prédit toujours la même classe
        if len(np.unique(y_pred)) == 1:
            print("⚠️  PROBLÈME: Le modèle prédit toujours la même classe!")
            print(f"   Classe prédite: {le.inverse_transform([y_pred[0]])[0]}")
    else:
        print("❌ Pas de données de test")


def test_hands_feet():
    """Test pour hands vs feet."""
    
    print("\n\n" + "="*70)
    print("🧪 TEST SIMPLE: HANDS vs FEET")
    print("="*70)
    
    p = Processing()
    
    # Runs pour hands/feet
    runs = [5, 6]  # Un real, un imagery
    
    # TRAIN: sujets 1, 2, 3
    print("\n📚 Chargement données d'entraînement...")
    X_train_all = []
    y_train_all = []
    
    for subject_id in [1, 2, 3]:
        for run_id in runs:
            X, y = p.get_all_data(subject_id, run_id)
            if len(X) > 0:
                X_train_all.extend(X)
                y_train_all.extend(y)
    
    # Uniformiser
    min_length = min(x.shape[1] for x in X_train_all)
    X_train = np.array([x[:, :min_length] for x in X_train_all])
    y_train = np.array(y_train_all)
    
    print(f"Shape train: {X_train.shape}")
    print(f"Labels train: {np.unique(y_train, return_counts=True)}")
    
    # Encoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    print(f"Classes: {le.classes_}")
    
    # CSP + Scaler + LogReg
    print("\n🔧 Entraînement du modèle...")
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    X_train_csp = csp.fit_transform(X_train, y_train_enc)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_csp)
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train_enc)
    
    train_score = clf.score(X_train_scaled, y_train_enc)
    print(f"✅ Score train: {train_score:.4f}")
    
    # TEST: sujets 90, 91
    print("\n🧪 Test sur sujets jamais vus (90, 91)...")
    X_test_all = []
    y_test_all = []
    
    for subject_id in [90, 91]:
        for run_id in runs:
            try:
                X, y = p.get_all_data(subject_id, run_id)
                if len(X) > 0:
                    X_test_all.extend(X)
                    y_test_all.extend(y)
            except Exception as e:
                print(f"   ⚠️  S{subject_id} run {run_id}: {e}")
    
    if len(X_test_all) > 0:
        # Uniformiser
        min_length = min(x.shape[1] for x in X_test_all)
        X_test = np.array([x[:, :min_length] for x in X_test_all])
        y_test = np.array(y_test_all)
        
        print(f"Shape test: {X_test.shape}")
        print(f"Labels test: {np.unique(y_test, return_counts=True)}")
        
        # Transformer
        y_test_enc = le.transform(y_test)
        X_test_csp = csp.transform(X_test)
        X_test_scaled = scaler.transform(X_test_csp)
        
        test_score = clf.score(X_test_scaled, y_test_enc)
        print(f"📊 Score test: {test_score:.4f}")
        
        # Vérifier les prédictions
        y_pred = clf.predict(X_test_scaled)
        print(f"\nDistribution prédictions: {np.unique(y_pred, return_counts=True)}")
        print(f"Distribution vraies: {np.unique(y_test_enc, return_counts=True)}")
        
        if len(np.unique(y_pred)) == 1:
            print("⚠️  PROBLÈME: Le modèle prédit toujours la même classe!")
            print(f"   Classe prédite: {le.inverse_transform([y_pred[0]])[0]}")
    else:
        print("❌ Pas de données de test")


if __name__ == "__main__":
    simple_train_test()
    test_hands_feet()
    
    print("\n" + "="*70)
    print("🏁 FIN DES TESTS")
    print("="*70)
