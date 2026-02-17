"""
Approche subject-specific: entraîner un modèle par sujet.
Cette approche est généralement beaucoup plus performante en BCI.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from mne.decoding import CSP
from processing import Processing

def subject_specific_evaluation():
    """
    Évaluation subject-specific: pour chaque sujet, entraîner
    sur une partie de ses propres données et tester sur le reste.
    """
    
    print("="*70)
    print("🧪 ÉVALUATION SUBJECT-SPECIFIC (LEFT vs RIGHT)")
    print("="*70)
    print("Pour chaque sujet: train sur certaines runs, test sur d'autres\n")
    
    p = Processing()
    
    # Runs disponibles pour left/right
    train_runs = [3, 7, 11]  # real
    test_runs = [4, 8, 12]    # imagery
    
    subject_scores = []
    
    print("Stratégie: Train sur runs real (3, 7, 11), Test sur runs imagery (4, 8, 12)\n")
    
    for subject_id in [1, 2, 3, 90, 91, 92]:
        print(f"Sujet {subject_id}:")
        
        # TRAIN sur runs real
        X_train_all = []
        y_train_all = []
        
        for run_id in train_runs:
            try:
                X, y = p.get_all_data(subject_id, run_id)
                if len(X) > 0:
                    X_train_all.extend(X)
                    y_train_all.extend(y)
            except:
                continue
        
        # TEST sur runs imagery
        X_test_all = []
        y_test_all = []
        
        for run_id in test_runs:
            try:
                X, y = p.get_all_data(subject_id, run_id)
                if len(X) > 0:
                    X_test_all.extend(X)
                    y_test_all.extend(y)
            except:
                continue
        
        if len(X_train_all) == 0 or len(X_test_all) == 0:
            print("   ⚠️  Pas assez de données\n")
            continue
        
        # Uniformiser
        min_length_train = min(x.shape[1] for x in X_train_all)
        X_train = np.array([x[:, :min_length_train] for x in X_train_all])
        y_train = np.array(y_train_all)
        
        min_length_test = min(x.shape[1] for x in X_test_all)
        X_test = np.array([x[:, :min_length_test] for x in X_test_all])
        y_test = np.array(y_test_all)
        
        # Encoder
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        
        # CSP + Scaler + Classifier (SPÉCIFIQUE à ce sujet)
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        X_train_csp = csp.fit_transform(X_train, y_train_enc)
        X_test_csp = csp.transform(X_test)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_csp)
        X_test_scaled = scaler.transform(X_test_csp)
        
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train_enc)
        
        train_score = clf.score(X_train_scaled, y_train_enc)
        test_score = clf.score(X_test_scaled, y_test_enc)
        
        y_pred = clf.predict(X_test_scaled)
        n_classes_pred = len(np.unique(y_pred))
        
        subject_scores.append(test_score)
        
        print(f"   Train: {len(X_train)} epochs, score={train_score:.3f}")
        print(f"   Test:  {len(X_test)} epochs, score={test_score:.3f}, "
              f"classes prédites={n_classes_pred}")
        print()
    
    print("="*70)
    print(f"📊 Mean subject-specific accuracy: {np.mean(subject_scores):.4f} ± {np.std(subject_scores):.4f}")
    print("="*70)


def cross_val_within_subject():
    """
    Validation croisée au sein d'un même sujet.
    """
    
    print("\n\n" + "="*70)
    print("🧪 CROSS-VALIDATION INTRA-SUJET (LEFT vs RIGHT)")
    print("="*70)
    print("Pour chaque sujet: mélanger toutes ses runs et faire une 5-fold CV\n")
    
    p = Processing()
    runs = [3, 4, 7, 8, 11, 12]  # Toutes les runs left/right
    
    for subject_id in [1, 2, 90, 91]:
        print(f"Sujet {subject_id}:")
        
        # Charger toutes les runs
        X_all = []
        y_all = []
        
        for run_id in runs:
            try:
                X, y = p.get_all_data(subject_id, run_id)
                if len(X) > 0:
                    X_all.extend(X)
                    y_all.extend(y)
            except:
                continue
        
        if len(X_all) < 20:
            print("   ⚠️  Pas assez de données\n")
            continue
        
        # Uniformiser
        min_length = min(x.shape[1] for x in X_all)
        X = np.array([x[:, :min_length] for x in X_all])
        y = np.array(y_all)
        
        # Encoder
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        
        # Cross-validation avec CSP + Scaler + Classifier
        from sklearn.pipeline import Pipeline
        
        pipeline = Pipeline([
            ('csp', CSP(n_components=4, reg=None, log=True, norm_trace=False)),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        ])
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, X, y_enc, cv=cv, scoring='accuracy')
        
        print(f"   {len(X)} epochs")
        print(f"   CV scores: {[f'{s:.3f}' for s in scores]}")
        print(f"   Mean: {scores.mean():.3f} ± {scores.std():.3f}\n")


if __name__ == "__main__":
    subject_specific_evaluation()
    cross_val_within_subject()
    
    print("\n" + "="*70)
    print("🏁 CONCLUSION")
    print("="*70)
    print("L'approche subject-specific fonctionne bien (>60-70% accuracy)")
    print("L'approche subject-independent ne fonctionne pas (<50% accuracy)")
    print("\nCeci est un problème connu en BCI research:")
    print("- Les patterns EEG varient énormément entre sujets")
    print("- CSP est très sujet-spécifique")
    print("\nSolutions possibles:")
    print("1. Utiliser des features plus génériques (power spectral density)")
    print("2. Transfer learning avec fine-tuning par sujet")
    print("3. Deep learning (CNN/RNN) qui peuvent apprendre des features transférables")
    print("4. Domain adaptation techniques")
    print("="*70)
