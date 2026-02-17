# Optimisation CSP - Strategy et Implémentation

## 📋 Résumé

Cette optimisation vise à améliorer l'accuracy des modèles subject-specific en adaptant le nombre de composantes CSP (Common Spatial Patterns) pour chaque sujet individuellement.

## 🎯 Objectif

Atteindre ≥ 65% de mean test accuracy en optimisant automatiquement le nombre de composantes CSP pour chaque sujet.

## 🔧 Stratégie Implémentée

### 1. Optimisation par Validation Croisée

Pour chaque sujet, on teste 3 configurations CSP:
- **4 composantes**: Configuration minimaliste, réduit le risque d'overfitting
- **6 composantes**: Configuration par défaut (baseline précédente)
- **8 composantes**: Configuration maximale, capture plus de variance

### 2. Validation Croisée Stratifiée (3-fold)

```python
def optimize_csp_components(X_calib, y_calib, n_splits=3):
    """
    Trouve le nombre optimal de composantes CSP via CV.
    
    Pour chaque n_components (4, 6, 8):
        - Split les données de calibration en 3 folds
        - Pour chaque fold:
            * Entraîne CSP + Scaler + LogisticRegression sur train
            * Évalue sur validation
        - Calcule le score CV moyen
    
    Retourne le n_components avec le meilleur score CV.
    """
```

### 3. Entraînement Final

Une fois le nombre optimal de composantes déterminé:
1. **Re-entraîner** sur toutes les données de calibration avec `n_components` optimal
2. **Évaluer** sur les runs de test (mouvement imaginé)
3. **Sauvegarder** le modèle avec métadonnées (n_components, cv_score)

## 📊 Avantages Attendus

### Adaptation Individuelle
- Certains sujets ont des patterns cérébraux simples → **4 composantes** suffisent
- D'autres ont des patterns complexes → **8 composantes** capturent mieux la variance

### Réduction de l'Overfitting
- Si les données d'un sujet sont bruitées, l'algorithme choisira moins de composantes
- Évite de sur-apprendre des patterns non généralisables

### Gain d'Accuracy Estimé
- **+1-2%** d'amélioration moyenne attendue
- Certains sujets "difficiles" pourraient gagner **+5-10%**

## 🔬 Détails Techniques

### Configuration CSP

```python
csp = CSP(
    n_components=optimal_n,  # 4, 6, ou 8 (optimisé par CV)
    reg=0.1,                 # Régularisation pour stabilité
    log=True,                # Transformation log-variance
    norm_trace=False         # Pas de normalisation de trace
)
```

### Régularisation

Le paramètre `reg=0.1` ajoute une régularisation Ledoit-Wolf pour:
- Stabiliser l'estimation de la matrice de covariance
- Améliorer la robustesse sur petits datasets
- Éviter les matrices singulières

### Pipeline Complet

```
Données EEG brutes (64 canaux, 801 timepoints)
    ↓
Filtrage 7-30 Hz (bande mu/beta)
    ↓
Epochs 0-4s après stimulus
    ↓
CSP avec n_components optimal (4, 6, ou 8)
    ↓
StandardScaler (normalisation features)
    ↓
LogisticRegression (C=0.1, max_iter=1000)
    ↓
Prédiction (left/right ou hands/feet)
```

## 📈 Résultats Attendus

### Distribution des Composantes Optimales

On s'attend à une distribution comme:
- **4 composantes**: ~30-40% des sujets (patterns simples)
- **6 composantes**: ~40-50% des sujets (complexité moyenne)
- **8 composantes**: ~15-25% des sujets (patterns complexes)

### Amélioration de Performance

**Baseline (6 composantes fixes)**:
- Mean accuracy: ~59-60%

**Avec optimisation CSP**:
- Mean accuracy: **61-62%** (objectif +1-2%)
- Meilleures sujets: **80-85%** (vs 75-80% avant)
- Sujets ≥ 65%: **35-45%** (vs 25-30% avant)

## 🛠️ Implémentation

### Fichiers Modifiés

1. **`train_incremental.py`**
   - Ajout de la fonction `optimize_csp_components()`
   - Intégration dans `train_subject_specific()`
   - Sauvegarde de `n_components` et `cv_score` dans les modèles

2. **`train_subject_specific_optimized.py`** (nouveau)
   - Script standalone pour entraînement complet
   - Lance les 2 tâches (left_right + hands_feet)
   - Affiche résultats et statistiques

3. **`analyze_optimized_results.py`** (nouveau)
   - Analyse des modèles sauvegardés
   - Statistiques détaillées par tâche
   - Distribution des composantes optimales

### Usage

```bash
# Entraîner tous les sujets avec optimisation CSP
python3 src/train_subject_specific_optimized.py

# Analyser les résultats
python3 src/analyze_optimized_results.py

# Ou utiliser train_incremental.py directement
python3 -c "from src.train_incremental import train_subject_specific; \
            train_subject_specific(task_name='left_right')"
```

## 🎓 Justification Théorique

### Pourquoi CSP?

CSP (Common Spatial Patterns) est la méthode de référence pour:
- **Maximiser** la variance d'une classe
- **Minimiser** la variance de l'autre classe
- Identifier les **canaux EEG les plus discriminants**

### Pourquoi Optimiser n_components?

Le nombre de composantes contrôle le **trade-off biais-variance**:
- **Trop peu** → perte d'information (underfitting)
- **Trop de** → capture du bruit (overfitting)
- **Optimal** → meilleur compromis pour chaque sujet

### Variabilité Inter-Sujets

Chaque cerveau est unique:
- Différences anatomiques
- Variabilité des patterns moteurs
- Qualité du signal EEG (placement électrodes, impédance)
- Capacité à générer des patterns distincts

→ **Solution**: Adapter le modèle à chaque sujet!

## 📊 Métriques de Suivi

Pour chaque sujet, on sauvegarde:
- `test_score`: Accuracy sur mouvement imaginé (métrique principale)
- `train_score`: Accuracy sur calibration (détection d'overfitting)
- `n_components`: Nombre optimal de composantes CSP
- `cv_score`: Score de validation croisée (confiance de la sélection)

### Indicateurs de Qualité

- **train_score >> test_score**: Overfitting (le modèle n'est pas assez régularisé)
- **train_score ≈ test_score**: Bonne généralisation
- **cv_score faible**: Peu de signal, choix de n_components peu fiable

## 🚀 Prochaines Étapes (si nécessaire)

Si l'optimisation CSP n'atteint pas 65%:

1. **Optimisation des bandes de fréquences**
   - Tester différentes bandes: [8-12 Hz], [13-30 Hz], [8-30 Hz]
   - Optimiser par sujet

2. **Features temporelles additionnelles**
   - Ajouter des features dans le domaine temporel
   - Moyenne, variance, pente des signaux

3. **Augmentation de données**
   - Segmentation temporelle (fenêtres glissantes)
   - Augmentation du nombre d'epochs de calibration

4. **Deep Learning**
   - EEGNet (CNN spécialisé pour EEG)
   - Entraînement avec plus de données

## 📝 Conclusions

L'optimisation CSP est une amélioration **simple mais efficace**:
- ✅ Pas de complexité algorithmique supplémentaire
- ✅ Temps d'entraînement minimal (validation croisée rapide)
- ✅ Interprétable (on sait pourquoi un sujet a 4 vs 8 composantes)
- ✅ Amélioration attendue: +1-2% d'accuracy moyenne

Cette optimisation démontre l'importance de l'**adaptation individuelle** en BCI, un principe fondamental pour des systèmes performants en conditions réelles.
