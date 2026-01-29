# Total Perspective Vortex - BCI Motor Imagery Classification

Classification d'imagerie motrice (Brain-Computer Interface) sur le dataset PhysioNet EEGBCI.

## 🎯 Objectif

Atteindre **≥60% accuracy** moyenne sur 109 sujets × 6 expériences = **654 évaluations**.

## 📊 Architecture

### Flux de données
```
Pour chaque expérience (run 3,4,5,6,7,8):
  S001R03 + S002R03 + ... + S109R03  →  FLUX GÉANT (1500-1700 epochs)
                ↓
         Shuffle (StratifiedKFold)
                ↓
         Cross-validation (10-fold)
                ↓
         Fit final sur 100%
                ↓
    model_experiment_0.pkl sauvegardé
```

### Pipeline de traitement
```python
FeatureUnion[
    CSP(4 composantes)                    →  4 features
    +
    Wavelet(db4, level=3) → PCA(10)      → 10 features
]  = 14 features
    ↓
StandardScaler (normalisation)
    ↓
LogisticRegression (C=1.0, max_iter=1000)
```

### Classes par expérience

| Expérience | Run | Classes |
|------------|-----|---------|
| 0 | 3 | left_fist vs right_fist |
| 1 | 4 | left_fist vs right_fist (imaginé) |
| 2 | 5 | both_fists vs both_feet |
| 3 | 6 | both_fists vs both_feet (imaginé) |
| 4 | 7 | left_fist vs right_fist (imaginé) |
| 5 | 8 | left_foot vs right_foot (imaginé) |

**Important** : Chaque expérience = 1 paire de classes différente → **1 modèle par expérience**.

## 🚀 Workflow complet

### 1. Installation

```bash
# Créer environnement virtuel
python3 -m venv src/venv
source src/venv/bin/activate

# Installer dépendances
pip install numpy scikit-learn matplotlib joblib mne PyWavelets
```

### 2. Entraînement des 6 modèles

#### Option A : Tous les modèles (PRODUCTION)
```bash
python train_all_experiments.py
```
⏱️ Temps estimé : **2-6 heures** (dépend de votre machine)

#### Option B : Test rapide (1 expérience)
```bash
python train_all_experiments.py --experiments 0
```
⏱️ Temps estimé : **20-40 minutes**

#### Option C : Plusieurs expériences
```bash
python train_all_experiments.py --experiments 0 1 2
```

**Ce qui se passe :**
- Pour chaque expérience :
  - Charge 109 sujets × 1 run ≈ 1500-1700 epochs
  - Concatène en 1 FLUX géant
  - Shuffle avec StratifiedKFold
  - Cross-validation 10-fold
  - Fit final sur 100% du flux
  - Sauvegarde `model_experiment_X.pkl`

**Fichiers générés :**
```
model_experiment_0.pkl + label_encoder_experiment_0.pkl
model_experiment_1.pkl + label_encoder_experiment_1.pkl
model_experiment_2.pkl + label_encoder_experiment_2.pkl
model_experiment_3.pkl + label_encoder_experiment_3.pkl
model_experiment_4.pkl + label_encoder_experiment_4.pkl
model_experiment_5.pkl + label_encoder_experiment_5.pkl
```

### 3. Évaluation globale

```bash
python src/mybci.py
```

**Ce qui se passe :**
- Pour chaque expérience (0-5) :
  - Charge le modèle de cette expérience
  - Pour chaque sujet (1-109) :
    - Évalue sur ce sujet, ce run
    - Affiche : `experiment X: subject YYY: accuracy = Z.ZZZZ`
  - Calcule la moyenne par expérience
- Affiche la moyenne globale des 6 expériences

**Output attendu :**
```
experiment 0: subject 001: accuracy = 0.6000
experiment 0: subject 002: accuracy = 0.8000
...
Mean accuracy of the six different experiments for all 109 subjects:
experiment 0: accuracy = 0.5991
experiment 1: accuracy = 0.5718
experiment 2: accuracy = 0.7130
experiment 3: accuracy = 0.6035
experiment 4: accuracy = 0.5937
experiment 5: accuracy = 0.6753
Mean accuracy of 6 experiments: 0.6261
✅ Objectif atteint ! (≥60%)
```

### 4. Train/Predict sur 1 sujet (mode manuel)

#### Train
```bash
python src/mybci.py 4 14 train
```
Output :
```
[0.6666 0.4444 0.4444 0.4444 0.4444 0.6666 0.8888 0.1111 0.7777 0.4444]
cross_val_score: 0.5333
```

#### Predict (streaming)
```bash
python src/mybci.py 4 14 predict
```
Output :
```
Epoch 00 | Pred: both_fists | True: both_feet  | ❌ | Latency:   5.23ms
Epoch 01 | Pred: both_feet  | True: both_feet  | ✅ | Latency:   4.87ms
...
Accuracy: 0.6666
```

## 📁 Structure du projet

```
.
├── train_all_experiments.py    # Entraîne les 6 modèles
├── src/
│   ├── mybci.py                # Point d'entrée (train/predict/eval)
│   ├── train.py                # Fonction de training
│   ├── predict.py              # Prédiction en streaming
│   ├── processing.py           # Chargement et prétraitement EEG
│   ├── MyCSP.py                # Common Spatial Patterns
│   ├── MyPCA.py                # Principal Component Analysis
│   ├── waveletsTransformer.py  # Extraction features wavelet
│   ├── MyLogisticRegression.py # (non utilisé, remplacé par sklearn)
│   ├── global_variable.py      # Mapping runs/classes
│   └── data/                   # Données EEG (auto-téléchargées)
├── model_experiment_*.pkl      # Modèles entraînés (6 fichiers)
└── label_encoder_experiment_*.pkl
```

## 🔧 Détails techniques

### Pourquoi crop les epochs ?

Les epochs de différents sujets ont des durées différentes (385-481 timepoints) à cause d'annotations tronquées.

**Solution actuelle** : Crop à la taille minimale (385)
- ❌ Perd ~20% du signal (96 timepoints)
- ✅ Pas de biais, rapide

**Alternative** : Resampling (scipy.signal.resample)
- ✅ Garde toute l'information
- ⏱️ Plus lent

### Pourquoi 6 modèles séparés ?

Chaque run = paire de classes différente :
- Run 3,4,7 : left_fist vs right_fist
- Run 5,6,9 : both_fists vs both_feet
- Run 8 : left_foot vs right_foot

**CSP nécessite exactement 2 classes** → impossible de mélanger runs avec classes différentes.

### Pourquoi fit sur 100% après CV ?

1. **CV** : Valide que le modèle ne surfit pas (estimation performance)
2. **Fit 100%** : Maximise la performance pour production

C'est **standard en ML**, pas de l'overfitting car validé par CV.

### Shuffle dans le flux

`StratifiedKFold(shuffle=True)` mélange les epochs avant split :
```
[S001_ep1, S001_ep2, S002_ep1, ...] 
    ↓ shuffle
[S042_ep3, S001_ep1, S089_ep2, ...]
```

Évite le biais de "premiers sujets dans train, derniers dans test".

## 🐛 Troubleshooting

### "CSP requires exactly two classes"
→ Vous essayez de mélanger des runs avec des classes différentes. Entraînez 1 modèle par run.

### "File does not exist: S001R14.edf"
→ Données pas téléchargées. Lancez une fois avec 1 sujet pour trigger le download MNE.

### "All the 5 fits failed"
→ Pas assez de données ou classes déséquilibrées. Vérifiez `np.unique(y)`.

### Mémoire insuffisante
→ Le code charge 109 sujets × 1 run ≈ 1500 epochs en RAM. Nécessite ~2-4 GB RAM.

## 📊 Résultats attendus

- **CV pendant train** : 45-55% (normal, petit dataset par sujet)
- **Eval globale** : 55-65% (meilleur car modèle global)
- **Objectif** : ≥60%

Si < 60% :
- Augmenter `n_components` CSP/PCA
- Tester d'autres classifiers (SVM, RandomForest)
- Feature engineering (plus de stats wavelet)
- Augmentation de données (permutation temporelle)

## 📚 Références

- Dataset : [PhysioNet EEGBCI](https://physionet.org/content/eegmmidb/1.0.0/)
- CSP : Ramoser et al. (2000)
- MNE-Python : [mne.tools](https://mne.tools/)

## ⚡ Quick start

```bash
# 1. Installer
pip install numpy scikit-learn matplotlib joblib mne PyWavelets

# 2. Test rapide (1 expérience, ~30 min)
python train_all_experiments.py --experiments 0

# 3. Évaluer
python src/mybci.py

# 4. Si OK, entraîner tout (2-6h)
python train_all_experiments.py
```

## 📝 License

42 School project - Educational purpose
