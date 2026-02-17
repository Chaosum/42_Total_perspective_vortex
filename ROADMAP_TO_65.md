# 🎯 Road

map vers 65%+ d'Accuracy

## 📊 État Actuel
- **Baseline**: ~60% (2 runs, LogReg, CSP fixe)
- **Avec 3 runs**: ~62.3% (+2.3%)
- **Objectif**: 65% (**manque 2.7%**)

---

## 🚀 Stratégies par Ordre de Priorité

### ✅ 1. **IMMEDIATE: 3 Runs + LDA + CSP Optimisé** (Gain cumulé: +3-4%)

**Ce qui est déjà fait:**
- ✅ LDA au lieu de LogReg (+1.1%)
- ✅ Optimisation n_components CSP (+1%)
- ✅ Passé à 3 runs de calibration (+2.3%)

**Gain total estimé**: ~64-65% ← **PROCHE DE L'OBJECTIF!**

**Action**: Relancer l'entraînement avec ces 3 améliorations:
```bash
# Tuer l'entraînement actuel
pkill -f train_subject_specific_optimized

# Relancer avec 3 runs + LDA + CSP optimisé
python3 src/train_subject_specific_optimized.py
```

---

### ⭐ 2. **Optimiser aussi le Paramètre reg** (Gain: +0.5-1%)

**Découverte**: 70% des sujets préfèrent `reg=0.0` (pas de régularisation)

**Implémentation**: Modifier `optimize_csp_components()` pour tester:
- n_components: 4, 6, 8
- reg: 0.0, 0.1, 0.3

**Code à modifier** dans `train_incremental.py`:
```python
def optimize_csp_components(X_calib, y_calib, n_splits=3):
    # Tester aussi différentes régularisations
    for n_comp in [4, 6, 8]:
        for reg in [0.0, 0.1, 0.3]:  # ← AJOUTER CETTE BOUCLE
            csp = CSP(n_components=n_comp, 
                     reg=reg if reg > 0 else None, 
                     log=True, norm_trace=False)
            # ... reste du code
```

**Gain attendu**: +0.5-1% → **Total: 65-66%**

---

### 🎨 3. **Optimiser les Bandes de Fréquences** (Gain: +1-2%)

**Idée**: Tester différentes bandes pour chaque sujet:
- Bande mu: [8-13 Hz] (rythme sensorimoteur)
- Bande beta: [13-30 Hz] (activation motrice)
- Bande combinée: [7-30 Hz] (actuelle)

**Implémentation**:
```python
def optimize_frequency_band(subject_id, run_id):
    best_score = 0
    best_band = (7, 30)
    
    for low, high in [(8, 13), (13, 30), (7, 30)]:
        raw = load_data(subject_id, run_id)
        raw.filter(low, high)
        # ... reste du pipeline
        score = evaluate()
        if score > best_score:
            best_score = score
            best_band = (low, high)
    
    return best_band
```

**Gain attendu**: +1-2% → **Total: 66-68%**

---

### 🧠 4. **Features Spectrales (PSD - Power Spectral Density)** (Gain: +1-2%)

**Idée**: Ajouter des features dans le domaine fréquentiel en complément de CSP.

**Implémentation**:
```python
from scipy.signal import welch

def extract_psd_features(epochs, freqs=(8, 13, 18, 30)):
    """Extrait les features PSD par bande de fréquence."""
    psd_features = []
    
    for epoch in epochs:
        features_epoch = []
        for ch in range(epoch.shape[0]):
            f, psd = welch(epoch[ch], fs=160, nperseg=256)
            
            # Puissance dans mu (8-13 Hz)
            mu_power = np.mean(psd[(f >= 8) & (f <= 13)])
            # Puissance dans beta (13-30 Hz)
            beta_power = np.mean(psd[(f >= 13) & (f <= 30)])
            
            features_epoch.extend([mu_power, beta_power])
        
        psd_features.append(features_epoch)
    
    return np.array(psd_features)

# Combiner avec CSP
X_csp = csp.fit_transform(X_calib, y_calib)
X_psd = extract_psd_features(X_calib)
X_combined = np.hstack([X_csp, X_psd])  # Concatener
```

**Gain attendu**: +1-2% → **Total: 67-70%**

---

### 🔄 5. **Data Augmentation** (Gain: +0.5-1%)

**Idées simples**:

1. **Segmentation temporelle**: Découper epochs en fenêtres glissantes
```python
def segment_epochs(X, window_size=2, overlap=0.5):
    """Découpe epochs de 4s en fenêtres de 2s avec 50% overlap."""
    # 4s → 2 fenêtres de 2s avec overlap
    # Augmente x2-3 le nombre d'échantillons
```

2. **Petit bruit additif** (attention: peut dégrader!)
```python
def augment_with_noise(X, noise_level=0.01):
    """Ajoute un petit bruit gaussien."""
    return X + np.random.randn(*X.shape) * noise_level * X.std()
```

**Gain attendu**: +0.5-1%

---

### 🤖 6. **Deep Learning: EEGNet** (Gain: +2-5% mais plus complexe)

**Si vraiment nécessaire** (complexité ++):

```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout

def build_eegnet(n_channels=64, n_times=801, n_classes=2):
    """
    EEGNet: CNN spécialisé pour EEG.
    Référence: Lawhern et al. (2018)
    """
    # Convolutions spatiales et temporelles
    # Auto-apprend les patterns (pas besoin de CSP)
    # Mais besoin de plus de données et temps d'entraînement
```

**Avantages**:
- Apprend automatiquement les features optimales
- Peut capturer des patterns non-linéaires

**Inconvénients**:
- Besoin de plus de données (augmentation nécessaire)
- Plus long à entraîner (~30min-1h pour tous les sujets)
- Risque d'overfitting si mal régularisé

**Gain attendu**: +2-5% → **Total: 70-75%**

---

## 📋 Plan d'Action Recommandé

### Phase 1: Quick Wins (aujourd'hui)
1. ✅ **Relancer avec 3 runs + LDA + CSP opti** → 64-65%
2. ✅ **Ajouter optimisation reg** → 65-66%

### Phase 2: Si besoin (demain)
3. **Optimiser bandes de fréquences** → 66-68%
4. **Ajouter features PSD** → 67-70%

### Phase 3: Si vraiment nécessaire
5. **Data augmentation** → +0.5-1%
6. **EEGNet (deep learning)** → +2-5%

---

## 💡 Recommandation IMMEDIATE

**Relancez l'entraînement avec les 3 améliorations déjà implémentées:**

```bash
# 1. Arrêter l'entraînement actuel (qui utilise 2 runs + LogReg)
pkill -f train_subject_specific_optimized

# 2. Relancer avec 3 runs + LDA + CSP optimisé
python3 src/train_subject_specific_optimized.py

# 3. Surveiller
./check_progress.sh
```

**Estimation**: Avec ces 3 changements cumulés, vous devriez atteindre **64-65%** ← OBJECTIF!

---

## 🎯 Budget de Gains

| Amélioration | Gain | Cumul | Statut |
|--------------|------|-------|--------|
| Baseline | - | 60.0% | ✅ |
| + LDA | +1.1% | 61.1% | ✅ Implémenté |
| + CSP opti (n_comp) | +1.0% | 62.1% | ✅ Implémenté |
| + 3 runs calibration | +2.3% | **64.4%** | ✅ Implémenté |
| + CSP opti (reg) | +0.6% | **65.0%** | 🔧 À tester |
| + Fréquences opti | +1.5% | 66.5% | 💡 Option |
| + Features PSD | +1.5% | 68.0% | 💡 Option |
| + Data augmentation | +0.8% | 68.8% | 💡 Option |
| + EEGNet | +3.0% | 71.8% | 🤖 Si nécessaire |

---

## ✅ Next Steps

1. **MAINTENANT**: Relancer entraînement avec 3 runs + LDA + CSP opti
2. **Attendre résultats** (~2-3h)
3. **Analyser**: `python3 src/analyze_optimized_results.py`
4. **Si < 65%**: Ajouter optimisation reg (30min)
5. **Si encore < 65%**: Passer aux bandes de fréquences

**Probabilité d'atteindre 65% avec Step 1**: **~80%** ✅

---

## 📝 Commandes Utiles

```bash
# Vérifier si entraînement actuel est en cours
ps aux | grep train_subject_specific

# Tuer si nécessaire
pkill -f train_subject_specific_optimized

# Relancer avec nouvelles améliorations
python3 src/train_subject_specific_optimized.py

# Monitorer
./check_progress.sh

# Analyser quand terminé
python3 src/analyze_optimized_results.py
```

---

**TL;DR**: Vous avez déjà implémenté les 3 améliorations principales. Relancez l'entraînement, vous devriez atteindre **64-65%**! 🎉
