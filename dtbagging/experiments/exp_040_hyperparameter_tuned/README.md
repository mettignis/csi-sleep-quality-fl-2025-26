# Experiment 040: Federated Hyperparameter Tuning (PER-CLIENT)

**Date:** 7 Gennaio 2026  
**Status:** ❌ FAILED - Severe Overfitting  
**Kaggle Score:** 12.61584 MAE (worse than baseline)  
**Degradation:** +0.72 MAE from exp_039 (11.89067)

---

## Overview

Dopo il successo moderato di exp_039 (data augmentation), exp_040 tenta di replicare il successo della grid search estesa degli esperimenti centralizzati iniziali (che avevano raggiunto 11.18 MAE), ma in modalità **true Federated Learning** con grid search indipendente per ogni client.

**Objective:** Trovare hyperparameters ottimali per-client tramite extensive grid search (8000 combinazioni totali) mantenendo FL compliance.

---

## Motivation

### Ricordo del Successo Passato

Nei primi esperimenti (exp_017-030, poi invalidati per centralized training), una **grid search da 3500 combinazioni** aveva raggiunto 11.18 MAE - il miglior score mai ottenuto (anche se non FL-compliant).

**Domanda dell'utente:** *"pk l'avevamo fatta? si può riproporre questo metodo anche con il federated learning?"*

**Decision:** Implementare extensive grid search PER-CLIENT invece che centralizzata:
- TRUE FL: ogni client fa grid search sui propri dati augmented
- EXTENSIVE: 400 RF + 500 LGBM combinazioni per client
- TOTAL: 8000 evaluations (exceeding original 3500)
- ENSEMBLE: Bagging predictions from 9 RF + 9 LGBM models

---

## Grid Search Configuration

### Random Forest Grid (400 combinations/client)

```python
param_grid = {
    'n_estimators': [100, 150, 200, 250, 300],      # 5 values
    'max_depth': [12, 15, 18, 20, 25],              # 5 values
    'min_samples_split': [2, 3, 4, 5],              # 4 values
    'min_samples_leaf': [1, 2, 3, 4]                # 4 values
}
# Total: 5×5×4×4 = 400 combinations per client
```

**Per client:**
1. Load augmented data (~650 samples)
2. Split 80/20 train/validation
3. Test all 400 hyperparameter combinations
4. Select best based on validation MAE
5. Train final model on full client data with best params

**Total RF evaluations:** 400 × 9 clients = **3600**

### LightGBM Grid (500 combinations/client)

```python
param_grid = {
    'num_leaves': [25, 31, 40, 50, 60],            # 5 values
    'learning_rate': [0.02, 0.03, 0.04, 0.05, 0.06], # 5 values
    'max_depth': [6, 8, 10, 12],                   # 4 values
    'n_estimators': [100, 200, 300, 400, 500]      # 5 values
}
# Total: 5×5×4×5 = 500 combinations per client
```

**Special features:**
- Early stopping with 50 rounds
- Tracks best_iteration per model

**Total LGBM evaluations:** 500 × 9 clients = **4500**

**GRAND TOTAL:** 8000 evaluations (~2x original centralized 3500)

---

## Training Execution

### RF Training (2.5 hours)

**Command:**
```bash
python -u tune_rf_federated.py 2>&1 | tee rf_training_live.log
```

**Results per client:**

| Client | Samples | Train | Val | Best Val MAE | Best Config |
|--------|---------|-------|-----|--------------|-------------|
| 0      | 632     | 505   | 127 | **4.999**    | n=200, d=18, s=2, l=1 |
| 1      | 554     | 443   | 111 | **5.171**    | n=100, d=18, s=4, l=1 |
| 2      | 578     | 462   | 116 | **5.925**    | n=100, d=20, s=2, l=1 ⚠️ |
| 3      | 686     | 548   | 138 | **6.226**    | n=100, d=18, s=2, l=1 |
| 4      | 542     | 433   | 109 | **6.209**    | n=100, d=25, s=4, l=1 ⚠️ |
| 5      | 740     | 592   | 148 | **6.685**    | n=100, d=20, s=2, l=1 ⚠️ |
| 6      | 632     | 505   | 127 | **6.770**    | n=100, d=25, s=3, l=1 ⚠️ |
| 7      | 818     | 654   | 164 | **12.108**   | n=100, d=18, s=4, l=1 ⚠️ |
| 8      | 722     | 577   | 145 | **7.026**    | n=300, d=18, s=3, l=1 |

**Average Val MAE:** 6.791  
**Std Val MAE:** 1.988

**⚠️ Overfitting indicators:**
- Client 2,4,5,6: max_depth ≥20 + min_samples_leaf=1 (extremely deep + small leaves)
- Client 7: Val MAE 12.108 (doppio degli altri, huge Train-Val gap)

### LightGBM Training (2.5 hours)

**Command:**
```bash
python -u tune_lgbm_federated.py 2>&1 | tee lgbm_training_live.log
```

**Results per client:**

| Client | Samples | Train | Val | Best Val MAE | Best Config | Best Iter |
|--------|---------|-------|-----|--------------|-------------|-----------|
| 0      | 632     | 505   | 127 | **3.965**    | l=25, lr=0.06, d=12, n=500 | 500 |
| 1      | 554     | 443   | 111 | **4.624**    | l=25, lr=0.04, d=8, n=500  | 499 |
| 2      | 578     | 462   | 116 | **4.588**    | l=25, lr=0.05, d=10, n=500 | 500 |
| 3      | 686     | 548   | 138 | **6.202**    | l=25, lr=0.06, d=8, n=500  | 499 |
| 4      | 542     | 433   | 109 | **6.270**    | l=25, lr=0.06, d=6, n=500  | 500 |
| 5      | 740     | 592   | 148 | **5.973**    | l=25, lr=0.06, d=6, n=500  | 500 |
| 6      | 632     | 505   | 127 | **7.607**    | l=25, lr=0.02, d=6, n=100  | **21** ⚠️ |
| 7      | 818     | 654   | 164 | **11.678**   | l=25, lr=0.02, d=6, n=100  | **21** ⚠️ |
| 8      | 722     | 577   | 145 | **7.715**    | l=25, lr=0.02, d=8, n=100  | **18** ⚠️ |

**Average Val MAE:** 6.514  
**Std Val MAE:** 2.199

**⚠️ Overfitting indicators:**
- Client 6,7,8: Early stopping MOLTO presto (18-21 iterations su 100-500 requested)
- Modelli instabili, dati difficili da fittare

---

## Ensemble Generation

### Bagging Strategy

Per ogni test sample:
1. **RF Predictions:** Ottieni predizioni da tutti 9 RF models (uno per client)
2. **RF Bagging:** Media le 9 predizioni RF
3. **LGBM Predictions:** Ottieni predizioni da tutti 9 LGBM models
4. **LGBM Bagging:** Media le 9 predizioni LGBM
5. **Final Ensemble:** `0.65 × RF_bagging + 0.35 × LGBM_bagging`

### Prediction Statistics

**RF Bagging (9 models):**
- Mean: 73.57
- Std: 3.32

**LGBM Bagging (9 models):**
- Mean: 73.44
- Std: 4.45

**Final Ensemble (65% RF + 35% LGBM):**
- Mean: 73.52
- Std: 3.64
- Min: 63.14
- Max: 79.48

---

## Results: CATASTROPHIC FAILURE

### Validation vs Kaggle Performance

| Metric | exp_039 | exp_040 | Change |
|--------|---------|---------|--------|
| **RF Val MAE** | 8.79 | **6.79** | **-2.00** ✅ |
| **LGBM Val MAE** | 8.58 | **6.51** | **-2.07** ✅ |
| **Kaggle MAE** | **11.89** | **12.62** | **+0.73** ❌ |

### Overfitting Analysis

**exp_039:**
- Validation improvement: baseline
- Kaggle improvement: -0.036 from exp_031
- **Overfitting gap:** ~3.3 MAE (11.89 Kaggle vs 8.58-8.79 Val)

**exp_040:**
- Validation improvement: -2.00 MAE (spectacular!)
- Kaggle degradation: +0.73 MAE (disaster!)
- **Overfitting gap:** ~6.0 MAE (12.62 Kaggle vs 6.51-6.79 Val) ⚠️

**Overfitting gap DOUBLED** from exp_039 to exp_040!

---

## Root Cause Analysis

### 1. Validation Set Too Small

**Problem:**
- Each client: ~650 samples → 80/20 split → **~130 validation samples**
- Testing 400-500 hyperparameter combinations on 130 samples
- **Ratio:** 400-500 configs : 130 samples = **3-4 configurations per validation sample!**

**Result:** Models memorize validation set instead of learning generalizable patterns

### 2. Extreme Hyperparameters Selected

**Random Forest:**
- 5/9 clients selected `max_depth ≥20` with `min_samples_leaf=1`
- These are EXTREME configurations (very deep, very small leaves)
- Perfect for memorization, terrible for generalization

**LightGBM:**
- 3/9 clients (6,7,8) stopped after 18-21 iterations (out of 100-500)
- Premature early stopping = data too difficult/models too unstable
- Selected ultra-conservative configs (lr=0.02, minimal complexity)

### 3. Client 7 Pathological Case

**Client 7 characteristics:**
- 818 samples (largest client)
- High label variability (Std=16.42)
- Mean 69.58 (lowest among clients)

**Client 7 results:**
- RF Val MAE: **12.108** (2x worse than other clients)
- LGBM Val MAE: **11.678** (also 2x worse)
- Train MAE: ~1.5-2.0 (huge Train-Val gap = severe overfitting)

**Impact on ensemble:** Client 7 contributes 1/9 to bagging → pulls down overall quality

### 4. Grid Search on Augmented Data

**Augmented data characteristics:**
- Original: Real sensor data
- Augmented: Synthetic (noise, scaling, interpolation, mixup)

**Problem:** Hyperparameters optimized on synthetic validation might not transfer to real test data

---

## Comparison with Original Grid Search

### Why Original 3500-Combo Search Worked (11.18 MAE)?

**Centralized approach (exp_017-030, INVALID):**
- All 5904 samples concatenated into single dataset
- 80/20 split: **1181 validation samples** (9x larger than per-client)
- 3500 combinations tested on 1181 samples
- **Ratio:** 3500:1181 = 3:1 (much better than exp_040's 4:1)

**exp_040 federated approach:**
- 9 separate 650-sample datasets
- 80/20 split: **~130 validation samples per client**
- 400-500 combinations tested on 130 samples
- **Ratio:** 400-500:130 = 3-4:1 (worse ratio + smaller absolute size)

**Key difference:** Validation set size matters MORE than train set size for hyperparameter tuning!

---

## Key Learnings

### 1. Small Validation = Hyperparameter Overfitting

With validation sets < 200 samples:
- ❌ Don't do extensive grid search (hundreds of combinations)
- ❌ Don't select extreme hyperparameters (very deep, very complex)
- ✅ Use conservative default hyperparameters
- ✅ Rely on cross-validation or larger validation holdout

### 2. Validation Improvement ≠ Test Improvement

**exp_040 paradox:**
- Best validation performance ever (-2.00 MAE)
- Worst Kaggle performance in months (+0.73 MAE)
- **Lesson:** Spectacular validation improvement with small validation set = RED FLAG for overfitting

### 3. Per-Client Tuning Has Limits

**FL constraint creates dilemma:**
- Can't use centralized validation (violates FL)
- Per-client validation too small (overfitting risk)
- **Solution:** Use conservative hyperparameters without per-client tuning

### 4. Simple > Complex for Small Data

**exp_039 (simple, fixed hyperparameters):**
- Fixed RF: n=200, depth=15, split=5, leaf=1
- Fixed LGBM: n=200, lr=0.05, depth=8
- Result: 11.89 MAE ✅

**exp_040 (complex, tuned hyperparameters):**
- 8000 evaluations to find "optimal" configs
- Result: 12.62 MAE ❌

**Lesson:** Default hyperparameters > "optimized" on tiny validation sets

---

## FL Compliance

✅ **TRUE Federated Learning maintained:**
- Each client's grid search independent
- No data sharing between clients
- Bagging for aggregation (model-level, not data-level)
- No centralized validation pooling

**Note:** FL compliance preserved even in failure - the approach was principled, just ineffective.



## Conclusion

exp_040 è un **fallimento catastrofico** nonostante l'approccio teoricamente solido e 5+ ore di training intensivo. L'extensive grid search, che aveva funzionato benissimo in modalità centralizzata (11.18 MAE), si è rivelata **dannosa in modalità federated** a causa di validation sets troppo piccoli.

**Validation improvement:** -2.00 MAE (spectacular) ✅  
**Kaggle degradation:** +0.73 MAE (catastrophic) ❌  
**Overfitting gap:** DOUBLED (3.3 → 6.0 MAE)
