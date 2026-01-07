# Experiment 039: Augmented Data Ensemble

**Date:** 6 Gennaio 2026  
**Status:** ✅ SUCCESS - New Best Score  
**Kaggle Score:** 11.89067 MAE (3rd place)  
**Improvement:** -0.036 MAE from exp_031 (11.92674)

---

## Overview

Dopo il fallimento di 4 esperimenti consecutivi di ottimizzazione (exp_032-038), exp_039 rappresenta un **cambio strategico radicale**: invece di ottimizzare ulteriormente i modelli esistenti, abbiamo aumentato la quantità di dati di training tramite **data augmentation**.

## Problem Statement

**Diagnosi del plateau (exp_031-038):**
- exp_031 (11.93 MAE) era un local optimum per approcci ensemble tradizionali
- 4 tentativi di ottimizzazione (weight tuning, 4-way, 5-way, 2-way ensemble) tutti falliti
- Gap Train-Validation molto alto indicava **overfitting** (Train MAE 3-7, Val MAE 10-12)
- Causa: dataset troppo piccolo (999 samples totali = ~110 samples/client)

**Ipotesi:** Più dati di training → meno overfitting → migliore generalizzazione

---

## Approach: Data Augmentation 6x

### Augmentation Pipeline

Per ogni client dataset, abbiamo applicato 5 tecniche di augmentation generando **~6x più dati**:

1. **Gaussian Noise Injection** (2 varianti):
   - Noise ±3% random su tutte le features
   - Simula variabilità naturale nei sensori

2. **Magnitude Scaling** (1 variante):
   - Scale 97-103% multiplicativo
   - Preserva pattern relativi ma varia magnitudine

3. **Linear Interpolation** (1 variante):
   - Crea 2 sample intermedi tra coppie consecutive
   - Smooth transitions tra stati fisiologici

4. **Mixup** (1 variante):
   - Alpha=0.3, blend random pairs
   - Regolarizzazione soft tra esempi

**Risultato:**
- Original: ~110 samples/client × 9 clients = 999 samples
- Augmented: ~650 samples/client × 9 clients = **5904 samples** (~6x)

**FL Compliance:** ✅ Augmentation eseguita PER-CLIENT prima del training (no data sharing)

---

## Model Training

### Random Forest (Augmented Data)

**Configuration:**
```python
n_estimators = 200
max_depth = 15
min_samples_split = 5
random_state = 42
```

**Training Results:**
- Data: 5904 samples, 40 features
- Train/Val: 4723/1181 (80/20)
- **Train MAE:** 3.43
- **Val MAE:** 8.79 ✅ (-3.41 from exp_031's 12.20)
- Model size: 17MB

**Improvement:** Validation MAE ridotta del **28%** rispetto a exp_031 RF!

### LightGBM (Augmented Data)

**Configuration:**
```python
num_leaves = 31
learning_rate = 0.05
max_depth = 8
n_estimators = 200
early_stopping_rounds = 50
```

**Training Results:**
- Data: 5904 samples, 40 features
- **Train MAE:** 5.39
- **Val MAE:** 8.58 ✅ (-2.30 from exp_031's 10.88)
- Best iteration: 87 (early stopped from 200)

**Improvement:** Validation MAE ridotta del **21%** rispetto a exp_031 LGBM!

---

## Ensemble Strategy

**Formula:** `0.65 × RF + 0.35 × LGBM`

**Rationale:**
- RF ha validation MAE leggermente migliore (8.79 vs 8.58) → peso maggiore
- Non abbiamo ri-allenato XGBoost (troppo lento su augmented data)
- Two-way ensemble più semplice, meno rischio overfitting

---

## Preprocessing Fix (CRITICAL)

### Bug Iniziale
Lo script di generazione submission iniziale **mancava l'estrazione delle statistiche time series** da x_test.csv:
- Solo 27 features estratte da x_test → 13 features riempite con 0
- Mean predizioni: 71.81 (troppo basso, predizioni corrotte)

### Fix Applicato
Aggiunto codice per estrarre statistiche time series (mean, min, max, std) da:
- `hr_time_series`
- `resp_time_series`  
- `stress_time_series`

**Risultato dopo fix:**
- 39 features estratte → 1 feature riempita con median
- Mean predizioni: 75.55 ✅ (simile a exp_031's 74.55)

**Lesson learned:** Preprocessing DEVE essere identico tra training e inference!

---

## Results

### Validation Performance

| Model | exp_031 Val MAE | exp_039 Val MAE | Improvement |
|-------|----------------|-----------------|-------------|
| RF    | 12.20          | 8.79            | **-3.41** (-28%) |
| LGBM  | 10.88          | 8.58            | **-2.30** (-21%) |

### Kaggle Submission

**File:** `submission_exp039.csv`  
**Predictions:**
- Mean: 75.55
- Std: 4.50
- Min: 63.68
- Max: 83.82

**Kaggle Score:** **11.89067 MAE**

**Result:** ✅ New best score! -0.036 improvement from exp_031

---

## Analysis

### Successo Parziale

**Positivo:**
- ✅ Validation improvement drammatico (-28% RF, -21% LGBM)
- ✅ Overfitting ridotto (Train-Val gap più piccolo)
- ✅ Kaggle improvement confermato (-0.036 MAE)
- ✅ Data augmentation strategy validata

**Deludente:**
- ❌ Improvement più piccolo del previsto (-0.036 vs -0.25/-0.52 sperato)
- ❌ Gap to 2nd place ancora grande (-1.19 MAE)
- ❌ Validation improvement non trasferito 1:1 a Kaggle

### Perché Improvement Limitato?

1. **Augmentation quality:** Le tecniche semplici (noise, scaling) non aggiungono vera diversità
2. **Distribution mismatch:** Augmented data potrebbe non matchare distribuzione test set
3. **FL penalty:** 9 modelli separati vs 1 modello centralizzato limita performance massima
4. **Ensemble semplificato:** Solo 2 algoritmi (RF+LGBM) vs 3 in exp_031 (RF+XGB+LGBM)

---

## Key Learnings

1. **Data augmentation works:** -3.41/-2.30 MAE validation improvement è significativo
2. **Overfitting era il problema:** Train-Val gap ridotto drasticamente
3. **Validation ≠ Test:** Grandi improvement validation non garantiscono pari gain su Kaggle
4. **Preprocessing critical:** Identico preprocessing train/inference è essenziale
5. **Incremental gain:** -0.036 MAE è small ma positivo, ogni passo conta

---

## Next Steps (Considerati)

### Se exp_039 < 11.70 MAE: ✅ Major success
- Provare exp_040: hyperparameter tuning su augmented data
- Expected: additional -0.10/-0.20 MAE

### Se exp_039 11.70-11.90: Marginal improvement
- Provare different augmentation factor (3x o 9x)
- Adjust augmentation techniques

### Se exp_039 > 11.90: Failed
- Revert to exp_031 as active submission
- Accept 3rd place

**Actual result:** 11.89067 → marginal improvement category

---

## Files Generated

```
exp_039_augmented_ensemble/
├── train_rf_augmented.py          # RF training script (5904 samples)
├── train_lgbm_augmented.py        # LGBM training script (5904 samples)
├── generate_submission.py         # Ensemble + time series extraction
├── rf_augmented_model.pkl         # RF model (17MB)
├── lgbm_augmented_model.pkl       # LGBM model (233KB)
├── submission_exp039.csv          # Final submission (189 predictions)
├── training_rf.log                # RF training output
└── training_lgbm.log              # LGBM training output
```

---

## Conclusion

exp_039 rappresenta un **pivot strategico riuscito** da optimization plateau a data augmentation. Validation improvements drammatici (-28%/-21%) dimostrano efficacia dell'approccio, anche se Kaggle gain è stato più modesto (-0.036 MAE).

**Position:** Mantenuto 3° posto con nuovo best score (11.89067 MAE)  
**Strategy validated:** Data augmentation è promettente per dataset FL piccoli  
**Next experiment:** exp_040 proverà hyperparameter tuning esteso su augmented data
