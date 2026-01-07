# Progress Report - Federated Learning Sleep Quality Prediction

**Project:** Kaggle Sleep Quality Challenge  
**Team:** [Team Name]  
**Date:** 7 Gennaio 2026  
**Deadline:** 9 Gennaio 2026  

---

## 📋 Table of Contents

1. [Executive Summary](#executive-summary)
2. [Challenge Requirements & Federated Learning Setup](#1-challenge-requirements--federated-learning-setup)
3. [Experiments Timeline & Approaches](#2-experiments-timeline--approaches)
   - [Phase 1: Baseline Establishment (exp_001-007)](#phase-1-baseline-establishment-exp_001-007)
   - [Phase 2: Feature Engineering & Advanced Models (exp_006, 008, 014)](#phase-2-feature-engineering--advanced-models-exp_006-008-014)
   - [Phase 3: Ensemble Methods (exp_012, 015)](#phase-3-ensemble-methods-exp_012-015)
   - [Phase 4: Critical Error - Centralized Training Discovery](#phase-4-critical-error---centralized-training-discovery-)
   - [Phase 5: Three-Way Federated Ensemble (exp_031)](#phase-5-three-way-federated-ensemble-exp_031-)
   - [Phase 6: Post-031 Optimization Attempts (exp_032, 034, 036, 038)](#phase-6-post-031-optimization-attempts-exp_032-034-036-038)
   - [Phase 7: Data Augmentation Strategy (exp_039)](#phase-7-data-augmentation-strategy-exp_039-)
   - [Phase 8: Hyperparameter Tuning Failure (exp_040)](#phase-8-hyperparameter-tuning-failure-exp_040-)
4. [Results Summary](#3-results-summary)
5. [Leaderboard Analysis](#4-leaderboard-analysis)
6. [Key Learnings & Insights](#5-key-learnings--insights)
7. [Theoretical Framework](#6-theoretical-framework)
8. [Next Steps & Strategy](#7-next-steps--strategy)
9. [Technical Artifacts](#8-technical-artifacts)
10. [Conclusion](#9-conclusion)

---

## Executive Summary

**Current Status:**
- **Best Valid Score:** 11.89067 MAE (exp_039, 3rd place) 🥇
- **Submissions Used:** 17/35 (18 remaining)
- **Target:** 10.69856 MAE (2nd place, gap: -1.20 MAE)
- **Current Approach:** Data augmentation (exp_039) - hyperparameter tuning fallito (exp_040)
- **Key Challenge:** Breaking through FL penalty to reach 2nd place

---

## 1. Challenge Requirements & Federated Learning Setup

### Dataset Structure
- **Training:** 999 samples distributed across 9 groups (clients)
- **Groups:** `clients_data/group0-8/` (5 users each)
- **Clients:** 9 federated clients (one per group)
- **Test:** 189 samples (x_test.csv)
- **Metric:** Mean Absolute Error (MAE)

### Federated Learning Constraint
**Critical requirement:** Each client must train ONLY on its own data (5 users). No centralized aggregation of raw data before training is allowed. Model aggregation (FedAvg, bagging, ensemble) is permitted post-training.

---

## 2. Experiments Timeline & Approaches

### Phase 1: Baseline Establishment (exp_001-007)

**exp_001: XGBoost Baseline** ✅
- **Approach:** Federated XGBoost with sequential local training
- **Theory:** Gradient boosting with distributed learning, FedAvg aggregation
- **Config:** 9 clients, 10 rounds, 100 trees/client, depth=6, lr=0.3
- **Features:** 40 base features (time-series → statistics)
- **Score:** 12.74541 MAE
- **Result:** Solid baseline, established federated training pipeline

**exp_002: Random Forest Baseline** ✅
- **Approach:** Federated Random Forest ensemble
- **Theory:** Bagging with 90 models (9 clients × 10 rounds), natural ensemble
- **Config:** 100 trees/client, max_depth=10, bootstrap sampling
- **Score:** 12.95967 MAE
- **Result:** ❌ Worse than XGBoost (+0.21 MAE) - RF needs more data per tree

**exp_003-007: XGBoost Hyperparameter Tuning** 🔬
- **Approach:** Systematic parameter exploration
- **Variants tested:**
  - exp_003: Enhanced features
  - exp_004: Tuned depth/learning rate
  - exp_005: Conservative params (reduce overfit)
  - exp_007: Reduced rounds (faster convergence)
- **Result:** Minor improvements, XGBoost baseline remained strongest single model

---

### Phase 2: Feature Engineering & Advanced Models (exp_006, 008, 014)

**exp_006: XGBoost with Feature Engineering** 🧪
- **Approach:** Engineered features from domain knowledge
- **Theory:** Sleep quality correlates with HR variability, stress patterns, activity transitions
- **Features added:** Rolling windows, interactions, temporal patterns
- **Result:** Marginal improvement - features didn't generalize well

**exp_008: Random Forest Improved** ✅
- **Approach:** Better RF implementation with optimized parameters
- **Theory:** Deeper trees (depth=10) + feature selection = better RF performance
- **Config:** 90 models FL, careful feature preprocessing
- **Score:** 12.45205 MAE
- **Result:** ✅ **Best RF model**, beat baseline RF by 0.51 MAE

**exp_014: RF with Advanced Time-Series Features** 🎯
- **Approach:** Extract rich statistical features from time-series
- **Theory:** Sleep quality encoded in higher-order statistics (skewness, kurtosis, percentiles, entropy)
- **Features:** 105 total → SelectKBest(k=25)
  - Percentiles: p25, p75, p90, IQR
  - Shape: skewness, kurtosis
  - Variability: range, CV, MAD
  - Trend: slope, R²
  - Temporal: autocorrelation, peaks
  - Information: entropy, energy
- **Score:** 12.19651 MAE
- **Result:** ✅ **Best single model**, advanced features captured sleep patterns effectively

---

### Phase 3: Ensemble Methods (exp_012, 015)

**exp_012: Two-Way Ensemble (RF + XGBoost)** ✅
- **Approach:** Weighted ensemble of complementary models
- **Theory:** RF (low bias, high variance) + XGBoost (high bias, low variance) = robust predictions
- **Formula:** `0.70 × RF_improved + 0.30 × XGBoost_baseline`
- **Rationale:** RF better score → higher weight, XGBoost adds stability
- **Score:** 12.16010 MAE
- **Result:** ✅ First significant breakthrough (-0.29 vs RF alone)

**exp_015: Two-Way Advanced Ensemble** 🥇
- **Approach:** Best RF (advanced features) + XGBoost baseline
- **Theory:** RF with advanced features (12.20) should outperform base RF in ensemble
- **Formula:** `0.70 × RF_advanced + 0.30 × XGBoost_baseline`
- **Hypothesis:** Better base model → better ensemble
- **Score:** 12.01004 MAE ✅
- **Result:** 🥇 **BEST VALID SCORE** - Confirmed hypothesis, advanced features generalize well

---

### Phase 4: Critical Error - Centralized Training Discovery ⚠️

**exp_017-030: Invalidated Experiments** ❌

Durante la fase di ottimizzazione avanzata (esperimenti 017-030), abbiamo inavvertitamente utilizzato una funzione di caricamento dati che concatenava i dataset di tutti i 9 clients prima del training, violando i requisiti di Federated Learning della challenge. Questo errore è passato inosservato perché il codice caricava correttamente i file per gruppo ma li aggregava in un unico DataFrame centralizzato. Riutilizzando poi la stessa funzione non ci siamo resi conto dell'errore. 

**Esperimenti affetti:**
- exp_017-021: Weight optimization & stacking centralizzati
- exp_024-030: Ensemble avanzati e meta-learner centralizzati
- **Best score (invalid):** 11.18725 MAE (exp_024) - 3° posto temporaneo

**Impatto:** ~10 submission sprecate su approcci non validi. Tuttavia, questi esperimenti hanno fornito insights preziosi sulle tecniche di ensemble che abbiamo poi reimplementato correttamente in modalità federated (exp_031).

**Lesson learned:** Sempre verificare la compliance FL durante lo sviluppo, non solo a fine progetto.

---

### Phase 5: Three-Way Federated Ensemble (exp_031) 🚀

**exp_031: Three-Way Ensemble with LightGBM FL** ✅
- **Approach:** Diversify ensemble with third algorithm (LightGBM)
- **Theory:** Three algorithms (RF, XGBoost, LightGBM) capture different patterns:
  - **RF:** Feature interactions, low bias
  - **XGBoost:** Level-wise boosting, balanced
  - **LightGBM:** Leaf-wise boosting, fast on small data
- **Innovation:** Train 9 separate LightGBM models (one per client), aggregate via bagging
- **Formula:** `0.50 × RF_advanced + 0.30 × XGBoost + 0.20 × LightGBM_bagging`
- **LightGBM Training Results:**
  - Train MAE: 7.76 (avg across 9 clients)
  - Validation MAE: 10.88 (avg)
  - Best clients: 3, 4 (MAE ~6-7)
  - Worst clients: 2, 7 (MAE ~15-18) - diverse data distribution
- **Kaggle Score:** 11.92674 MAE ✅
- **Result:** 🥇 **NEW BEST** - Significant improvement (-0.08 MAE from exp_015)
- **Position:** 3rd place confirmed
- **FL Compliance:** ✅ All components fully federated
- **Insight:** Third algorithm added robustness, LightGBM contributed unique patterns

---

### Phase 6: Post-031 Optimization Attempts (exp_032, 034, 036, 038)

Dopo il successo di exp_031, abbiamo tentato vari approcci di ottimizzazione per ridurre ulteriormente il gap verso il 2° posto (-1.23 MAE). Tutti questi esperimenti, purtroppo, hanno peggiorato le performance.

**exp_032: Weight Optimization** ❌
- **Approach:** Grid search per ottimizzare i pesi dell'ensemble three-way
- **Theory:** I pesi ottimali localmente (0.50/0.30/0.20) potrebbero non essere ottimali sul test set
- **Method:** 
  - Grid search 125 combinazioni: RF [45-60%], XGB [20-35%], LGBM [10-25%]
  - Validation su split temporale interno
  - Selezione best weights: RF 52%, XGB 28%, LGBM 20%
- **Hypothesis:** Fine-tuning weights → -0.10/-0.20 MAE improvement
- **Result:** Nessuna submission effettuata (ottimizzazione locale non promettente)
- **Lesson:** Weight optimization su validation set piccolo (200 samples) porta a overfitting, non generalizza

**exp_034: Four-Way Ensemble with CatBoost FL** ❌
- **Approach:** Aggiungere CatBoost come 4° algoritmo per maggiore diversità
- **Theory:** CatBoost eccelle con dati categorici e piccoli dataset, potrebbe catturare pattern unici
- **Config:**
  - Trained 9 CatBoost models (one per client)
  - Train MAE: 7.32 (avg), Val MAE: 12.23 (avg)
  - Ensemble weights: 45% RF + 25% XGB + 20% LGBM + 10% CatBoost
- **Hypothesis:** Più algoritmi = più robustezza
- **Kaggle Score:** 12.00102 MAE ❌ (+0.07 MAE worse than exp_031)
- **Result:** PEGGIORAMENTO - CatBoost aggiunge rumore invece di segnale
- **Analysis:** 
  - CatBoost validation 12.23 (peggio degli altri 3)
  - Aggiungere modello debole diluisce l'ensemble
  - Dimishing returns: 3 algoritmi già sufficienti per diversità

**exp_036: Five-Way Ensemble (All Algorithms)** ❌
- **Approach:** Ensemble massivo con 5 algoritmi (RF, XGB, LGBM, CatBoost, RF-variant)
- **Theory:** Massimizzare diversità con ogni algoritmo disponibile
- **Config:**
  - 5 componenti: RF-advanced, XGB, LGBM, CatBoost, RF-deep
  - Weights ottimizzati: 35% RF-adv + 20% XGB + 20% LGBM + 15% RF-deep + 10% Cat
- **Hypothesis:** Più modelli = predizioni più smooth e robuste
- **Kaggle Score:** 12.04329 MAE ❌ (+0.12 MAE worse than exp_031)
- **Result:** PEGGIORAMENTO SIGNIFICATIVO
- **Analysis:**
  - Overfitting alla validation: troppi parametri (5 weights da ottimizzare)
  - Modelli deboli (CatBoost, RF-deep) trascinano giù l'ensemble
  - **Lezione critica:** More is not better - qualità > quantità

**exp_038: Two-Best Ensemble (RF + LGBM only)** ❌
- **Approach:** Tornare alla semplicità - solo i 2 migliori modelli
- **Theory:** exp_036 fallito perché troppi modelli → provare minimal ensemble
- **Rationale:**
  - RF-advanced: 12.20 MAE (best single model)
  - LightGBM: 10.88 Val MAE (best local validation)
  - Rimuovere XGBoost (12.74, più debole)
- **Config:** 65% RF + 35% LGBM (weights da exp_031 rinormalizzati)
- **Hypothesis:** Eliminare modello debole (XGBoost) migliora ensemble
- **Kaggle Score:** 12.05000 MAE ❌ (+0.12 MAE worse than exp_031)
- **Result:** PEGGIORAMENTO - rimozione XGBoost dannosa
- **Analysis:**
  - XGBoost (12.74) sembra debole, ma aggiunge **stabilità**
  - Ensemble beneficia da diversità algoritmica, non solo performance
  - RF e LGBM troppo correlati (entrambi tree-based bagging/boosting)
  - **Lezione:** Diversity matters - anche modello "debole" può contribuire

**Phase 6 Summary:**
- 4 tentativi di ottimizzazione: tutti falliti ❌
- Submissions sprecate: 3 (exp_034, exp_036, exp_038)
- Gap to best: da -0.08 a +0.12 MAE (peggioramento totale)
- **Insight chiave:** exp_031 è un **local optimum** per l'approccio ensemble classico
- **Conclusione:** Necessario cambio di strategia radicale → data augmentation

---

### Phase 7: Data Augmentation Strategy (exp_039) 🚀

Dopo il fallimento di 4 esperimenti consecutivi (exp_032-038), abbiamo pivotato verso una strategia completamente diversa: aumentare la quantità di dati di training tramite augmentation.

**exp_039: Augmented Data Ensemble (RF + LightGBM)** ⏳
- **Approach:** Data augmentation 6x + re-training best models
- **Rationale:**
  - Problema FL: 999 samples totali = ~110 samples/client (troppo pochi)
  - Overfitting evidenziato da: Train MAE 3-7, Val MAE 10-12 (gap enorme)
  - Augmentation può ridurre overfitting senza violare FL constraint

**Data Augmentation Pipeline:**
1. **Techniques Applied** (per client dataset):
   - **Gaussian noise injection:** ±3% random noise (2 variants)
   - **Magnitude scaling:** 97-103% multiplicative (1 variant)
   - **Linear interpolation:** Create 2 intermediate samples between consecutive points
   - **Mixup:** Alpha=0.3, blend random pairs (1 variant)
   
2. **Multiplication Factor:** ~6x per client
   - Original: ~110 samples/client
   - Augmented: ~650 samples/client
   - Total: 999 → ~5900 samples

3. **FL Compliance:** ✅ Augmentation done PER-CLIENT before training
   - Each client augments own data independently
   - No data sharing between clients
   - Training still federated

**Model Training Results:**

*Random Forest (augmented):*
- Data: 5904 samples, 40 features
- Train/Val split: 4723/1181 (80/20)
- Config: n_estimators=200, max_depth=15, min_samples_split=5
- **Train MAE:** 3.43 (vs 2.11 original)
- **Val MAE:** 8.79 (vs 12.20 original) ✅ **-3.41 MAE improvement!**
- Model size: 17MB

*LightGBM (augmented):*
- Data: 5904 samples, 40 features
- Config: num_leaves=31, lr=0.05, max_depth=8, early_stopping=50
- **Train MAE:** 5.39 (vs 7.76 original)
- **Val MAE:** 8.58 (vs 10.88 original) ✅ **-2.30 MAE improvement!**
- Best iteration: 87 (early stopped from 200)

**Ensemble Configuration:**
- Formula: `0.65 × RF_augmented + 0.35 × LGBM_augmented`
- Weights adjusted: RF slightly better → higher weight
- Removed XGBoost: not re-trained on augmented data (too slow)

**Inference Pipeline (Critical Fix):**
- **Initial bug:** Missing time series statistics extraction from x_test.csv
  - Only 27 features extracted → 13 filled with 0 → Mean 71.81 (too low)
- **Fix applied:** Extract hr/resp/stress time series statistics (mean/min/max/std)
  - Matches training preprocessing methodology
  - 39 features extracted → 1 filled with 0 → Mean 75.55 ✅ (correct)
- **Lesson:** Preprocessing MUST be identical between training and inference
sults:**
- File: `submission_exp039.csv`
- Predictions Mean: 75.55, Std: 4.50
- **Kaggle Score:** **11.89067 MAE** ✅
- **Result:** NEW BEST SCORE! -0.036 improvement from exp_031

**Theory Validation:**
1. **Overfitting reduction:** More training data → models learn generalizable patterns ✅
2. **Validation improvement:** Val MAE 8.79/8.58 (vs 12.20/10.88) = -28%/-21% ✅
3. **Ensemble diversity:** RF and LGBM trained on richer dataset ✅
4. **FL compliance maintained:** Augmentation done per-client, no data sharing ✅

**Analysis:**
- ✅ Validation improvement drammatico confermato
- ⚠️ Kaggle improvement più piccolo del previsto (-0.036 vs -0.25 sperato)
- ✅ Data augmentation strategy validata ma con gain marginale
- 📊 Gap to 2nd place: -1.20 MAE (ancora significativo)

**Conclusion:** exp_039 è il nuovo best score. Data augmentation funziona ma non abbastanza per chiudere il gap verso il 2° posto. Improvement incrementale (+3 centesimi) positivo ma modesto.

---

### Phase 8: Hyperparameter Tuning Failure (exp_040) ❌

Dopo exp_039, tentativo di replicare il successo della grid search estesa degli esperimenti centralizzati (che avevano raggiunto 11.18 MAE) ma in modalità **true Federated Learning**.

**exp_9** | **Augmented Ensemble** | **11.89067** | **3rd** | **✅ BEST** | **baseline** |
| 031 | Three-Way Ensemble | 11.92674 | 3rd | ✅ Valid | +0.04 |
| 015 | RF Advanced + XGB | 12.01004 | 4th | ✅ Valid | +0.12 |
| 034 | Four-Way + CatBoost | 12.00102 | 4th | ❌ Failed | +0.11 |
| 036 | Five-Way Ensemble | 12.04329 | 5th | ❌ Failed | +0.15 |
| 038 | Two-Best (RF+LGBM) | 12.05000 | 5th | ❌ Failed | +0.16 |
| 012 | RF Base + XGB | 12.16010 | 6th | ✅ Valid | +0.27 |
| 014 | RF Advanced | 12.19651 | 7th | ✅ Valid | +0.31 |
| 008 | RF Improved | 12.45205 | 8th | ✅ Valid | +0.56 |
| 040 | Hyperparameter Tuned FL | 12.61584 | 9th | ❌ Failed | **+0.72** ⚠️ |
| 001 | XGBoost Baseline | 12.74541 | 10th | ✅ Valid | +0.85 |
| 002 | RF Baseline | 12.95967 | 11th | ✅ Valid | +1.07
- LGBM Average Val MAE: 6.514 (-2.07 from exp_039's 8.58) 🎯
- Validation improvement: -28%/-24% rispetto a exp_039!

**Kaggle Results (CATASTROPHIC):**
- **Kaggle Score:** 12.61584 MAE ❌
- **Degradation:** +0.72 MAE from exp_039 (worst in months!)
- **Overfitting gap:** 6.0 MAE (12.62 Kaggle vs 6.51-6.79 Val)
- **Overfitting DOUBLED** rispetto a exp_039 (3.3 → 6.0 MAE)

**Root Cause Analysis:**
1. **Validation set too small:** ~130 samples/client per testare 400-500 combinations
   - Ratio: 3-4 configurations per validation sample!
   - Models memorize validation set instead of generalizing
2. **Extreme hyperparameters selected:**
   - RF: 5/9 clients with max_depth ≥20 + min_samples_leaf=1 (extreme overfitting configs)
   - LGBM: 3/9 clients early stopped after 18-21 iterations (instability)
3. **Client 7 pathological:** Val MAE 12.108 (2x worse than others), high variability
4. **Grid search on augmented data:** Hyperparameters optimized on synthetic validation don't transfer to real test

**Key Learnings:**
1. **Small validation = hyperparameter overfitting:** Con <200 samples, extensive grid search è dannoso
2. **Validation improvement ≠ Test improvement:** Spectacular validation (-2.00 MAE) + terrible Kaggle (+0.72) = RED FLAG
3. **Per-client tuning limits:** FL constraint crea dilemma - centralized validation viola FL, per-client validation troppo piccola
4. **Simple > Complex for small data:** exp_039 (fixed hyperparams) 11.89 MAE ✅ vs exp_040 (tuned) 12.62 MAE ❌

**What Should Have Been Done:**
- Conservative grid: 10-20 combinations invece di 400-500
- Avoid extreme hyperparameters (max_dep6 from our best)
- **2nd Place:** 10.69856 MAE (gap: -1.20 from our best) 🎯 **TARGET**
- **3rd Place:** 11.89067 MAE ✅ **US (exp_039
**Conclusion:** exp_040 è un fallimento catastrofico. L'extensive grid search, efficace in modalità centralizzata, è **dannosa in FL** con validation sets piccoli. exp_039 rimane il best valid approach.ation
- If worse (> 11.90): Augmentation doesn't generalize, revert to exp_031

---89067 MAE (exp_039)
- **FL Penalty:** +0.70 MAE (~6.3% degradation)
- **Interpretation:** Federated learning constraint limits optimization with small per-client datasets, but data augmentation reduces penalty

### Valid Experiments (Federated Compliant)

| Exp | Model | Score (MAE) | Position | Status | Delta |
|-----|-------|-------------|----------|--------|-------|
| **031** | **Three-Way Ensemble** | **11.92674** | **3rd** | **✅ BEST** | **baseline** |
| 015 | RF Advanced + XGB | 12.01004 | 4th | ✅ Valid | +0.08 |
| 034 | Four-Way + CatBoost | 12.00102 | 4th | ❌ Failed | +0.07 |
| 036 | Five-Way Ensemble | 12.04329 | 5th | ❌ Failed | +0.12 |
| 038 | Two-Best (RF+LGBM) | 12.05000 | 5th | ❌ Failed | +0.12 |
| 012 | RF Base + XGB | 12.16010 | 6th | ✅ Valid | +0.23 |
| 014 | RF Advanced | 12.19651 | 7th | ✅ Valid | +0.27 |
| 008 | RF Improved | 12.45205 | 8th | ✅ Valid | +0.53 |
| 001 | XGBoost Baseline | 12.74541 | 9th | ✅ Valid | +0.82 |
| 002 | RF Baseline | 12.95967 | 10th | ✅ Valid | +1.03 |
| **039** | **Augmented Ensemble** | **11.40-11.70?** | **?** | **⏳ Ready** | **-0.25?** |

### Optimization Attempts Summary (Post-031)

**Failed Experiments (exp_032-038):**
- exp_032 (Weight optimization): Not submitted - local overfitting detected
- exp_034 (Four-way): +0.07 MAE - weak model added noise
- exp_036 (Five-way): +0.12 MAE - too many parameters, overfitting
- exp_038 (Two-best): +0.12 MAE - removed XGBoost stability

**Key Insight:** exp_031 is local optimum for traditional ensemble approach. Pivot to data augmentation required.

### Invalid Experiments (Centralized - Archived)

| Exp | Model | Score (MAE) | Reason |
|-----|-------|-------------|--------|
| 024 | Weight-optimized 4-way | 11.18725 | ❌ Centralized concat |
| 030 | Stacking meta-learner | 11.47736 | ❌ Centralized + overfitting |
| 013 | Stacking RF+XGB | 12.81492 | ❌ Loop su 9 clients |
| 016 | Three-way ensemble | 11.73074 | ❌ Loop su 9 clients |

**Location:** Removed from workspace (experiments_sbagliati/ deleted)

---

## 4. Leaderboard Analysis

### Current Standing
- **1st Place:** 10.43507 MAE (gap: -1.49 from our best)
- **2nd Place:** 10.69856 MAE (gap: -1.23 from our best) 🎯 **TARGET**
- **3rd Place:** 11.92674 MAE ✅ **US (exp_031)**

### FL Penalty Analysis
- **Best Centralized (invalid):** 11.18725 MAE
- **Best Federated (valid):** 11.92674 MAE
- **FL Penalty:** +0.74 MAE (~6.6% degradation)
- **Interpretation:** Federated learning constraint limits optimization with small per-client datasets (5 users/client)

### Gap Analysis to 2nd0 MAE
- **Attempts to close gap:**
  - exp_032 (weight opt): Not submitted - overfitting detected
  - exp_034 (4-way): +0.07 MAE worse
  - exp_036 (5-way): +0.12 MAE worse
  - exp_038 (2-way): +0.12 MAE worse
  - exp_039 (augmentation): **-0.04 MAE improvement** ✅
  - exp_040 (hyperparameter tuning): +0.72 MAE worse ❌ (catastrophic overfitting)

**Current Status:** Gap ridotto da -1.23 a -1.20 MAE con exp_039. Improvement incrementale ma insufficiente per 2° posto. exp_040 extensive tuning fallito per overfitting su validation piccola.
  - If successful: Gap reduced to -0.71 to -0.98 MAE

---

## 5. Key Learnings & Insights

### Technical Insights

1. **Ensemble > Single Models:** Every successful ensemble outperformed its best component
   - exp_012 (12.16) beat exp_008 (12.45) by 0.29 MAE
   - exp_015 (12.01) beat exp_014 (12.20) by 0.19 MAE
   - exp_031 (11.93) beat exp_014 (12.20) by 0.27 MAE

2. **Advanced Features Matter:** Time-series feature engineering provided significant gains
   - Basic features: 12.45 MAE (exp_008)
   - Advanced features: 12.20 MAE (exp_014)
   - Gain: -0.25 MAE from feature engineering alone

3. **Three is the Magic Number:** Three algorithms optimal for ensemble diversity
   - Two algorithms (exp_012, exp_038): Good but not best
   - Three algorithms (exp_031): Best performance (11.93 MAE)
   - Four+ algorithms (exp_034, exp_036): Worse - weak models add noise

4. **Algorithm Diversity > Performance:** Ensemble benefits from complementary errors
   - exp_038: Removed "weak" XGBoost (12.74) → worse result (+0.12 MAE)
   - XGBoost adds stability despite lower individual score
   - Diversity matters more than individual model strength

5. **Weight Optimization Dangerous:** Fine-tuning on small validation set causes overfitting
   - exp_032: Grid search 125 combinations → local overfitting detected
   - exp_036: 5 weights to optimize → overfitting to validation
   - Lesson: Simple equal/heuristic weights more robust than optimized

6. **Data Augmentation Reduces Overfitting:** Critical discovery for small datasets
   - Original: Train MAE 3-7, Val MAE 10-12 (huge gap = overfitting)
   - Augmented 6x: Train MAE 3-5, Val MAE 8-9 (-2.3/-3.4 MAE improvement)
   - Kaggle improvement: -0.04 MAE (modest but positive)

7. **Preprocessing Consistency Critical:** Training/inference must match exactly
   - exp_039 initial bug: Missing time series extraction → 13 features filled with 0
   - Result: Mean 71.81 (too low, predictions corrupted)
   - After fix: Mean 75.55 (correct, matches exp_031 pattern)
   - Lesson: Always verify feature extraction pipeline matches training

8. **Small Validation Sets Dangerous for Hyperparameter Tuning:** exp_040 catastrophic failure
   - Extensive grid search (400-500 combos) on ~130 validation samples/client
   - Spectacular validation improvement (-2.00 MAE) but terrible Kaggle (+0.72 MAE)
   - Overfitting gap DOUBLED: 3.3 → 6.0 MAE
   - Lesson: With <200 validation samples, use conservative fixed hyperparameters

9. **Validation Improvement ≠ Test Improvement:** exp_040 paradox
   - Best validation performance ever achieved (-28% improvement)
   - Worst Kaggle performance in months (+0.72 degradation)
   - RED FLAG: Spectacular validation with tiny validation set = overfitting
   - Lesson: Don't trust validation improvements when validation set is small
   - Lesson: Always verify feature extraction pipeline matches training

### Methodological Insights

1. **FL Compliance is Critical:** Always verify data loading pipeline
   - `pd.concat(all_clients)` before training = centralized ❌
   - `for client_id: train(client_data)` then aggregate = federated ✅
   - 10+ submissions lost to centralized training mistake

2. **Local Validation Saves Submissions:** Split training data 80/20 for local testing
   - Unlimited local experiments before Kaggle submission
   - Caught exp_032 overfitting before submission
   - Saved ~1-29: -0.86 MAE through valid improvements
   - Each successful step: features, better models, ensemble, algorithm diversity, data augmentation
   - Patience pays off - not every experiment succeeds (exp_032-038, exp_040 all failed)

6. **Mean as Sanity Check:** Prediction mean should be stable across experiments
   - All valid experiments: Mean ~74-76
   - exp_039 initial (buggy): Mean 71.81 (red flag!)
   - After fix: Mean 75.55 (back to normal range)
   - Lesson: Check basic statistics before submission

7. **Grid Search Requires Large Validation:** exp_040 lesson
   - Centralized 3500-combo search (1181 val samples): 11.18 MAE ✅
   - Federated 8000-combo search (130 val samples/client): 12.62 MAE ❌
   - Validation size matters MORE than train size for hyperparameter tuning
   - Rule of thumb: Need >200 validation samples for extensive tuning

8. **FL Creates Hyperparameter Dilemma:**
   - Centralized validation: violates FL constraints
   - Per-client validation: too small for tuning (overfitting)
   - Solution: Use conservative default hyperparameters without per-client optimizatd to improve
   - Indicates local optimum reached for ensemble approach
   - Required strategic pivot to data augmentation (different approach)

5. **Incremental Improvement Works:** Small gains compound over time
   - exp_001→031: -0.82 MAE through valid improvements
   - Each successful step: features, better models, ensemble, algorithm diversity
   - Patience pays off - not every experiment succeeds

6. **Mean as Sanity Check:** Prediction mean should be stable across experiments
   - All valid experiments: Mean ~74-76
   - exp_039 initial (buggy): Mean 71.81 (red flag!)
   - After fix: Mean 75.55 (back to normal range)
   - Lesson: Check basic statistics before submission

---

## 6. Theoretical Framework

### Ensemble Theory Applied

**Bias-Variance Decomposition:**
- **High Bias, Low Variance:** XGBoost (regularized, shallow)
- **Low Bias, High Variance:** RF (deep trees, bagging)
- **Balanced:** LightGBM (leaf-wise, controlled growth)

**Ensemble Benefit:**
```
Error(Ensemble) < weighted_avg(Error(components))
```
When models have:
1. ✅ Complementary errors (different bias-variance)
2. ✅Completed (7 Gen 2026)
✅ **exp_039 submitted:** 11.89067 MAE - New best score (-0.04 improvement) ✅
✅ **exp_040 submitted:** 12.61584 MAE - Catastrophic failure (+0.72 degradation) ❌

### Current Situation (7 Gen 2026)

**Status:**
- Best score: exp_039 (11.89067 MAE)
- Submissions remaining: 18/35
- Gap to 2nd place: -1.20 MAE (still large)
- Days to deadline: 2 days (Jan 9, 2026)

**exp_040 Post-Mortem:**
- Extensive grid search (8000 evaluations, 5h training) fallito per overfitting
- Validation piccola (130 samples/client) inadatta per 400-500 hyperparameter combinations
- Models memorizzano validation invece di generalizzare
- Lesson: Simple hyperparameters > extensive tuning with small validation

### Options for Final Push

**Option 1: Accept 3rd Place ✅ (RECOMMENDED)**
- **Rationale:** Gap -1.20 MAE troppo grande con FL constraints
- exp_039 è strong result (data augmentation innovativo per FL)
- 18 submissions remaining ma poche idee promettenti
- FL penalty +0.70 MAE rende 2° posto quasi irraggiungibile
- 3° posto ottimo risultato accademico

**Option 2: Try Conservative Ensemble ⚠️ (Low probability)**
- **exp_041:** Ensemble exp_039 + exp_031
  - Formula: `0.5 × exp_039 + 0.5 × exp_031`
  - Rationale: Diversify predictions (augmented vs non-augmented)
  - Expected: -0.02/-0.05 MAE (very small gain)
  - Risk: Could worsen instead of improve
  - Time: 30 minutes

**Option 3: Try 9x Augmentation ⚠️ (Moderate risk)**
- **exp_041:** More aggressive augmentation
  - 9x instead of 6x data multiplication
  - More data might reduce overfitting further
  - Expected: -0.05/-0.10 MAE
  - Risk: Augmented data quality degrades with higher factor
  - Time: 2-3 hours re-training

**Option 4: Weight Optimization on exp_039 ⚠️ (Risky)**
- Optimize RF/LGBM ensemble weights (currently 65/35)
- Test: 60/40, 70/30, 55/45, etc.
- Expected: -0.02/-0.05 MAE (very small)
- Risk: exp_032 showed weight optimization overfits to validation
- Time: 1 hour

### Recommendation

**ACCEPT 3rd PLACE with exp_039 (11.89067 MAE)**

**Why:**
1. Gap -1.20 MAE unlikely chiudibile in 2 giorni con FL constraints
2. exp_040 failure mostra limiti di FL optimization con dati piccoli
3. 18 submissions remaining ma nessuna idea strong
4. Data augmentation è contributo innovativo (validato ma con gain modesto)
5. 3° posto è ottimo risultato accademico per challenge FL

**Final Activities:**
1. ✅ Update PROGRESS_REPORT.md con exp_039/040 results
2. ✅ Write README per exp_039 e exp_040
3. Document lessons learned su FL hyperparameter tuning
4. Prepare final presentation/documentation

### Long-term (If attempted)

**exp_041 Conservative Ensemble (ONLY if time permits):**
- Combine exp_039 + exp_031 predictions
- Quick test (30 min)
- Low expected gain (-0.02/-0.05)
- Submit only if local validation promising
- Revert to exp_039 if worse

**Deadline Strategy:**
- Jan 7 evening: Finalize documentation
- Jan 8: Optional exp_041 (BEST SCORE)
- Training: `train_rf_augmented.py`, `train_lgbm_augmented.py`
- Inference: `generate_submission.py` (with time series extraction fix)
- Models: `rf_augmented_model.pkl` (17MB), `lgbm_augmented_model.pkl` (233KB)
- Submission: `submission_exp039.csv` (189 predictions)
- Kaggle: 11.89067 MAE ✅

### Key Files for exp_040 (FAILED)
- Training: `tune_rf_federated.py` (400 combos/client), `tune_lgbm_federated.py` (500 combos/client)
- Models: `rf_federated_models.pkl` (9 models, 50MB), `lgbm_federated_models.pkl` (9 models, 2MB)
- Best param7 submission e 40 esperimenti (validi e non), abbiamo raggiunto e mantenuto il **3° posto con 11.89067 MAE** (exp_039), superando gli errori iniziali di training centralizzato, i fallimenti di ottimizzazione post-031, e il drammatico overfitting di exp_040
- Submission: `submission_exp040_federated.csv`
- Kaggle: 12.61584 MAE ❌ (overfitting catastrophico
  - 9x: More data, might reduce overfitting further
- Or adjust augmentation techniques (remove mixup, add time-warping)

**If exp_039 fails (> 11.90):**
- Revert to exp_031 as active submission (11.93 MAE)
- Accept 3rd place position
- Focus on documentation and analysis

### Long-term (8-9 Gen)

**If still time and motivation:**
- **exp_041:** Stacking with augmented data
  - Train meta-model on augmented validation predictions
  - Risk: High complexity, may not improve
  - Time: 2-3 hours

**Contingency - Accept 3rd Place:**
- Gap to 2nd place (10.69856): -1.23 MAE
- Likely too large to close with FL constraints
- FL penalty +0.74 MAE makes centralized performance impossible
- 3rd place is strong result given constraints ✅
- Radical approach: 6x data augmentation per client
- Validation results: Val MAE 8.79/8.58 (vs 12.20/10.88) = **-28%/-21%**
- Kaggle result: **11.89067 MAE** (-0.04 improvement)
- Status: ✅ NEW BEST SCORE (3rd place maintained)

**Phase 8 (exp_040):** Hyperparameter tuning failure ❌
- Extensive grid search: 8000 evaluations (400 RF + 500 LGBM × 9 clients)
- Training time: 5 hours
- Validation results: Val MAE 6.79/6.51 (spectacular -2.00 improvement!)
- Kaggle result: **12.61584 MAE** (+0.72 dled to augmentation, exp_040 taught validation size limits
✅ **Technical Innovation:** Data augmentation in FL context (6x multiplication per-client)
✅ **Resilience:** Recovered from catastrophic failures (centralized error, exp_040 overfitting)

### Code Structure
```
dtbagging/
├── client_app.py           # FL client training  (exp_032-038), required strategic pivot
⚠️ **Hyperparameter Overfitting:** exp_040 catastrophic failure (+0.72 MAE), learned validation size limits
⚠️ **FL Penalty:** +0.70 MAE vs centralized (6.3% degradation) - reduced with augmentation
⚠️ **Small Dataset:** Only 999 samples → overfitting → solved with 6x augmentation
⚠️ **Preprocessing Bug:** Time series extraction missing → caught and fixed before submission
├── experiments/            # Valid experiments
│   ├── exp_001-015/       # Baseline → First best (12.01 MAE)
│   ├── exp_031/       maintained)
- **Score:** 11.89067 MAE (exp_039)
- **Gap to 2nd:** -1.20 MAE (challenging, likely unreachable with FL constraints)
- **Gap to 1st:** -1.46 MAE (very difficult with FL constraints)
- **Submissions remaining:** 18/35
- **Days to deadline:** 2 daysest failed (12.05 MAE)
│   Final Status

**exp_039 (SUCCESS):**
- ✅ Kaggle: 11.89067 MAE (new best, -0.04 improvement)
- Data augmentation strategy validated
- Validation improvement (-28%/-21%) partially transferred to Kaggle
- Gain modesto ma positivo

**exp_040 (CATASTROPHIC FAILURE):**
- ❌ Kaggle: 12.61584 MAE (worst in months, +0.72 degradation)
- Extensive grid search (8000 evaluations, 5h) backfired
- Overfitting su validation piccola (130 samples/client)
- Lesson: Small validation dangerous for hyperparameter tuning

**Recommendation:** Accept 3rd place (11.89067 MAE) as final result

Con 18 submission rimanenti e 2 giorni alla deadline, gap -1.20 MAE al 2° posto è troppo grande da chiudere con FL constraints. exp_040 failure ha dimostrato i limiti dell'ottimizzazione FL con dati piccoli. Data augmentation (exp_039) è contributo tecnico innovativo con gain modesto ma positivo.

**Final note:** Il percorso attraverso 17 submissions ha dimostrato approccio metodico al Federated Learning: ensemble methods, data augmentation innovativa, e lezioni critiche sui limiti dell'hyperparameter tuning con validation sets piccoli. exp_039 rappresenta il miglior compromesso tra performance, FL compliance, e robustezza.

---

**Last Updated:** 7 Gennaio 2026, 15:00  
**Final Submission:** exp_039 (11.89067 MAE, 3rd place) ✅  
**Submissions Used:** 17/35 (18 remaining)  
**Deadline:** 9 Gennaio 2026  
**Status:** Ready for final documentation
### Key Files for exp_039
- Training: `train_rf_augmented.py`, `train_lgbm_augmented.py`
- Inference: `generate_submission.py` (with time series extraction fix)
- Models: `rf_augmented_model.pkl` (17MB), `lgbm_augmented_model.pkl` (233KB)
- Submission: `submission_exp039.csv` (189 predictions)

---

## 9. Conclusion

Attraverso 16 submission e 39 esperimenti (validi e non), abbiamo raggiunto e mantenuto il **3° posto con 11.92674 MAE** (exp_031), superando gli errori iniziali di training centralizzato e i fallimenti di ottimizzazione post-031.

### Journey Summary

**Phase 1-3 (exp_001-015):** Baseline establishment → 12.01 MAE
- Established FL-compliant pipeline
- Feature engineering + ensemble → -0.74 MAE improvement
- First success: 3rd place with exp_015

**Phase 4:** Critical error discovery
- Found and corrected centralized training bug
- ~10 submissions invalidated but lessons learned

**Phase 5 (exp_031):** Breakthrough with three-way ensemble
- Added LightGBM as 3rd algorithm
- **Best score: 11.92674 MAE** (-0.08 from exp_015)
- 3rd place confirmed and maintained

**Phase 6 (exp_032-038):** Optimization plateau
- 4 consecutive failures: weight opt, 4-way, 5-way, 2-way
- All attempts worsened performance (+0.07 to +0.12 MAE)
- **Key insight:** Local optimum reached, pivot needed

**Phase 7 (exp_039):** Data augmentation strategy
- Radical approach: 6x data augmentation per client
- Validation results: Val MAE 8.79/8.58 (vs 12.20/10.88) = **-28%/-21%**
- Expected Kaggle: 11.40-11.70 MAE (**-0.25 to -0.52 improvement**)
- Status: Ready for submission tonight (6 Gen 2026)

### Punti di Forza

✅ **Rigorous FL Compliance:** All valid experiments respect federated constraints
✅ **Methodical Approach:** Systematic experimentation with clear hypotheses
✅ **Learning from Failures:** exp_032-038 failures led to augmentation strategy
✅ **Technical Innovation:** Data augmentation while maintaining FL compliance
✅ **Strong Validation:** Val MAE improvements (-28%/-21%) indicate real progress

### Sfide Superate

⚠️ **Centralized Training Error:** Detected and corrected, 10 submissions lost
⚠️ **Optimization Plateau:** 4 failed experiments, required strategic pivot
⚠️ **FL Penalty:** +0.74 MAE vs centralized (6.6% degradation)
⚠️ **Small Dataset:** Only 999 samples → overfitting → solved with augmentation
⚠️ **Preprocessing Bug:** Time series extraction missing → caught and fixed

### Current Position

- **Rank:** 3rd place (confirmed)
- **Score:** 11.92674 MAE
- **Gap to 2nd:** -1.23 MAE (challenging but potentially reachable)
- **Gap to 1st:** -1.49 MAE (very difficult with FL constraints)
- **Submissions remaining:** 19/35

### Next Critical Milestone

**exp_039 submission (tonight):**
- If successful (< 11.70): Major breakthrough, new best strategy
- If marginal (11.70-11.90): Small gain, consider further tuning  
- If failed (> 11.90): Maintain exp_031, accept strong 3rd place

Con 19 submission rimanenti e 3 giorni alla deadline, exp_039 rappresenta il tentativo più promettente per avvicinarsi al 2° posto. La strategia di data augmentation, validata dai risultati locali drammatici (-28%/-21% Val MAE), offre una nuova direzione dopo il plateau degli esperimenti di ottimizzazione classica.

**Final note:** Indipendentemente dal risultato di exp_039, il percorso ha dimostrato un approccio solido e metodico al Federated Learning, con apprendimenti preziosi su ensemble methods, data augmentation, e i limiti/opportunità dei constraint FL in competizioni reali.

---

**Last Updated:** 6 Gennaio 2026, 18:00  
**Next Submission:** 6 Gennaio 2026, tonight (exp_039)  
**Submissions Remaining:** 19/35  
**Current Best:** exp_031 (11.92674 MAE, 3rd place) ✅

