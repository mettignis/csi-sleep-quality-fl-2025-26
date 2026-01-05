# Progress Report - Federated Learning Sleep Quality Prediction

**Project:** Kaggle Sleep Quality Challenge  
**Team:** [Team Name]  
**Date:** 4 Gennaio 2026  
**Deadline:** 9 Gennaio 2026  

---

## Executive Summary

**Current Status:**
- **Best Valid Score:** 12.01004 MAE (exp_015, 3rd place)
- **Submissions Used:** 12/35 (23 remaining)
- **Target:** 10.69856 MAE (2nd place, gap: -1.31 MAE)
- **Approach:** Federated Learning ensemble methods
- **Key Challenge:** Maintaining FL compliance while optimizing performance

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

**exp_031: Three-Way Ensemble with LightGBM FL** ⏳
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
- **Expected Score:** 11.5-11.8 MAE
- **Status:** ⏳ Ready for submission (5 Gen 2026)
- **FL Compliance:** ✅ All components fully federated

---

## 3. Results Summary

### Valid Experiments (Federated Compliant)

| Exp | Model | Score (MAE) | Position | Status |
|-----|-------|-------------|----------|--------|
| **015** | **RF Advanced + XGB** | **12.01004** | **3rd** | **✅ BEST** |
| 012 | RF Base + XGB | 12.16010 | 4th | ✅ Valid |
| 014 | RF Advanced | 12.19651 | 5th | ✅ Valid |
| 008 | RF Improved | 12.45205 | 6th | ✅ Valid |
| 001 | XGBoost Baseline | 12.74541 | 9th | ✅ Valid |
| 002 | RF Baseline | 12.95967 | 10th | ✅ Valid |
| **031** | **Three-Way Ensemble** | **11.5-11.8?** | **?** | **⏳ Pending** |

### Invalid Experiments (Centralized - Archived)

| Exp | Model | Score (MAE) | Reason |
|-----|-------|-------------|--------|
| 024 | Weight-optimized 4-way | 11.18725 | ❌ Centralized concat |
| 030 | Stacking meta-learner | 11.47736 | ❌ Centralized + overfitting |
| 013 | Stacking RF+XGB | 12.81492 | ❌ Loop su 9 clients |
| 016 | Three-way ensemble | 11.73074 | ❌ Loop su 9 clients |

**Location:** Moved to `experiments_sbagliati/` for documentation

---

## 4. Leaderboard Analysis

### Current Standing
- **1st Place:** 10.43507 MAE (gap: -1.57 from our best)
- **2nd Place:** 10.69856 MAE (gap: -1.31 from our best) 🎯 **TARGET**
- **3rd Place:** 12.01004 MAE ✅ **US (exp_015)**

### FL Penalty Analysis
- **Best Centralized (invalid):** 11.18725 MAE
- **Best Federated (valid):** 12.01004 MAE
- **FL Penalty:** +0.82 MAE (~7.3% degradation)
- **Interpretation:** Federated learning constraint significantly limits optimization potential with small per-client datasets (5 users/client)

---

## 5. Key Learnings & Insights

### Technical Insights

1. **Ensemble > Single Models:** Every ensemble outperformed its best component
   - exp_012 (12.16) beat exp_008 (12.45) by 0.29 MAE
   - exp_015 (12.01) beat exp_014 (12.20) by 0.19 MAE

2. **Advanced Features Matter:** Time-series feature engineering provided significant gains
   - Basic features: 12.45 MAE (exp_008)
   - Advanced features: 12.20 MAE (exp_014)
   - Gain: -0.25 MAE from feature engineering alone

3. **XGBoost for Stability:** XGBoost at 30% consistently improved ensemble robustness
   - Lower weight but high impact on reducing variance
   - Complementary to RF's high-variance predictions

4. **Algorithm Diversity:** Three algorithms better than two (theory)
   - RF: feature interactions, bagging
   - XGBoost: level-wise boosting
   - LightGBM: leaf-wise boosting
   - Different bias-variance tradeoffs

5. **FL Constraint is Real:** 0.82 MAE penalty for FL compliance
   - Small per-client data (5 users) limits model capacity
   - Ensemble & bagging help mitigate this

### Methodological Insights

1. **FL Compliance is Critical:** Always verify data loading pipeline
   - `pd.concat(all_clients)` before training = centralized ❌
   - `for client_id: train(client_data)` then aggregate = federated ✅

2. **Local Validation Saves Submissions:** Split training data 80/20 for local testing
   - Unlimited local experiments before Kaggle submission
   - Saved ~10+ submissions with this approach

3. **Incremental Improvement:** Small gains compound
   - exp_001→015: -0.74 MAE through 6 valid improvements
   - Each step: feature eng, better RF, ensemble, advanced features

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
2. ✅ Diversity (different algorithms/features)
3. ✅ Above-baseline performance (all components useful)

**Our Results Confirm Theory:**
- exp_015: 12.01 < 0.70×12.20 + 0.30×12.74 = 12.32 (theoretical)
- Actual gain: -0.31 MAE from ensemble synergy

### Federated Learning Challenges

**Data Heterogeneity:**
- 9 clients with different sleep patterns
- Client 2: Val MAE 18.77 (hardest)
- Client 3: Val MAE 6.46 (easiest)
- Solution: Bagging across clients averages out local biases

**Limited Local Data:**
- 5 users/client × ~20 samples/user = ~100 samples/client
- Too small for complex models (overfitting)
- Solution: Simpler models per client + ensemble aggregation

---

## 7. Next Steps & Strategy

### Immediate (5 Gen 2026)
✅ Submit exp_031 (three-way ensemble)
- If > 12.01: Stop, conserve submissions
- If 11.5-12.01: Proceed to weight optimization
- If < 11.5: Celebrate and analyze

### Short-term (6-7 Gen)

**exp_032: Weight Optimization** (if needed)
- Grid search 125 combinations locally
- Range: RF 45-60%, XGB 20-35%, LGBM 10-25%
- Expected: -0.10/-0.30 MAE

**exp_033: XGBoost FL Tuning** (if still > 11.5)
- Deeper trees, more rounds, lower lr
- Expected: XGBoost 12.7→12.3, ensemble improvement

### Long-term (8-9 Gen)

**exp_034: Four-Way with CatBoost FL**
- Add CatBoost trained per client
- 45% RF + 25% XGB + 20% LGBM + 10% Cat
- Expected: 11.2-11.5 MAE

---

## 8. Technical Artifacts

### Code Structure
```
dtbagging/
├── train_*.py              # FL training scripts
├── inference_*.py          # Prediction generation
├── task.py                 # FL data loading (critical)
├── experiments/            # Valid experiments
│   ├── exp_001-015/       # Baseline → Best
│   └── exp_031/           # Current frontier
└── experiments_sbagliati/ # Invalid (archived)
```

### Validation Strategy
- Local validation: 799 train / 200 validation (temporal split)
- No cross-validation (preserves temporal order)
- Metric: MAE (matches Kaggle)

---

## 9. Conclusion

Attraverso 15+ esperimenti validi e superando l'errore delle 6-7 submission centralizzate, abbiamo raggiunto il 3° posto (12.01 MAE) con approccio ensemble robusto e FL-compliant.

**Punti di forza:**
- ✅ Approccio metodologico rigoroso
- ✅ FL compliance verificata
- ✅ Ensemble theory applicata correttamente
- ✅ Miglioramento incrementale (-0.74 MAE)

**Sfide:**
- ⚠️ FL penalty +0.82 MAE
- ⚠️ Errore centralized: 6-7 submission perse

Con 23 submission rimanenti, exp_031 (three-way) rappresenta il tentativo più avanzato per colmare il gap verso il 2° posto.

---

**Last Updated:** 4 Gennaio 2026  
**Next Submission:** 5 Gennaio 2026 (exp_031)  
**Submissions Remaining:** 23/35

