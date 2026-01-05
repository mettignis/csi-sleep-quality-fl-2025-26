# Experiment 034: Four-Way Ensemble with CatBoost FL

## Objective
Aggiungere CatBoost come 4° modello all'ensemble per massimizzare la diversità e migliorare le predizioni.

## Approach

### Why CatBoost?
CatBoost è ideale per questo scenario perché:
1. **Ordered Boosting:** Riduce overfitting su small data
2. **Robust to small datasets:** CatBoost è ottimizzato per dataset piccoli (~100 samples/client)
3. **Different algorithm:** Ordered boosting vs level-wise (XGB) vs leaf-wise (LGBM)
4. **Built-in regularization:** L2 leaf regularization integrato
5. **Categorical handling:** Anche se non usiamo categoriche, gestisce meglio la varianza

### Four-Way Ensemble Components
1. **Random Forest Advanced (exp_014):** 12.19651 MAE
   - Bagging, low bias, high variance
   - Advanced time-series features
   
2. **XGBoost Base (exp_001):** 12.74541 MAE
   - Level-wise growth, sequential boosting
   - Stable baseline
   
3. **LightGBM FL (exp_031):** ~10.88 MAE (validation)
   - Leaf-wise growth, fast
   - 9 models bagging
   
4. **CatBoost FL (exp_034):** NEW
   - Ordered boosting
   - 9 models bagging
   - Conservative params

### Training Strategy
- **Per-client training:** 9 CatBoost models (one per client)
- **Bagging aggregation:** Average predictions from 9 models
- **Conservative params:** 
  - depth=6 (shallow trees)
  - learning_rate=0.05 (slow learning)
  - l2_leaf_reg=3 (strong regularization)
  - early_stopping=20 rounds

### Ensemble Weights
**Initial configuration:**
- 40% RF Advanced (best single model)
- 25% XGBoost (stability)
- 20% LightGBM (good validation)
- 15% CatBoost (new, conservative)

**Rationale:**
- RF ha il miglior score Kaggle singolo → peso maggiore
- CatBoost è nuovo → peso conservativo (15%)
- Totale: 100% (0.40 + 0.25 + 0.20 + 0.15 = 1.0)

## Expected Results

### Scenarios
1. **Best case:** 11.50-11.65 MAE
   - CatBoost aggiunge valore significativo
   - 4 algoritmi catturano pattern complementari
   
2. **Good case:** 11.70-11.85 MAE
   - Leggero miglioramento vs exp_031 (11.92)
   - CatBoost non danneggia ensemble
   
3. **Worst case:** 11.90-12.00 MAE
   - Nessun miglioramento o peggioramento marginale
   - CatBoost non aggiunge diversità utile

### Risk Assessment
- **Risk:** Medium
- **Reason:** Nuovo modello potrebbe non generalizzare bene
- **Mitigation:** Peso conservativo (15%), early stopping, L2 regularization

## Execution

```bash
cd experiments/exp_034_four_way_catboost

# 1. Train CatBoost (9 models)
python train_catboost_exp034.py  # ~1 hour

# 2. Generate submission
python inference_exp034.py       # ~2 minutes
```

## FL Compliance
✅ **FEDERATED LEARNING COMPLIANT**
- CatBoost: 9 models trained separately (one per client)
- No centralized training
- Bagging aggregation (post-processing only)
- All ensemble components are FL-compliant

## Theoretical Framework

### Algorithm Diversity
Four different approaches to gradient boosting:
1. **RF:** Random feature selection + bagging (parallel)
2. **XGBoost:** Level-wise tree growth (sequential)
3. **LightGBM:** Leaf-wise tree growth (sequential, fast)
4. **CatBoost:** Ordered boosting with oblivious trees

### Bias-Variance Analysis
- **Low bias:** RF (deep trees, advanced features)
- **Medium bias:** XGBoost, LightGBM (regularized boosting)
- **Controlled bias:** CatBoost (ordered boosting, L2 reg)
- **Ensemble effect:** Reduces overall variance

### Expected Ensemble Benefit
```
Error(4-way) < Error(3-way) if:
1. CatBoost errors are complementary
2. Ordered boosting captures different patterns
3. Weight 15% non danneggia gli altri
```

## Next Steps

### If exp_034 < 11.70 MAE (SUCCESS)
✅ Submit exp_034
✅ Monitor leaderboard
✅ Consider exp_035: Weight optimization 4-way

### If exp_034 = 11.70-11.90 MAE (GOOD)
✅ Submit exp_034
→ exp_033: XGBoost hyperparameter tuning
→ exp_035: Alternative weight configurations

### If exp_034 > 11.90 MAE (NO IMPROVEMENT)
❌ Do not submit exp_034
→ exp_033: Focus on improving base models
→ Analyze why 4-way doesn't help

## Files
- `train_catboost_exp034.py`: Training 9 CatBoost models FL
- `inference_exp034.py`: Four-way ensemble inference
- `catboost_models_federated_exp034.pkl`: Trained models
- `training_metrics_exp034.pkl`: Training statistics
- `submission_exp034.csv`: Final submission
- `README.md`: This file

---

**Status:** Ready to train  
**Expected time:** 1 hour training + 2 min inference  
**Submissions remaining:** 22/35
