# Experiment 032: Weight Optimization for Three-Way Ensemble

## Objective
Ottimizzare i pesi di RF, XGBoost e LightGBM per migliorare l'ensemble rispetto a exp_031.

## Approach

### exp_031 Results
- **Pesi:** 50% RF + 30% XGB + 20% LGBM
- **Kaggle MAE:** 11.92674
- **Improvement vs exp_015:** -0.08 MAE

### Weight Optimization Strategy
1. **Grid Search Locale:**
   - RF weights: 40%, 45%, 50%, 55%, 60%
   - XGBoost weights: 20%, 25%, 30%, 35%, 40%
   - LightGBM weights: 10%, 15%, 20%, 25%, 30%
   - Constraint: sum = 1.0
   - Total combinations: 125

2. **Validation:**
   - Split: 80% training / 20% validation
   - Metric: MAE on validation set
   - Modelli già allenati (no re-training)

3. **Ensemble Components:**
   - RF Advanced (exp_014): 12.19651 MAE
   - XGBoost Base (exp_001): 12.74541 MAE
   - LightGBM Bagging (exp_031): 9 models averaged

## Rationale

### Why Weight Optimization?
- I pesi 50-30-20 di exp_031 erano arbitrari
- RF (12.20 MAE) migliore di XGB (12.75 MAE) → merita più peso?
- LGBM (10.88 MAE su validation) potrebbe meritare più peso
- Grid search locale trova combinazione ottimale senza rischi

### Expected Improvements
- **Low-risk:** No training, solo pesi
- **Fast:** 30-40 minuti esecuzione
- **Potential:** -0.10/-0.30 MAE improvement
- **Submission saved:** 22/35 submissions remaining


## Expected Results

### Scenarios
1. **Best case:** 11.60-11.70 MAE (LGBM peso aumentato)
2. **Good case:** 11.75-11.85 MAE (leggero miglioramento)
3. **Worst case:** 11.90-12.00 MAE (nessun miglioramento)

### Next Steps Based on Results
- **If < 11.70:** Submit exp_032, poi exp_034 (4-way con CatBoost)
- **If 11.70-11.85:** Submit exp_032, poi exp_033 (XGBoost tuning)
- **If > 11.90:** Skip submission, go to exp_033 o exp_034

## FL Compliance
✅ **FEDERATED LEARNING COMPLIANT**
- No training (solo ensemble post-processing)
- Tutti i modelli base sono federated:
  - RF (exp_014): ✅ Trained per client
  - XGBoost (exp_001): ✅ Trained per client
  - LightGBM (exp_031): ✅ 9 models, one per client
ì