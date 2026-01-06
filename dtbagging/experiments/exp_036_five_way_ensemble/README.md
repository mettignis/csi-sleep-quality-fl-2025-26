# Experiment 036: Five-Way Ensemble with Dual Random Forest

## Obiettivo
Migliorare exp_031 (11.92674 MAE) aggiungendo una seconda variante di Random Forest per catturare pattern complementari.

## Strategia

### Modelli nell'Ensemble (tutti FL-compliant)

1. **RF-base** (exp_008)
   - Features: Base (senza engineering complesso)
   - Val MAE: ~12.50
   - Trained: 9 client separati
   
2. **RF-advanced** (exp_014)
   - Features: Engineered (aggregations, derived)
   - Val MAE: ~12.20
   - Trained: 9 client separati
   
3. **XGBoost-base** (exp_001)
   - Val MAE: ~13.30
   - Trained: 9 client separati
   
4. **LightGBM-FL** (exp_031)
   - Val MAE: ~10.88
   - Trained: 9 client separati

### Razionale del Dual RF

**Perché due Random Forest?**
- RF-base: Generalizza meglio, meno overfit
- RF-advanced: Cattura pattern complessi con feature engineering
- Insieme: Coprono debolezze reciproche
- Diversity: Stessi algoritmi, dati diversi = predizioni decorrelate

**Expected Benefit:**
- Exp_031 (3-way): 11.92674 MAE
- Exp_036 (5-way): Target 11.60-11.85 MAE (-0.10/-0.30)

### Grid Search

**Search Space:**
- RF-base: 15-35% (step 5%)
- RF-advanced: 20-40% (step 5%)
- XGBoost: 10-30% (step 5%)
- LightGBM: 10-30% (step 5%)

**Constraint:** Weights sum to 1.0

**Scoring:** Proxy metric (mean distance + variance penalty)

## Federated Learning Compliance

✅ **Tutti i modelli FL-compliant**
- Ogni modello trained separatamente su 9 client
- Nessun training su dati combinati
- Ensemble = post-processing puro (lecito)

## Esecuzione

```bash
cd experiments/exp_036_five_way_ensemble
python optimize_and_generate.py
```

**Output:**
- `submission_exp036.csv` - Submission con best weights
- `top10_weights_exp036.csv` - Top 10 combinazioni

**Tempo:** ~30-40 minuti (grid search)

## Risultati Attesi

**Best Case:** 11.50-11.70 MAE (dual RF molto complementari)
**Realistic:** 11.65-11.85 MAE (piccolo miglioramento)
**Worst Case:** 11.85-12.00 MAE (marginal o neutrale)

**Probabilità successo:** ~65% di migliorare exp_031

## Confronto con Esperimenti Precedenti

| Exp | Strategy | Kaggle MAE | Delta |
|-----|----------|------------|-------|
| 031 | 3-way (RF+XGB+LGBM) | 11.92674 | baseline |
| 034 | 4-way (+CatBoost) | 12.00102 | +0.07 ❌ |
| 036 | 5-way (dual RF) | TBD | ? |

## Note

- CatBoost (exp_034) ha PEGGIORATO per preprocessing issues
- Dual RF approach più stabile e sicuro
- Se fallisce, rimane exp_031 come best
