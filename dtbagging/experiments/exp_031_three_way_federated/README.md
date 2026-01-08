# Esperimento 031: Three-Way Ensemble FEDERATED

**Data:** 4 Gennaio 2026  
**Algoritmo:** Three-Way Ensemble Federated Learning  
**Score Atteso:** 11.5-11.8 MAE

## Obiettivo

Creare ensemble con 3 modelli diversificati, tutti trainati in **FEDERATED LEARNING** per rispettare i requisiti della challenge Kaggle.

## Federated Learning Compliance ✅

**TUTTI i componenti rispettano FL:**

### 1. LightGBM Federated (20%, exp_031)
- ✅ 9 modelli separati, uno per client
- ✅ Ogni modello allena SOLO su `clients_data/group{partition_id}`
- ✅ Predizione: bagging (media dei 9 modelli)
- ✅ NO concatenazione dati

### 2. Random Forest Advanced (50%, exp_014)
- ✅ Usa `load_data_with_advanced_features(partition_id)`
- ✅ Carica SOLO dati del suo gruppo
- ✅ 90 modelli RF (9 clients × 10 rounds)

### 3. XGBoost Base (30%, exp_001)
- ✅ Usa `train_local.py` con `load_data(partition_id)`
- ✅ Training federated sequenziale

### 4. Ensemble (post-processing)
- ✅ Weighted average: 50% RF + 30% XGB + 20% LGBM
- ✅ NO training aggiuntivo
- ✅ Solo combinazione predizioni

## Razionale

**Perché Three-Way?**
- **Diversità:** 3 algoritmi diversi (RF, XGBoost, LightGBM)
- **Robustezza:** Bagging + boosting + boosting leaf-wise
- **Provato:** exp_016 (centralizzato) aveva 11.73 MAE con three-way
- **Target:** 11.5-11.8 MAE (vs best FL: exp_015 12.01 MAE)

**Perché questi pesi?**
- **50% RF**: Migliore singolo (exp_014: 12.19 MAE), bassa std (4.78)
- **30% XGB**: Diversità boosting, provato robusto (exp_001: 12.74 MAE)
- **20% LGBM**: Nuovo, leaf-wise growth diverso da XGB

## Training LightGBM FEDERATED (exp_031)

### Configurazione

**Parametri LightGBM:**
```python
params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'max_depth': 6,
    'min_child_samples': 10,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1
}
```

**Training:**
- 9 clients (uno per gruppo)
- 200 boost rounds max
- Early stopping: 20 rounds
- Features: 40 (base, no advanced)

### Risultati Training

**Media MAE:**
- Train: 7.76
- Validation: 10.88

**Dettagli per Client:**
| Client | Train MAE | Val MAE | N_train | N_val | Best Iter |
|--------|-----------|---------|---------|-------|-----------|
| 0      | 5.50      | 9.36    | 77      | 20    | 33        |
| 1      | 4.26      | 7.29    | 64      | 16    | 71        |
| 2      | 14.14     | 18.77   | 51      | 13    | 2         |
| 3      | 7.61      | 6.46    | 84      | 22    | 16        |
| 4      | 9.06      | 7.60    | 63      | 16    | 3         |
| 5      | 7.80      | 11.80   | 90      | 23    | 32        |
| 6      | 9.05      | 12.24   | 79      | 20    | 1         |
| 7      | 4.32      | 15.58   | 103     | 26    | 56        |
| 8      | 8.08      | 8.82    | 86      | 22    | 1         |

**Osservazioni:**
- Client 2,6,8: Early stopping molto presto (iter 1-3) - possibile overfitting
- Client 1,7: Training più lungo (iter 56-71) - convergenza graduale
- Validation MAE varia molto (6.46 - 18.77) - diversità tra gruppi

## Ensemble Finale

### Componenti

| Modello | Peso | Score Standalone | Mean Pred | Std Pred |
|---------|------|------------------|-----------|----------|
| RF Advanced (exp_014) | 50% | 12.19651 MAE | 73.84 | 4.78 |
| XGBoost Base (exp_001) | 30% | 12.74541 MAE | 75.01 | 7.84 |
| LightGBM FL (exp_031) | 20% | ~10.88 val | 77.63 | 1.25 |

### Formula

```python
prediction = 0.50 × RF_advanced + 0.30 × XGBoost + 0.20 × LightGBM_bagging
```

### Statistiche Ensemble

**Test Set:**
- Mean: 74.94
- Std: 4.41
- Range: [63.25, 82.53]

## Aspettative

### Best Case (30%): 11.5-11.7 MAE
- LightGBM aggiunge diversità leaf-wise
- Three-way ensemble cattura pattern complementari
- Bagging di 9 modelli riduce variance
- **Miglioramento:** -0.31/-0.51 vs exp_015 (12.01)

### Expected Case (50%): 11.7-11.9 MAE
- Ensemble simile a exp_016 centralizzato (11.73)
- Penalty FL minima grazie a bagging LGBM
- **Miglioramento:** -0.11/-0.31 vs exp_015

### Worst Case (20%): 11.9-12.1 MAE
- LGBM non aggiunge valore (overfitting su alcuni client)
- Diversità non sufficiente
- **Miglioramento:** -0.01/+0.09 vs exp_015

## Confronto con Esperimenti Precedenti

| Esperimento | Tipo | MAE | Federated | Note |
|-------------|------|-----|-----------|------|
| **exp_015** | 2-way ensemble | **12.01** | ✅ | BEST precedente |
| exp_012 | 2-way ensemble | 12.16 | ✅ | RF base + XGB |
| exp_014 | RF advanced | 12.20 | ✅ | Standalone |
| exp_008 | RF improved | 12.45 | ✅ | Base component |
| **exp_016** | 3-way ensemble | **11.73** | ❌ | Centralizzato (reference) |
| **exp_031** | 3-way ensemble FL | **11.5-11.8?** | ✅ | **NEW** |


