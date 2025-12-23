# Esperimento 005: XGBoost Conservative Tuning

**Data:** 23 Dicembre 2025  
**Algoritmo:** XGBoost (Federated Learning)  
**Obiettivo:** Migliorare baseline con modifiche conservative ai parametri

## Configurazione

### Parametri Training
- **Round FL:** 10 (come baseline exp_001)
- **Alberi per client:** 100 (come baseline exp_001)
- **Alberi totali:** 1000 (10 round × 9 clients × 100 trees + aggregazione)
- **Learning rate:** 0.2 (era 0.3 in baseline) ⬇️
- **Max depth:** 7 (era 6 in baseline) ⬆️
- **Min child weight:** 1 (invariato)
- **Subsample:** 0.8 (invariato)
- **Colsample_bytree:** 0.8 (invariato)

### Dataset
- **Training:** 697 campioni totali (9 gruppi)
- **Test:** 189 campioni
- **Features:** 40

## Razionale

Dopo i fallimenti di exp_003 e exp_004 (entrambi con 6000 alberi), questo esperimento torna alla dimensione del baseline (1000 alberi) e applica **modifiche minime**:

### Modifiche Conservative
1. **Learning Rate ↓ (0.3 → 0.2)**
   - Riduzione moderata del 33%
   - Apprendimento più graduale
   - Meno rischio di overfitting

2. **Max Depth ↑ (6 → 7)**
   - Incremento di solo 1 livello
   - Permette pattern leggermente più complessi
   - Bilanciato con learning rate più basso

### Perché dovrebbe funzionare
✅ **Dimensione modello corretta** (1000 vs 6000 degli esperimenti falliti)  
✅ **Solo 2 parametri modificati** (minimal changes principle)  
✅ **Direzioni opposte** (lr↓ + depth↑) per bilanciamento  
✅ **Vicino al baseline vincente** (12.74541)

## Training Details

```
🎯 XGBOOST CONSERVATIVE TUNING - ESPERIMENTO 005
Round FL: 10 (come baseline)
Alberi per client: 100 (come baseline)
Alberi totali finali: 1,000
```

### Performance Round 10 (finale)
- Client 0: Test RMSE 14.50, Train RMSE 0.00
- Client 1: Test RMSE 10.49, Train RMSE 0.00
- Client 2: Test RMSE 24.78, Train RMSE 0.00
- Client 3: Test RMSE 9.71, Train RMSE 0.00
- Client 4: Test RMSE 15.32, Train RMSE 0.00
- Client 5: Test RMSE 18.65, Train RMSE 0.00
- Client 6: Test RMSE 13.00, Train RMSE 0.00
- Client 7: Test RMSE 16.24, Train RMSE 0.00
- Client 8: Test RMSE 13.64, Train RMSE 0.00

**Nota:** Train RMSE ancora ~0.00 ma con modello più piccolo

## Predizioni

### Statistiche Output
- **Min:** 57.43
- **Max:** 96.10
- **Mean:** 80.43
- **Std:** 5.95

### Confronto con altri esperimenti

| Esperimento | Alberi | LR | Depth | Min | Max | Mean | Std | Kaggle |
|------------|--------|----|----|-----|-----|------|-----|--------|
| exp_001 (baseline) | 900 | 0.3 | 6 | 62.21 | 88.24 | 75.12 | 4.41 | **12.74541** ✅ |
| exp_002 (RF) | 9000 | - | 10 | 66.58 | 79.00 | 72.90 | 2.32 | 12.95967 |
| exp_003 (enhanced) | 6000 | 0.3 | 6 | 51.10 | 92.99 | 77.74 | 6.58 | 14.46567 ❌ |
| exp_004 (tuned) | 6000 | 0.1 | 8 | 66.18 | 94.61 | 81.69 | 4.36 | 14.26652 ❌ |
| **exp_005 (conservative)** | **1000** | **0.2** | **7** | 57.43 | 96.10 | 80.43 | 5.95 | **TBD** |

## Analisi

### Predizioni vs Baseline
- Range più ampio: 57-96 vs 62-88 (baseline)
- Mean più alta: 80.43 vs 75.12 (+5.31)
- Std più alta: 5.95 vs 4.41 (+1.54)

### Osservazioni
⚠️ **Mean spostata in alto** → potenziale problema di bias  
⚠️ **Range più ampio** → maggiore varianza nelle predizioni  
✅ **Dimensione corretta** → evita overfitting da troppi alberi  

### Aspettative su Kaggle
- **Best case:** Migliora baseline (< 12.74541) grazie a depth+1
- **Likely case:** Simile a baseline (~12.7-12.9)
- **Worst case:** Peggio per mean spostata verso l'alto

## File Prodotti

- `final_model_exp005.json` - Modello XGBoost 1000 alberi
- `submission_exp005.csv` - Predizioni per Kaggle

---

**Status:** ⏳ Pronto per submission Kaggle
