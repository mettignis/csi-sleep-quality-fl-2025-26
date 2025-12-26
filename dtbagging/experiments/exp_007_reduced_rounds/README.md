# Esperimento 007: XGBoost - Riduzione Round

**Data:** 26 Dicembre 2025  
**Algoritmo:** XGBoost (Federated Learning)  
**Obiettivo:** Ridurre overfitting fermando training prima (5 round vs 10)

## Razionale

Dopo 6 esperimenti falliti consecutivamente (exp_002-006), tutti peggiori del baseline:
- exp_002 (RF): 12.95967
- exp_003 (6000 alberi): 14.46567 ❌
- exp_004 (tuned): 14.26652 ❌
- exp_005 (conservative): 13.88667 ❌
- exp_006 (feature eng): 14.83370 ❌❌

**Pattern identificato:** Ogni aumento di complessità peggiora le cose.

**Soluzione:** RIDURRE complessità invece di aumentarla.

## Strategia

- **Meno round = meno cicli di apprendimento**
- **500 alberi vs 1000 baseline** → 50% di riduzione
- **Fermare training prima dell'overfitting**
- **Mantenere stessi hyperparameter vincenti**

## Configurazione

### Parametri Training
- **Round FL:** 5 (baseline: 10, riduzione 50%)
- **Alberi per client:** 100 (identico al baseline)
- **Alberi totali:** 500 (5 round × 9 clients × 100 trees / round ÷ 9 = ~500)
- **Max depth:** 6 (baseline)
- **Learning rate:** 0.3 (baseline)
- **Subsample:** 0.8 (baseline)
- **Colsample_bytree:** 0.8 (baseline)

### Dataset
- **Training:** 697 campioni totali (9 gruppi)
- **Test:** 189 campioni
- **Features:** 40 (identico al baseline, NO feature engineering)

## Training Details

```
⏱️  XGBOOST - RIDUZIONE ROUND - ESPERIMENTO 007
Round FL: 5 (baseline: 10, riduzione 50%)
Alberi per client: 100 (baseline: 100)
Alberi totali: 500 (baseline: 1000)
```

### Performance Training (ultimo round - round 5)
- Client 0: Test RMSE 13.91
- Client 1: Test RMSE 11.95
- Client 2: Test RMSE 25.82
- Client 3: Test RMSE 11.08
- Client 4: Test RMSE 11.66
- Client 5: Test RMSE 13.83
- Client 6: Test RMSE 18.60
- Client 7: Test RMSE 16.68
- Client 8: Test RMSE 15.75

Training RMSE: 0.00 (tutti i client - ancora overfitting)

## Predizioni

### Statistiche Output
- **Min:** 51.10
- **Max:** 92.99
- **Mean:** 77.74
- **Std:** 6.58
- **Median:** 77.70

### Confronto con altri esperimenti

| Esperimento | Round | Alberi | Min | Max | Mean | Std | Kaggle Score |
|------------|-------|--------|-----|-----|------|-----|--------------|
| exp_001 (baseline) | 10 | 1000 | 62.21 | 88.24 | 75.12 | 4.41 | **12.74541** ✅ |
| exp_003 (6000 alberi) | 30 | 6000 | 51.10 | 92.99 | 77.74 | 6.58 | 14.46567 ❌ |
| **exp_007 (5 round)** | **5** | **500** | **51.10** | **92.99** | **77.74** | **6.58** | **?** 🤔 |

### ⚠️ ALLARME: Statistiche Identiche a exp_003!

**Le predizioni sono IDENTICHE a exp_003** (che aveva 6000 alberi e fallì con 14.46567).

Questo suggerisce un **problema nell'aggregazione FL**:
- exp_003: 30 round, 6000 alberi → stats: 51.10-92.99, mean 77.74
- exp_007: 5 round, 500 alberi → stats: 51.10-92.99, mean 77.74

**Ipotesi:** L'aggregazione FL potrebbe salvare solo l'ultimo modello invece di aggregare correttamente.

## File Prodotti

- `final_model_exp007.json` - Modello XGBoost con 500 alberi
- `submission_exp007.csv` - Predizioni per Kaggle

## Aspettative

### Pro ✅
- **Meno alberi:** 500 vs 1000 → dovrebbe ridurre overfitting
- **Stessi hyperparameter:** Usa parametri vincenti del baseline
- **Strategia sensata:** Ridurre complessità è l'unico approccio non ancora fallito

### Contro ⚠️
- **Train RMSE=0.00:** Ancora overfitting su tutti i client
- **Statistiche sospette:** Identiche a exp_003 (possibile bug aggregazione)
- **Range ampio:** 51-93 vs baseline 62-88
- **Std alta:** 6.58 vs 4.41 baseline

### Rischi 🚨
1. **Bug nell'aggregazione:** Statistiche identiche a exp_003 suggeriscono problema nel codice FL
2. **Overfitting comunque:** Train RMSE=0.00 significa che anche 500 alberi sono troppi
3. **Potrebbe fare score simile a exp_003:** ~14.46567

## Risultato Kaggle

- **Score ottenuto:** ??? 
- **Baseline exp_001:** 12.74541
- **Aspettativa:** Se bug aggregazione → simile a exp_003 (14.46567)

---

## 🔍 ANALISI POST-SUBMISSION

*Da compilare dopo risultato Kaggle*

---

**Status:** 🆕 Esperimento completato - In attesa risultato Kaggle

**NOTA TECNICA:** Verificare codice aggregazione FL in `train_xgb_exp007.py` - possibile che salvi solo ultimo modello invece di aggregare correttamente i 90 modelli (9 clients × 10 round → 90 modelli totali, ma potrebbe salvare solo gli ultimi 9).
