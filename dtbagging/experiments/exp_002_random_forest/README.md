# Esperimento 002 - Random Forest

**Data:** 22 Dicembre 2025
**Score Kaggle:** 12.95967 (peggiore di XGBoost)
**Posizione:** ~10°

## Confronto con XGBoost Baseline
| Modello | Score | Differenza |
|---------|-------|-----------|
| XGBoost Baseline | 12.74541 | **MIGLIORE** |
| Random Forest    | 12.95967 | +0.21426 |

## Configurazione

### Modello
- **Algoritmo:** Random Forest (ensemble)
- **n_estimators per client:** 100
- **max_depth:** 10
- **n_jobs:** -1 (tutti i core)
- **random_state:** 42 + round

### Federated Learning
- **Metodo:** Training locale sequenziale (train_rf.py)
- **Client:** 9
- **Round:** 10
- **Modelli totali:** 90 (9 client × 10 round)
- **Alberi totali:** 9000

### Preprocessing
- Identico a XGBoost
- Time-series → statistiche (mean, min, max, std)
- 40 features finali

## Risultati Training

**RMSE Test medio per client (round 10):**
- Client 0: 13.95
- Client 1: 11.52
- Client 2: 22.25 (worst)
- Client 3: 8.22 (best)
- Client 4: 15.24
- Client 5: 17.23
- Client 6: 14.84
- Client 7: 16.57
- Client 8: 12.33

**Media RMSE test:** ~14.68

## Predizioni Kaggle
- Min: 66.58
- Max: 79.00
- Mean: 72.90
- Std: 2.32 (bassa variabilità)

## Analisi

**Perché RF è peggiore:**
- RF ha predizioni meno variabili (std 2.32 vs più alta in XGBoost)
- Potrebbe essere troppo conservativo per questo dataset
- XGBoost gradient boosting si adatta meglio ai pattern

**Conclusione:** XGBoost rimane il modello migliore per ora.

## Prossimi Esperimenti
- [ ] XGBoost con più round (20-30)
- [ ] XGBoost con più alberi (200-500)
- [ ] Ensemble XGBoost + RF
