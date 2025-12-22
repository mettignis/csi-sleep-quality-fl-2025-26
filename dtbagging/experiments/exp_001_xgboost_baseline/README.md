# Esperimento 001 - XGBoost Baseline

**Data:** 22 Dicembre 2025
**Score Kaggle:** 12.74541
**Posizione:** 9°

## Configurazione

### Modello
- **Algoritmo:** XGBoost
- **n_estimators:** 100
- **max_depth:** 6
- **objective:** reg:squarederror

### Federated Learning
- **Metodo:** Training locale sequenziale (train_local.py)
- **Client:** 9
- **Round:** 10
- **Test fraction:** 0.2

### Preprocessing
- Time-series → statistiche (mean, min, max, std)
- Rimozione colonne vuote
- 40 features finali (incluso Unnamed: 0)

## Risultati Training

| Client | Train Samples | Test Samples | Final RMSE | Final MAE |
|--------|--------------|--------------|------------|-----------|
| 0      | 77           | 20           | ~15        | ~13       |
| 1      | 64           | 16           | ~14        | ~12       |
| 2      | 51           | 13           | ~13        | ~11       |
| ...    | ...          | ...          | ...        | ...       |

## Note
- Primo modello baseline
- Ray simulation non funzionante su macOS (usato train_local.py)
- Bug risolto: act_activeTime colonna vuota causava 0 samples
