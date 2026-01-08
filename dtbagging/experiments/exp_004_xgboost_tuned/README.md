# Esperimento 004: XGBoost Hyperparameter Tuning

**Data:** 23 Dicembre 2025  
**Algoritmo:** XGBoost (Federated Learning)  
**Obiettivo:** Ottimizzare hyperparameter per migliorare generalizzazione

## Configurazione

### Parametri Ottimizzati
- **learning_rate (eta):** 0.1 (era 0.3) → apprendimento più lento e preciso
- **max_depth:** 8 (era 6) → alberi più profondi, pattern più complessi
- **min_child_weight:** 3 (era 1) → maggiore regolarizzazione, meno overfitting
- **subsample:** 0.8 (invariato)
- **colsample_bytree:** 0.8 (invariato)

### Training Configuration
- **Round FL:** 30
- **Alberi per client:** 200
- **Alberi totali:** 6000
- **Dataset:** 697 campioni training, 189 test, 40 features

## Razionale del Tuning

1. **Learning Rate ↓ (0.3 → 0.1)**
   - Apprendimento più graduale
   - Riduce rischio overfitting
   - Migliore convergenza su dati piccoli

2. **Max Depth ↑ (6 → 8)**
   - Alberi più profondi catturano interazioni complesse
   - Utile per dataset con feature non lineari
   - Bilanciato con regolarizzazione aumentata

3. **Min Child Weight ↑ (1 → 3)**
   - Previene split su pochi campioni
   - Riduce variance, aumenta bias
   - Migliore generalizzazione su test set

## Training Details

### Performance Round 30 (finale)
- Client 0: Test RMSE 14.72, Train RMSE 0.00
- Client 1: Test RMSE 12.71, Train RMSE 0.25
- Client 2: Test RMSE 23.60, Train RMSE 0.13
- Client 3: Test RMSE 8.91, Train RMSE 0.09
- Client 4: Test RMSE 16.86, Train RMSE 0.18
- Client 5: Test RMSE 16.17, Train RMSE 0.12
- Client 6: Test RMSE 13.93, Train RMSE 0.11
- Client 7: Test RMSE 19.30, Train RMSE 0.16
- Client 8: Test RMSE 14.12, Train RMSE 0.19

**Osservazione:** Train RMSE > 0 (meno overfitting rispetto exp_003!)

## Predizioni

### Statistiche Output
- **Min:** 66.18
- **Max:** 94.61
- **Mean:** 81.69
- **Std:** 4.36

### Confronto con altri esperimenti

| Esperimento | Learning Rate | Max Depth | Min Weight | Min | Max | Mean | Std | Kaggle |
|------------|--------------|-----------|------------|-----|-----|------|-----|--------|
| exp_001 (baseline) | 0.3 | 6 | 1 | 62.21 | 88.24 | 75.12 | 4.41 | **12.74541** ✅ |
| exp_002 (RF) | - | 10 | - | 66.58 | 79.00 | 72.90 | 2.32 | 12.95967 |
| exp_003 (enhanced) | 0.3 | 6 | 1 | 51.10 | 92.99 | 77.74 | 6.58 | 14.46567 ❌ |
| **exp_004 (tuned)** | **0.1** | **8** | **3** | 66.18 | 94.61 | 81.69 | 4.36 | **14.26652** ❌ |

## Analisi

### Vantaggi del Tuning
✅ **Train RMSE non zero** → Meno overfitting (exp_003 aveva 0.00)  
✅ **Range più controllato** → Min 66 vs 51 di exp_003  
✅ **Std intermedia** → 4.36 (tra 2.32 RF e 6.58 exp_003)  
✅ **Mean più alta** → 81.69 potrebbe catturare pattern reali

### Risultato Kaggle
- **Score ottenuto:** 14.26652 ❌
- **Baseline exp_001:** 12.74541
- **Peggioramento:** +1.52 punti (11.9% peggio)
- **vs exp_003:** Marginalmente meglio (-0.20 punti) ma comunque pessimo

---

## Perché ha fallito?

### Problema: TROPPI ALBERI = OVERFITTING
La regolarizzazione (eta=0.1, min_child_weight=3) **NON** è bastata a compensare:
- **6000 alberi totali** (troppi per 697 campioni)
- **30 round FL** (troppi cicli)

### Imparato che
1. **6000 alberi >> 900 alberi baseline** → Overfitting inevitabile
2. **Regolarizzazione non basta** se il modello è troppo complesso
3. **Il baseline (900 alberi) aveva ragione** → era già dimensionato bene

### Confronto critico
```
exp_001: 10 round × 9 clients × 100 trees = 900 alberi  → 12.74 ✅
exp_003: 30 round × 9 clients × 200 trees = 6000 alberi → 14.46 ❌
exp_004: 30 round × 9 clients × 200 trees = 6000 alberi → 14.27 ❌
```

### Prossime strategie
✅ **Tornare a dimensioni baseline** (10 round, 100 alberi)  
✅ **Modificare solo 1-2 parametri** per volta  
✅ **Provare feature engineering** invece di più complessità  
✅ **Considerare ensemble** di modelli piccoli


---

**Status:** Esperimento fallito - Regolarizzazione non compensa troppa complessità
