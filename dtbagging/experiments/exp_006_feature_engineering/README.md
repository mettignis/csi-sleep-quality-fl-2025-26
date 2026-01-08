# Esperimento 006: Feature Engineering

**Data:** 26 Dicembre 2025  
**Algoritmo:** XGBoost (Federated Learning)  
**Obiettivo:** Migliorare baseline tramite feature engineering (NON hyperparameter tuning)

## Strategia

Dopo 5 esperimenti falliti (002-005) dove:
- Random Forest ha fatto peggio
- Più alberi → overfitting severo
- Hyperparameter tuning → sempre peggio del baseline

**Lezione appresa:** Il baseline (exp_001) ha già hyperparameter ottimali per questo dataset. Il problema NON sono i parametri, ma le feature.

**Soluzione:** Mantenere parametri baseline + aggiungere feature engineered

## Configurazione

### Parametri Training (IDENTICI AL BASELINE)
- **Round FL:** 10 (come exp_001)
- **Alberi per client:** 100 (come exp_001)
- **Alberi totali:** 1000 (10 round × 9 clients × 100 trees / round)
- **Max depth:** 6
- **Learning rate:** 0.3
- **Subsample:** 0.8
- **Colsample_bytree:** 0.8

### Feature Engineering (🆕)

#### Feature Originali: 40
- Time-series preprocessate (hr, resp, stress → mean/min/max/std)
- Sleep metrics, activity metrics

#### Feature Aggiunte: +18

**1. Interazioni (4 features)**
- `hr_stress_interaction` = hr_mean × stress_mean
- `sleep_hr_interaction` = sleep_efficiency × hr_mean
- `activity_intensity` = calories / (distance + ε)
- `resp_stress_interaction` = resp_mean × stress_mean

**2. Advanced Statistics (6 features)**
- `hr_range` = hr_max - hr_min
- `hr_cv` = hr_std / (hr_mean + ε) (coefficiente variazione)
- `resp_range`, `resp_cv`
- `stress_range`, `stress_cv`

**3. Ratios (7 features)**
- `deep_sleep_ratio` = deep_sleep / total_sleep
- `rem_sleep_ratio` = rem_sleep / total_sleep
- `light_sleep_ratio` = light_sleep / total_sleep
- `active_efficiency` = active_kcal / (active_time + ε)
- `hr_variability_index` = hr_std / (hr_range + ε)
- `sleep_fragmentation` = awake_count / (total_sleep + ε)
- (1 more computed ratio)

**4. Transformations (2 features)**
- `act_totalCalories_log` = log(calories + 1)
- `act_distance_log` = log(distance + 1)

**Feature Totali: 60** (40 base + 18 engineered + Unnamed: 0 + day)

## Training Details

```
🎨 XGBOOST + FEATURE ENGINEERING - ESPERIMENTO 006
Round FL: 10 (baseline)
Alberi per client: 100 (baseline)
Feature: 42 → 60 (+18 engineered)
```

### Performance Training (ultimo round - round 10)
- Client 0: Test RMSE 16.52
- Client 1: Test RMSE 10.06 ✅
- Client 2: Test RMSE 17.09
- Client 3: Test RMSE 15.81
- Client 4: Test RMSE 15.26
- Client 5: Test RMSE 13.49
- Client 6: Test RMSE 15.22
- Client 7: Test RMSE 21.20
- Client 8: Test RMSE 16.37

Training RMSE: 0.00 (tutti i client)

## Predizioni

### Statistiche Output
- **Min:** 25.73 ⚠️
- **Max:** 95.24
- **Mean:** 78.60
- **Std:** 9.52
- **Median:** 80.15

### Confronto con altri esperimenti

| Esperimento | Features | Min | Max | Mean | Std | Kaggle Score |
|------------|----------|-----|-----|------|-----|--------------|
| exp_001 (baseline) | 40 | 62.21 | 88.24 | 75.12 | 4.41 | **12.74541** ✅ |
| exp_002 (Random Forest) | 40 | 66.58 | 79.00 | 72.90 | 2.32 | 12.95967 |
| exp_003 (6000 trees) | 40 | 51.10 | 92.99 | 77.74 | 6.58 | 14.46567 ❌ |
| exp_004 (tuned) | 40 | 56.44 | 93.15 | 78.98 | 6.36 | 14.26652 ❌ |
| exp_005 (conservative) | 40 | 65.85 | 92.32 | 80.43 | 4.80 | 13.88667 ❌ |
| **exp_006 (feature eng)** | **60** | **25.73** | **95.24** | **78.60** | **9.52** | **14.83370** ❌❌ |


## Aspettative

### Pro 
- **Feature più informative:** Cattura relazioni complesse (hr×stress, sleep ratios)
- **Parametri ottimali:** Usa esattamente parametri vincenti del baseline
- **Generalizzazione:** Feature ingegnerizzate dovrebbero aiutare a generalizzare

### Contro 
- **Range molto ampio:** Min 25.73 è MOLTO basso (baseline min 62.21)
- **Std molto alta:** 9.52 vs 4.41 baseline → maggiore incertezza
- **Mean più alta:** 78.60 vs 75.12 → possibile bias verso l'alto
- **Possibile overfitting:** Train RMSE=0.00 su tutti i client

### Segnali Meh 
1. **Min 25.73** → Predizione estrema, probabilmente errata
2. **Std 9.52** → Più del doppio del baseline (4.41)
3. **Range amplissimo** → 25-95 vs baseline 62-88
4. **Train RMSE=0.00** → Possibile overfitting anche con parametri baseline

## Risultato Kaggle

- **Score ottenuto:** 14.83370 ❌❌
- **Baseline exp_001:** 12.74541
- **Peggioramento:** +2.09 punti (16.4% peggio)
- **Rank:** Peggio di TUTTI gli altri esperimenti (anche exp_003 con 6000 alberi!)

---

## Perché ha fallito, pure MALISSIMO direi?

### Problema Principale: OVERFITTING AMPLIFICATO

**Feature engineering ≠ Soluzione magica**

#### 1. Troppi Gradi di Libertà
- **60 features** su solo **697 campioni di training**
- Ratio: ~11.6 campioni per feature (PESSIMO, dovrebbe essere >>100)
- Baseline aveva 40 features → ratio 17.4 (già al limite)
- **18 feature in più** = 50% più parametri da imparare con stesso dataset

#### 2. Feature Engineered = Rumore
Le feature aggiunte probabilmente hanno catturato:
- **Pochi pattern reali**
- **Molto rumore specifico del training set**
- **Correlazioni spurie** che non generalizzano

Esempio: `hr_stress_interaction` potrebbe essere correlata al training set ma non al test

#### 3. Conferma: Train RMSE = 0.00
- **Tutti i client**: Train RMSE = 0.00
- **Significato**: Memorizzazione perfetta dei 60 pattern
- **Risultato**: Zero capacità di generalizzazione

#### 4. Statistiche Anomale
- **Min 25.73** → Outlier estremo (baseline min 62.21)
- **Std 9.52** → Doppia del baseline (4.41) → instabilità predittiva
- **Range 70 punti** (25-95) → Modello "confuso"

### Cause Tecniche

1. **Curse of Dimensionality**
   - 697 campioni / 60 features = 11.6 samples/feature
   - Regola del pollice ML: serve almeno 10-100x più campioni che feature
   - Siamo SOTTO la soglia minima

2. **Feature Non Selezionate**
   - Aggiunte TUTTE le 18 feature senza validazione
   - Nessuna feature selection o importance analysis
   - Probabilmente molte feature sono ridondanti o dannose

3. **Interazioni Complesse = Overfitting Facile**
   - Feature come `hr_stress_interaction` sono non-lineari
   - Catturano pattern molto specifici
   - Alto rischio di imparare correlazioni casuali


#### Cosa Funziona
- **exp_001 baseline**: 10 rounds, 100 trees, 40 features
- **Parametri**: lr=0.3, depth=6, subsample=0.8
- **Score**: 12.74541

#### 🔍 Cosa Rimane da Provare
**Strategia: RIDURRE, NON AUMENTARE**

1. **Meno Round** (exp_007)
   - Provare 5 round invece di 10
   - Fermare training prima dell'overfitting
   - Meno alberi totali (500 vs 1000)

2. **Feature Selection** (exp_008)
   - Usare solo top-10 feature più importanti
   - Eliminare feature ridondanti
   - Ridurre da 40 a 10-15 feature critiche

3. **Regolarizzazione Forte** (exp_009)
   - `min_child_weight`: 1 → 10+ (forzare generalizzazione)
   - `gamma`: 0 → 5-10 (penalità split)
   - `alpha` (L1): 1-5 (feature sparsity)

4. **Ensemble Baseline** (exp_010)
   - Allenare 5 modelli baseline con seed diversi
   - Average delle predizioni
   - Riduce varianza senza aumentare complessità

### Conclusione

**6 esperimenti consecutivi hanno TUTTI fatto peggio del baseline.**

Il problema NON è:
- Pochi alberi
- Parametri sbagliati
- Feature insufficienti

Il problema È:
- **Dataset piccolo (697 campioni)**
- **Facilità di overfitting**
- **Baseline già vicino al limite della generalizzazione**

**Prossimo approccio: SEMPLIFICARE, non complicare**

---

**Status:** FALLITO - Feature engineering ha PEGGIORATO le cose (+2.09 punti)
