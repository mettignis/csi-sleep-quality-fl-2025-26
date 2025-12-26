# Esperimento 008: Random Forest Migliorato

**Data:** 26 Dicembre 2025  
**Algoritmo:** Random Forest (Federated Learning)  
**Obiettivo:** Migliorare exp_002 (RF) che fece 12.95967 (solo 0.21 peggio del baseline!)

## Razionale

**exp_002 (Random Forest) è il SECONDO miglior risultato finora:**
- exp_001 (XGBoost baseline): 12.74541 ✅ BEST
- **exp_002 (Random Forest)**: 12.95967 (solo +0.21)
- exp_005 (XGBoost conservative): 13.88667 (+1.14)
- exp_004 (XGBoost tuned): 14.26652 (+1.52)
- exp_003 (XGBoost 6000 trees): 14.46567 (+1.72)
- exp_006 (Feature engineering): 14.83370 (+2.09)

**Osservazione chiave:** Random Forest era TROPPO CONSERVATIVO
- exp_002 std: 2.32 (molto bassa)
- exp_002 range: 66-79 (molto stretto)
- Potrebbe migliorare con più variabilità

## Strategia

**Miglioramenti vs exp_002:**
1. **Più alberi:** 100 → 200 per modello
2. **Più profondità:** max_depth 10 → 15
3. **Più generalizzazione:** min_samples_split 2 → 5
4. **Seed diversi:** random_state diverso per ogni round
5. **Ensemble averaging:** Media di 90 modelli (9 clients × 10 rounds)

**Obiettivo:** Aumentare variabilità mantenendo il buon score di exp_002

## Configurazione

### Parametri Training
- **Round FL:** 10 (identico a exp_002)
- **Alberi per client:** 200 (exp_002: 100, +100%)
- **Alberi totali:** 18,000 (90 modelli × 200 alberi)
- **Max depth:** 15 (exp_002: 10, +50%)
- **Min samples split:** 5 (exp_002: 2, più generalizzazione)
- **Min samples leaf:** 2
- **Max features:** 'sqrt'
- **Random state:** 42 + round_num (seed diverso per round)

### Dataset
- **Training:** 697 campioni totali (9 gruppi)
- **Test:** 189 campioni
- **Features:** 42 (40 base + Unnamed: 0 + day)

## Training Details

```
🌲 RANDOM FOREST MIGLIORATO - ESPERIMENTO 008
Round FL: 10
Alberi per client: 200 (exp_002: 100)
Max depth: 15 (exp_002: 10)
Min samples split: 5 (exp_002: 2)
```

### Performance Training (ultimo round - round 10)
- Client 0: Train RMSE 7.45, Test RMSE 12.51
- Client 1: Train RMSE 7.94, Test RMSE 8.64
- Client 2: Train RMSE 8.62, Test RMSE 13.09
- Client 3: Train RMSE 7.80, Test RMSE 11.39
- Client 4: Train RMSE 8.04, Test RMSE 13.11
- Client 5: Train RMSE 7.94, Test RMSE 13.85
- Client 6: Train RMSE 6.84, Test RMSE 12.17
- Client 7: Train RMSE 8.18, Test RMSE 17.25
- Client 8: Train RMSE 6.49, Test RMSE 12.31

**Train RMSE: 6-9** (NO overfitting estremo come XGBoost!)

### Ensemble
- **Modelli totali:** 90 (9 clients × 10 rounds)
- **Predizione:** Media di tutti i 90 modelli
- **Alberi coinvolti:** 18,000 totali

## Predizioni

### Statistiche Output
- **Min:** 68.77
- **Max:** 78.35
- **Mean:** 74.42
- **Std:** 2.24
- **Median:** 74.79

### Confronto con altri esperimenti

| Esperimento | Algoritmo | Alberi | Min | Max | Mean | Std | Kaggle Score |
|------------|-----------|--------|-----|-----|------|-----|--------------|
| exp_001 (baseline) | XGBoost | 1000 | 62.21 | 88.24 | 75.12 | 4.41 | **12.74541** ✅ |
| exp_002 (RF) | RF | 9000 | 66.58 | 79.00 | 72.90 | 2.32 | 12.95967 |
| **exp_008 (RF improved)** | **RF** | **18000** | **68.77** | **78.35** | **74.42** | **2.24** | **?** 🤔 |

### Analisi Predizioni

**Miglioramenti vs exp_002:**
- ✅ **Mean più vicina al baseline:** 74.42 vs exp_002 (72.90), baseline (75.12)
- ✅ **Std simile:** 2.24 vs exp_002 (2.32) → mantiene conservativismo
- ✅ **Range più realistico:** 68-78 vs baseline 62-88 (meno estremi)
- ✅ **NO overfitting:** Train RMSE 6-9 (sano, non 0.00)

**Rispetto al baseline XGBoost:**
- Mean: 74.42 vs 75.12 → Differenza -0.70 (molto vicino!)
- Std: 2.24 vs 4.41 → Più conservativo (come exp_002)
- Range: più stretto (68-78 vs 62-88) → Meno variabilità

## File Prodotti

- `final_model_rf_exp008.pkl` - 90 modelli Random Forest (pickle)
- `submission_exp008.csv` - Predizioni per Kaggle (ensemble averaging)

## Aspettative

### Pro ✅
1. **Secondo miglior algoritmo:** exp_002 RF fece 12.95967 (solo 0.21 peggio)
2. **Mean ottima:** 74.42 molto vicina a baseline 75.12
3. **NO overfitting:** Train RMSE sano (6-9 vs 0.00 di XGBoost)
4. **Ensemble potente:** 90 modelli × 200 alberi = 18,000 alberi mediati
5. **Statistiche realistiche:** Range 68-78 sensato, no outlier estremi

### Contro ⚠️
1. **Std bassa:** 2.24 (come exp_002) → Potrebbe essere troppo conservativo
2. **Range stretto:** 68-78 (10 punti) vs baseline 62-88 (26 punti)
3. **Mean leggermente bassa:** 74.42 vs 75.12 baseline (-0.70)

### Confronto Diretto con exp_002

| Metrica | exp_002 | exp_008 | Variazione |
|---------|---------|---------|------------|
| Trees | 9000 | 18000 | +100% |
| Max depth | 10 | 15 | +50% |
| Mean | 72.90 | 74.42 | +1.52 ⬆️ |
| Std | 2.32 | 2.24 | -0.08 ⬇️ |
| Range | 66-79 | 68-78 | Simile |
| Score | 12.95967 | ??? | ??? |

**Aspettativa:** Mean +1.52 più vicina al baseline → Potrebbe battere exp_002!

## Risultato Kaggle

- **Score atteso:** < 12.95967 (migliore di exp_002)
- **Target:** Avvicinarsi a 12.74541 (baseline)
- **Best case:** 12.5-12.8 range

---

## 🔍 ANALISI POST-SUBMISSION

*Da compilare dopo risultato Kaggle*

---

**Status:** 🆕 Esperimento completato - In attesa risultato Kaggle

**RACCOMANDAZIONE:** Provare PRIMA questo esperimento. Ha le migliori statistiche finora dopo il baseline, e RF ha dimostrato di generalizzare meglio di XGBoost su questo dataset piccolo.
