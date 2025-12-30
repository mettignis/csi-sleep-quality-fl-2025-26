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
| **exp_008 (RF improved)** | **RF** | **18000** | **68.77** | **78.35** | **74.42** | **2.24** | **12.45205** ✅✅ |
| exp_001 (baseline) | XGBoost | 1000 | 62.21 | 88.24 | 75.12 | 4.41 | 12.74541 |
| exp_002 (RF) | RF | 9000 | 66.58 | 79.00 | 72.90 | 2.32 | 12.95967 |

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
| **Score** | **12.95967** | **12.45205** | **-0.51** ✅ |

**Miglioramento:** -0.51 punti rispetto a exp_002 (4% meglio!)

## Risultato Kaggle

- **Score ottenuto:** 12.45205 ✅✅ **NUOVO BEST!**
- **Baseline exp_001:** 12.74541
- **Miglioramento:** -0.29 punti (2.3% meglio del baseline XGBoost!)
- **Posizione:** PRIMO esperimento a battere il baseline

---

## ✅✅ POST-MORTEM: Perché ha FUNZIONATO?

### Successo Straordinario

**PRIMO esperimento su 8 a battere il baseline!**

Score progression:
1. **exp_008 (RF improved): 12.45205** ✅✅ **BEST**
2. exp_001 (XGBoost baseline): 12.74541 (-0.29)
3. exp_002 (RF base): 12.95967 (-0.51)
4. exp_005 (XGBoost): 13.88667 (-1.43)
5. exp_004 (XGBoost): 14.26652 (-1.81)
6. exp_003 (XGBoost): 14.46567 (-2.01)
7. exp_006 (XGBoost + FE): 14.83370 (-2.38)

### Fattori Chiave del Successo

#### 1. **Random Forest > XGBoost su Dataset Piccoli**
- **697 campioni di training** → Troppo pochi per XGBoost
- XGBoost tende all'overfitting (Train RMSE=0.00 in TUTTI gli esperimenti)
- RF mantiene Train RMSE sano (6-9) → Generalizzazione migliore

#### 2. **Ensemble Averaging è Potente**
- **90 modelli mediati** (9 clients × 10 rounds)
- Riduce varianza delle predizioni
- Std 2.24 molto stabile

#### 3. **Più Alberi = Più Diversità**
- exp_002: 100 trees/modello
- exp_008: 200 trees/modello (+100%)
- Più diversità → Migliore generalizzazione

#### 4. **Profondità Aumentata Aiuta**
- exp_002: max_depth=10
- exp_008: max_depth=15 (+50%)
- Cattura pattern più complessi senza overfitting

#### 5. **Regolarizzazione Efficace**
- min_samples_split=5 (vs 2 in exp_002)
- Forza più generalizzazione
- Evita split troppo specifici

#### 6. **NO Feature Engineering**
- exp_006 (XGBoost + FE): 14.83370 ❌
- exp_008 (RF + base features): 12.45205 ✅
- Conferma: **il problema era l'algoritmo, non le feature**

### Analisi Statistica

**Perché mean 74.42 < baseline 75.12 produce score migliore?**

Mean più bassa (-0.70) indica predizioni leggermente più conservative:
- **Meno outlier estremi** → Range 68-78 vs baseline 62-88
- **Più consistenza** → Std 2.24 vs baseline 4.41
- **Meno errori grossolani** → Evita predizioni molto sbagliate

RMSE/MAE penalizzano errori grandi quadraticamente → Ridurre outlier è più importante che matchare la mean esatta.

### Lezioni Apprese

#### ✅ Cosa Funziona
1. **Random Forest > XGBoost** su dataset piccoli (<1000 samples)
2. **Ensemble averaging** (90 modelli) riduce varianza
3. **Più alberi** aumenta diversità senza overfitting
4. **Regolarizzazione moderata** (min_samples_split=5)
5. **Statistiche conservative** (std bassa, range stretto) → score migliore

#### ❌ Cosa NON Funziona
1. ~~Aumentare round/alberi XGBoost~~ → Overfitting
2. ~~Hyperparameter tuning XGBoost~~ → Sempre peggio
3. ~~Feature engineering massivo~~ → Troppi parametri, overfitting
4. ~~Ridurre round XGBoost~~ → Non basta per evitare overfitting

### Prossimi Passi Suggeriti

**Continuare sulla strada Random Forest:**

#### exp_009: RF con Ancora Più Alberi
- Trees per modello: 200 → 300-500
- Mantenere max_depth=15
- Obiettivo: Aumentare ancora diversità

#### exp_010: RF con Feature Selection
- Selezionare top 20-25 feature più importanti
- Ridurre rumore
- Mantenere trees=200, depth=15

#### exp_011: RF con Tuning Conservativo
- Provare max_depth=12-18
- min_samples_split=3-7
- Ottimizzare regolarizzazione

#### exp_012: Ensemble RF + XGBoost Baseline
- Mediare predizioni exp_008 (RF) e exp_001 (XGBoost)
- Combinare punti di forza di entrambi
- Weighted average (es. 60% RF, 40% XGBoost)

---

**Status:** ✅✅ **SUCCESSO! NUOVO RECORD!**

**Ranking esperimenti:**
1. 🥇 **exp_008: 12.45205** (RF improved)
2. 🥈 exp_001: 12.74541 (XGBoost baseline)
3. 🥉 exp_002: 12.95967 (RF base)
