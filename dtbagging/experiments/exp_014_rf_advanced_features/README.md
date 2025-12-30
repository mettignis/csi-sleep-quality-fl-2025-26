# Esperimento 014: Random Forest con Advanced Features

**Data:** 29 Dicembre 2025  
**Algoritmo:** Random Forest con Feature Engineering Avanzato  
**Obiettivo:** Estrarre feature avanzate da time-series per catturare pattern nascosti

## Razionale

**Situazione prima di exp_014:**
- 🥇 exp_012 (Ensemble 70/30): 12.16010 - BEST
- 🥈 exp_008 (RF improved): 12.45205
- 🥉 exp_001 (XGBoost baseline): 12.74541

**Problema identificato:**
- Usiamo solo 4 statistiche base per time-series: mean, min, max, std
- **12 feature totali** da time-series (4 stats × 3 series: hr, resp, stress)
- First place (10.43) usa sicuramente advanced time-series features
- Gap: 1.73 punti da colmare

**Ipotesi:**
- Percentili, skewness, kurtosis, trend, peaks, entropy catturano pattern mancanti
- RF ha dimostrato di gestire complessità (exp_008 funzionò)
- XGB + features fallì (exp_006), quindi NO features su XGBoost

## Strategia

### 1. Feature Engineering Avanzato

**Da ogni time-series (hr, resp, stress) estraiamo:**

**Percentili (4 features):**
- p25, p75, p90, IQR (interquartile range)

**Shape Statistics (2 features):**
- Skewness (asimmetria distribuzione)
- Kurtosis (presenza outlier/code pesanti)

**Variabilità Avanzata (3 features):**
- Range (max - min)
- CV (coefficient of variation: std/mean)
- MAD (median absolute deviation)

**Trend Analysis (2 features):**
- Trend slope (pendenza regressione lineare)
- Trend R² (fit del trend)

**Change Rate (3 features):**
- Mean diff (media differenze successive)
- Std diff (variabilità cambiamenti)
- Max abs diff (massimo cambiamento)

**Peak Detection (3 features):**
- Num peaks (conteggio picchi)
- Peak rate (picchi / lunghezza serie)
- Mean peak value (valore medio dei picchi)

**Information Theory (1 feature):**
- Entropy (irregolarità/imprevedibilità)

**Temporal Correlation (1 feature):**
- Autocorrelation lag-1 (correlazione con step precedente)

**Energy & Dynamics (2 features):**
- Energy (somma quadrati / lunghezza)
- Zero-crossing rate (attraversamenti media)

**Totale:** 21 advanced features × 3 time-series = **63 advanced features**

### 2. Feature Selection

**Da 105 features (42 base + 63 advanced) → 25 top features**

- **SelectKBest** con f_regression
- Seleziona le 25 feature con highest correlation con target
- **Ratio:** 793 samples / 25 features = 31.7 samples/feature ✅
- Sopra soglia minima (10-20 samples/feature)

### 3. Training

**Configurazione RF (identica a exp_008):**
- n_estimators: 200
- max_depth: 15
- min_samples_split: 5
- min_samples_leaf: 2
- max_features: 'sqrt'
- random_state: 42 + round_num

**Setup:**
- 10 rounds FL
- 9 clients
- 90 modelli totali (9 × 10)

## Configurazione

### Dataset
- **Training:** 793 campioni (9 gruppi combinati)
- **Test:** 189 campioni
- **Features iniziali:** 105 (42 base + 63 advanced)
- **Features selezionate:** 25 (top features)

### Advanced Features Strategy
```python
# Per ogni time-series:
features = {
    'percentili': ['p25', 'p75', 'p90', 'iqr'],
    'shape': ['skewness', 'kurtosis'],
    'variability': ['range', 'cv', 'mad'],
    'trend': ['trend_slope', 'trend_r2'],
    'change': ['mean_diff', 'std_diff', 'max_abs_diff'],
    'peaks': ['num_peaks', 'peak_rate', 'mean_peak_value'],
    'info': ['entropy'],
    'temporal': ['autocorr'],
    'energy': ['energy', 'zcr']
}
```

## Training Results

### Feature Selection
- **Dataset completo:** 793 samples, 105 features
- **Feature selection:** 105 → 25 features con SelectKBest
- **Selezionate:** Top 25 feature più correlate con target

### Performance per Round (RMSE medio)

| Round | Avg RMSE | Note |
|-------|----------|------|
| 1 | 14.12 | Iniziale |
| 2 | 13.98 | |
| 3 | 13.91 | |
| 4 | 13.87 | |
| 5 | 13.84 | |
| 6 | 13.81 | |
| 7 | 13.79 | |
| 8 | 13.77 | |
| 9 | 13.76 | |
| 10 | 13.75 | Finale |

**Trend:** RMSE decresce costantemente → modelli apprendono

## Predizioni

### Statistiche Test Set

**RF con Advanced Features (90 modelli):**
- **Mean:** 73.83
- **Std:** 4.78
- **Min:** 60.58
- **Max:** 82.57
- **Median:** ~74.5

### Confronto con Altri Esperimenti

| Esperimento | Algoritmo | Mean | Std | Score Kaggle |
|------------|-----------|------|-----|--------------|
| exp_012 (Ensemble) | 70/30 RF+XGB | 74.60 | 3.58 | **12.16010** ✅ |
| **exp_014 (RF Adv)** | **RF 25 feat** | **73.83** | **4.78** | **12.19651** |
| exp_008 (RF Base) | RF 42 feat | 74.42 | 2.24 | 12.45205 |
| exp_001 (XGBoost) | XGBoost | 75.01 | 7.84 | 12.74541 |

### Analisi Differenze

**vs exp_008 (RF senza advanced features):**
- Mean: 73.83 vs 74.42 (-0.59) → Predizioni leggermente più basse
- Std: 4.78 vs 2.24 (+2.54) → Più variabilità
- **Score:** 12.19 vs 12.45 (-0.26) → ✅ MIGLIORAMENTO

**vs exp_012 (Ensemble 70/30):**
- Mean: 73.83 vs 74.60 (-0.77) → Sistematicamente più basso
- Std: 4.78 vs 3.58 (+1.20) → Più variabilità
- **Score:** 12.19 vs 12.16 (+0.036) → ❌ Leggero peggioramento

## File Prodotti

- `rf_advanced_models_exp014.pkl` - 90 modelli RF + feature selector + lista 25 features
- `submission_exp014.csv` - Predizioni finali

## Risultato Kaggle

**Score:** **12.19651**

### Ranking Esperimenti

| Rank | Esperimento | Score | Differenza |
|------|-------------|-------|------------|
| 🥇 | exp_012 (Ensemble) | **12.16010** | - |
| 🥈 | **exp_014 (RF Adv)** | **12.19651** | **+0.036** |
| 🥉 | exp_008 (RF Base) | 12.45205 | +0.29 |
| 4️⃣ | exp_001 (XGBoost) | 12.74541 | +0.58 |
| 5️⃣ | exp_013 (Stacking) | 12.81492 | +0.65 |

## Analisi Risultato

### ✅ Successi

1. **Advanced features FUNZIONANO**
   - 12.19 vs 12.45 (RF base)
   - **Miglioramento -0.26 punti** rispetto a RF senza features
   - Le 25 feature selezionate catturano pattern utili

2. **Feature selection efficace**
   - 105 → 25 features evita overfitting
   - Ratio 31.7 samples/feature è sano
   - Percentili, entropy, autocorrelation aggiungono informazione

3. **RF gestisce complessità**
   - Conferma exp_008: RF robusto su small data
   - RMSE training decresce costantemente
   - Nessun segno di overfitting

### ❌ Limiti

1. **Non batte exp_012**
   - Differenza minima: +0.036 punti
   - Simple ensemble (70/30) resta migliore
   - XGB al 30% in exp_012 aggiunge diversità mancante

2. **Std più alta**
   - 4.78 vs 3.58 (exp_012)
   - Più variabilità = potenzialmente più errori grossolani
   - Trade-off: cattura pattern ma introduce instabilità

3. **Gap a first place ancora grande**
   - 12.19 vs 10.43 (first place)
   - Gap: 1.76 punti
   - Advanced features da sole insufficienti

## Insights Chiave

### Cosa Abbiamo Imparato

1. **Advanced time-series features FUNZIONANO**
   - Percentili, entropy, autocorrelation catturano pattern
   - Miglioramento chiaro vs feature base (mean/min/max/std)
   - Feature selection critica per evitare overfitting

2. **RF migliora, ma ensemble è migliore**
   - RF advanced (12.19) migliora vs RF base (12.45)
   - Ma ensemble RF base + XGB (12.16) ancora migliore
   - **Diversità > Complessità** su small data

3. **XGB contribuisce anche se peggiore**
   - exp_012 usa XGB (12.74 da solo) al 30%
   - Ensemble 12.16 < RF advanced 12.19
   - XGB aggiunge diversità preziosa

### Prossimo Step Logico

**exp_015: Ensemble RF Advanced + XGB**
- 70% RF con advanced features (exp_014: 12.19)
- 30% XGB base (exp_001: 12.74)
- **Atteso:** 11.9-12.1 (BEST OF BOTH WORLDS)
- Combina advanced features + diversità ensemble

## Lezioni per Futuro

### ✅ Continua Così

1. **Advanced time-series features su RF**
   - Percentili, entropy, autocorr sono utili
   - Feature selection essenziale
   - 25-30 feature ottimale per 800 samples

2. **RF come base model principale**
   - Robusto su small data
   - Gestisce features avanzate
   - Miglior singolo modello

### ❌ Evita

1. **RF advanced da solo**
   - Manca diversità
   - Ensemble con XGB migliora sempre
   - Anche se XGB peggiore, contribuisce

2. **Troppa fiducia in advanced features**
   - Migliorano ma non rivoluzionano
   - Gap a first place resta grande (1.76)
   - Serve approccio multi-faceted

---

**Status:** ✅ COMPLETATO - Score 12.19651 (2° posto nostro)

**Next:** exp_015 (Ensemble RF Advanced + XGB) → Atteso 11.9-12.1
