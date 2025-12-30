# Esperimento 015: Ensemble RF Advanced + XGBoost Base

**Data:** 29 Dicembre 2025  
**Algoritmo:** Ensemble (RF Advanced Features + XGBoost Base)  
**Obiettivo:** Combinare RF migliorato (exp_014) con XGB per massimizzare performance

## Razionale

**Situazione dopo exp_014:**
- 🥇 exp_012 (Ensemble 70/30 RF base + XGB): 12.16010 - BEST
- 🥈 exp_014 (RF advanced): 12.19651 - Nuovo
- 🥉 exp_008 (RF base): 12.45205
- 4️⃣ exp_001 (XGBoost): 12.74541

**Osservazione Critica:**
- exp_014 (RF advanced solo): 12.19651
- exp_012 (RF base + XGB): 12.16010
- **exp_012 è migliore** anche con RF più debole!

**Perché exp_012 batte exp_014?**
- **Diversità:** XGB al 30% aggiunge pattern diversi da RF
- **Robustezza:** Ensemble compensa debolezze individuali
- **Proven:** Simple ensemble > single model anche se più complesso

**Ipotesi exp_015:**
Se:
- exp_012 = RF base (12.45) + XGB (12.74) → 12.16
- exp_014 = RF advanced (12.19) migliore di RF base

Allora:
- **exp_015 = RF advanced (12.19) + XGB (12.74) → 11.9-12.1?**

## Strategia

### Componenti Ensemble

**1. Random Forest Advanced (70%) - da exp_014**
- 90 modelli RF (9 clients × 10 rounds)
- 25 feature selezionate (da 105 totali)
- Advanced time-series features:
  * Percentili (p25, p75, p90, IQR)
  * Shape (skewness, kurtosis)
  * Variabilità (range, CV, MAD)
  * Trend (slope, R²)
  * Change rate (diffs)
  * Peaks (count, rate, values)
  * Info theory (entropy)
  * Temporal (autocorrelation)
  * Energy (energy, zero-crossing)
- **Score standalone:** 12.19651

**2. XGBoost Base (30%) - da exp_001**
- Modello baseline provato robusto
- 42 feature base (NO advanced features)
- Config: depth=6, lr=0.3, 100 trees
- **Score standalone:** 12.74541

### Formula Ensemble

```python
prediction_final = 0.7 × prediction_RF_advanced + 0.3 × prediction_XGBoost_base
```

**Perché 70/30?**
- RF advanced (12.19) molto migliore di XGB (12.74)
- Ma XGB contribuisce diversità preziosa
- Stesso ratio di exp_012 (funzionò)

## Configurazione

### Input Models

**RF Advanced (exp_014):**
- Source: `rf_advanced_models_exp014.pkl`
- Models: 90 RandomForest
- Features: 25 selezionate con SelectKBest
- Training RMSE: 13.75 (round 10)

**XGBoost Base (exp_001):**
- Source: `model.json`
- Features: 40 base
- Training RMSE: ~11.50

### Dataset
- **Test:** 189 campioni
- **Features RF:** 25 (advanced selected)
- **Features XGB:** 40 (base)

### Implementation

**Key aspects:**
1. **Preprocessing separato:**
   - RF: 105 features → 25 selected
   - XGB: 42 features base → riordinate per match training

2. **Feature alignment:**
   - XGB richiede ESATTO ordine colonne del training
   - Rimosse 'day' e 'act_activeTime' (non in training XGB)

3. **Aggregation:**
   - RF: media di 90 predizioni
   - XGB: predizione singola
   - Weighted average finale

## Predizioni

### Statistiche Componenti

**RF Advanced (exp_014):**
- Mean: 73.83
- Std: 4.78
- Range: 60.58-82.57

**XGBoost Base (exp_001):**
- Mean: ~75.01
- Std: ~7.84
- Range: ~62-88

**Ensemble exp_015 (atteso):**
- Mean: ~74.2 (tra 73.83 e 75.01)
- Std: ~5.5-6.0 (tra 4.78 e 7.84)
- Range: ~61-85

### Confronto Teorico

| Componente | Weight | Score Standalone | Contributo |
|------------|--------|------------------|------------|
| RF Advanced | 70% | 12.19651 | 8.54 |
| XGBoost Base | 30% | 12.74541 | 3.82 |
| **Ensemble** | **100%** | **~11.9-12.1** | **~12.36** |

**Nota:** Formula sopra è semplificazione. Score non è media lineare pesata.

## Aspettative

### Best Case (30% probabilità): 11.8-11.95
- RF advanced + XGB diversità = sinergia perfetta
- Ensemble cattura best of both worlds
- XGB compensa instabilità RF (std 4.78)
- **Risultato:** Nuovo BEST, miglioramento -0.20/-0.35 vs exp_012

### Expected Case (50% probabilità): 11.95-12.1
- Leggero miglioramento su exp_012 (12.16)
- RF advanced migliora predizioni base
- XGB aggiunge robustezza necessaria
- **Risultato:** -0.05/-0.20 vs exp_012, nuovo BEST

### Worst Case (20% probabilità): 12.1-12.2
- Std alta RF (4.78) dannosa più del previsto
- XGB non riesce a compensare instabilità
- Advanced features introducono noise
- **Risultato:** Simile o peggio di exp_012

## Analisi Strategica

### Perché Dovrebbe Funzionare

1. **RF Migliorato:**
   - exp_014 (12.19) > exp_008 (12.45)
   - Advanced features catturano pattern
   - Base più forte per ensemble

2. **Proven Ensemble Strategy:**
   - exp_012 dimostra: ensemble > single model
   - 70/30 ratio testato e funzionante
   - XGB contributo diversità prezioso

3. **Complementarietà:**
   - RF: 25 feature avanzate, cattura pattern complessi
   - XGB: 40 feature base, robusto e stabile
   - Diversi spazi feature = diversi pattern

### Rischi

1. **Std RF Alta:**
   - 4.78 vs 2.24 (RF base exp_008)
   - Potrebbe introdurre instabilità
   - XGB deve compensare

2. **Complessità Non Sempre Vince:**
   - exp_013 (stacking) fallì (12.81)
   - exp_012 (simple) migliore di exp_014 (advanced)
   - Small data preferisce semplicità

3. **Assumption Lineare:**
   - Ensemble lineare potrebbe non essere ottimale
   - Stacking fallì, ma forse pesi diversi funzionano

## File Prodotti

- `submission_exp015.csv` - Predizioni ensemble finale

## Risultato Kaggle

**Score:** _Da compilare dopo submission_

### Post-Mortem

_Da completare dopo risultato Kaggle_

**Se MIGLIORA (< 12.16):**
- ✅ Conferma: RF advanced + ensemble = winning combo
- ✅ Diversità cruciale anche con base model forte
- Next: Provare altri pesi (75/25, 65/35)

**Se SIMILE (12.1-12.2):**
- 🤔 Advanced features marginalmente utili
- 🤔 Ensemble non scala linearmente
- Next: Provare LightGBM invece di XGBoost

**Se PEGGIORA (> 12.2):**
- ❌ Std alta RF (4.78) troppo dannosa
- ❌ XGB non compensa instabilità
- Next: Ridurre peso RF o tornare a RF base

---

**Status:** 🆕 Generato - In attesa submission Kaggle

**Hypothesis:** exp_015 (RF adv + XGB) > exp_012 (RF base + XGB) perché RF component migliorato mantiene diversità ensemble

**Target:** < 12.10 (beat exp_012 di almeno 0.06)
