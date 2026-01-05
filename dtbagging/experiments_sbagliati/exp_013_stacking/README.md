# Esperimento 013: Stacking con Meta-Learner

**Data:** 29 Dicembre 2025  
**Algoritmo:** Stacking (RF + XGBoost + Ridge Meta-Learner)  
**Obiettivo:** Superare exp_012 (12.16010) usando stacking vero invece di weighted average

## Razionale

**Situazione attuale:**
- 🥇 exp_012 (Ensemble 70/30): 12.16010 - BEST
- 🥈 exp_008 (RF improved): 12.45205
- 🥉 exp_001 (XGBoost baseline): 12.74541

**Limite di exp_012:** Pesi fissi (70% RF, 30% XGB)
- Pesi scelti manualmente
- Non ottimali per ogni predizione
- Non impara dalle performance dei modelli

**Stacking risolve questo problema:**
- **Meta-learner** apprende pesi ottimali dai dati
- **Out-of-fold predictions** prevengono overfitting
- **Combina predizioni in modo intelligente** invece di semplice media

## Strategia

### Architettura Stacking

**Livello 1 - Base Models:**
1. **Random Forest** (exp_008 config)
   - 200 trees, max_depth=15
   - 5 modelli diversi (uno per fold)
   
2. **XGBoost** (exp_001 config)
   - 100 trees, depth=6, lr=0.3
   - 5 modelli diversi (uno per fold)

**Livello 2 - Meta-Learner:**
- **Ridge Regression** (alpha=1.0)
- Input: predizioni di RF e XGBoost
- Output: predizione finale ottimale
- Apprende pesi automaticamente tramite training

### Cross-Validation (5-Fold)

**Perché 5-fold CV?**
- Genera **out-of-fold (OOF) predictions** per training meta-learner
- OOF predictions: ogni sample predetto da modello NON trained su quel sample
- **Previene overfitting** del meta-learner

**Processo:**
```
Fold 1: Train su 80%, predict su 20% → OOF predictions fold 1
Fold 2: Train su 80%, predict su 20% → OOF predictions fold 2
...
Fold 5: Train su 80%, predict su 20% → OOF predictions fold 5

→ Concatena tutti OOF predictions
→ Train meta-learner su OOF predictions
```

## Configurazione

### Base Models
**Random Forest:**
- n_estimators: 200
- max_depth: 15
- min_samples_split: 5
- min_samples_leaf: 2
- max_features: 'sqrt'
- random_state: 42 + fold

**XGBoost:**
- num_boost_round: 100
- max_depth: 6
- eta: 0.3
- subsample: 0.8
- colsample_bytree: 0.8

### Meta-Learner
- **Algoritmo:** Ridge Regression
- **Alpha:** 1.0 (regolarizzazione L2)
- **Input:** [RF_prediction, XGB_prediction]
- **Output:** Final prediction

### Dataset
- **Training:** 999 campioni totali (includes all client data)
- **Test:** 189 campioni
- **Features:** 42 (40 base + Unnamed: 0 + day)
- **CV Splits:** 5 folds

## Training Results

### Out-of-Fold Performance (5-Fold CV)

**Fold Performance:**
- Fold 1: RF RMSE 13.64, XGB RMSE 13.84
- Fold 2: RF RMSE 14.98, XGB RMSE 16.30
- Fold 3: RF RMSE 13.52, XGB RMSE 14.70
- Fold 4: RF RMSE 14.03, XGB RMSE 14.69
- Fold 5: RF RMSE 14.15, XGB RMSE 14.94

**Overall OOF Performance:**
- **RF OOF RMSE:** 14.08
- **XGB OOF RMSE:** 14.92
- **Meta-Learner RMSE:** 13.95 ✅

**Improvement:** Meta-learner RMSE 13.95 < RF 14.08 < XGB 14.92

### Meta-Learner Weights

**Pesi appresi automaticamente:**
```
Final_Prediction = 0.9817 × RF_pred + 0.2050 × XGB_pred - 13.9684
```

**Analisi pesi:**
- **RF weight: 0.9817** → ~98% del peso totale!
- **XGB weight: 0.2050** → ~20% contributo addizionale
- **Intercept: -13.97** → Bias correction

**Insight:** Meta-learner ha confermato che RF è MOLTO superiore a XGBoost su questo dataset, dandogli peso dominante (~98%).

## Predizioni

### Statistiche Test Set

**Base Models (media dei 5 fold):**
- **RF avg:** mean 74.00, std 3.73
- **XGB avg:** mean 74.34, std 10.54

**Stacking (meta-learner):**
- **Mean:** 73.92
- **Std:** 5.55
- **Min:** 52.88
- **Max:** 83.04
- **Median:** 75.46

### Confronto con Altri Esperimenti

| Esperimento | Algoritmo | Mean | Std | Range | Score |
|------------|-----------|------|-----|-------|-------|
| exp_012 (Ensemble) | 70/30 avg | 74.60 | 3.58 | 64-81 | **12.16010** ✅ |
| exp_008 (RF) | RF improved | 74.42 | 2.24 | 69-78 | 12.45205 |
| **exp_013 (Stacking)** | **Meta-learner** | **73.92** | **5.55** | **53-83** | **?** 🤔 |
| exp_001 (XGBoost) | XGBoost | 75.01 | 7.84 | 62-88 | 12.74541 |

### Analisi Differenze

**vs exp_012 (Ensemble 70/30):**
- Mean: 73.92 vs 74.60 (-0.68) → Leggermente più bassa
- Std: 5.55 vs 3.58 (+1.97) → Più variabile
- Range: più ampio (53-83 vs 64-81)

**Differenze chiave:**
1. **Pesi adattivi:** exp_013 usa pesi ~98/20 appresi, vs 70/30 fissi
2. **Bias correction:** intercept -13.97 aggiusta sistematicamente
3. **Più variabilità:** std 5.55 cattura meglio valori estremi

## File Prodotti

- `stacking_model_exp013.pkl` - 5 RF models + 5 XGBoost models + meta-learner
- `submission_exp013.csv` - Predizioni finali da stacking

## Aspettative

### Pro ✅
1. **Pesi ottimali:** Appresi automaticamente dai dati tramite CV
2. **No overfitting:** OOF predictions garantiscono generalizzazione
3. **Meta RMSE migliore:** 13.95 < 14.08 (RF) < 14.92 (XGB) su training
4. **Bias correction:** Intercept -13.97 può migliorare calibrazione
5. **Teoria solida:** Stacking tipicamente > simple ensemble

### Contro ⚠️
1. **Std più alta:** 5.55 vs exp_012 (3.58) → Più variabilità
2. **Mean più bassa:** 73.92 vs exp_012 (74.60) → Potenziale underprediction
3. **Range ampio:** 53-83 include outlier che potrebbero essere errori
4. **Peso XGB basso:** 0.21 (vs 0.30 in exp_012) → Meno diversità?

### Scenari

**Best case:** Score < 12.16 (battere exp_012)
- Meta-learner ha trovato pesi veramente ottimali
- Bias correction migliora calibrazione
- Std 5.55 cattura meglio variabilità reale

**Expected case:** Score 12.1-12.3
- Simile a exp_012
- Beneficio marginale ma positivo
- Conferma superiorità RF

**Worst case:** Score 12.3-12.5
- Std più alta dannosa (più errori grossolani)
- Mean più bassa causa underprediction
- Simple average era meglio

## Differenza Stacking vs Simple Ensemble

| Aspetto | exp_012 (Simple) | exp_013 (Stacking) |
|---------|------------------|-------------------|
| Pesi | Fissi 70/30 | Appresi 98/21 |
| Ottimizzazione | Manuale | Automatica (CV) |
| Bias correction | No | Sì (-13.97) |
| Overfitting risk | Basso | Molto basso (OOF) |
| Complessità | Semplice | Media |
| Interpretabilità | Alta | Media |

## Risultato Kaggle

- **Score atteso:** 12.0-12.3 range
- **Target:** < 12.16010 (battere exp_012)
- **Probabilità successo:** 40-60%

---

## 🔍 ANALISI POST-SUBMISSION

**Data submission:** 29 Dicembre 2025  
**Score Kaggle:** **12.81492** ❌

### Ranking Esperimenti

| Rank | Esperimento | Algoritmo | Score | Differenza da Best |
|------|-------------|-----------|-------|-------------------|
| 🥇 | exp_012 | Ensemble 70/30 | **12.16010** | - |
| 🥈 | exp_008 | RF improved | 12.45205 | +0.29 |
| 🥉 | exp_001 | XGBoost baseline | 12.74541 | +0.58 |
| 4️⃣ | **exp_013** | **Stacking** | **12.81492** | **+0.65** ❌ |

**Risultato:** FALLITO - Peggio sia di exp_012 che di exp_001 baseline

### Perché è Fallito? 🤔

**1. Std Troppo Alta (5.55 vs 3.58)**
- **Problema:** Variabilità eccessiva → errori grossolani su alcuni campioni
- exp_012 aveva std 3.58 (conservativo)
- exp_013 aveva std 5.55 (troppo aggressivo)
- **Impatto:** Predizioni outlier (53-83) probabilmente molto sbagliate

**2. Mean Più Bassa (73.92 vs 74.60)**
- **Problema:** Sistematica underprediction
- Bias correction (-13.97) forse troppo aggressivo
- exp_012: mean 74.60 (vicino alla verità)
- exp_013: mean 73.92 (sottostima)

**3. Pesi Appresi Non Generalizzano**
- **OOF RMSE:** 13.95 (sembrava buono)
- **Test RMSE:** 12.81 (molto peggio!)
- **Problema:** Overfitting della strategia di combinazione
- Pesi 98/20 ottimali per CV, non per test set

**4. XGBoost Sottopesato (20% vs 30%)**
- exp_012 usava 30% XGBoost
- exp_013 usava solo 20% XGBoost
- **Problema:** Meno diversità → meno robusto
- XGBoost contribuisce anche se peggiore

**5. Complessità Non Necessaria**
- Ridge meta-learner aggiunge complessità
- Su dataset piccolo (697 samples) la complessità danneggia
- **Simple is better:** exp_012 con 70/30 fisso era più robusto

### Lezioni Apprese

❌ **Non fare:**
1. **Stacking su dataset piccoli (<1000 samples)**
   - Troppo complesso per generalizzare
   - Simple ensemble più robusto
   
2. **Fidarsi solo di OOF metrics**
   - OOF RMSE 13.95 sembrava promettente
   - Test RMSE 12.81 completamente diverso
   - CV non garantisce generalizzazione

3. **Pesi troppo sbilanciati**
   - 98% RF, 20% XGB elimina diversità
   - exp_012 70/30 aveva più balance

4. **Std alta in predizioni**
   - Std 5.55 troppo rischioso
   - Conservativo (std 2-4) meglio per questo task

✅ **Conferme:**
1. **Simple ensemble (exp_012) è il migliore**
   - 70/30 fisso > pesi appresi
   - Meno parametri = più robusto
   
2. **RF domina, ma XGBoost serve**
   - 30% XGBoost aggiunge valore
   - Non ridurre troppo (20% insufficiente)

3. **Conservatività vince**
   - exp_012 std 3.58 > exp_013 std 5.55
   - exp_008 std 2.24 funzionava bene

### Cosa Provare Ora

**Priorità 1: Variare Pesi Simple Ensemble**
- exp_014: 80% RF + 20% XGB (più conservativo)
- exp_015: 60% RF + 40% XGB (più diversità)
- exp_016: 75% RF + 25% XGB (middle ground)

**Priorità 2: LightGBM al Posto di XGBoost**
- exp_017: 70% RF + 30% LightGBM
- LightGBM spesso > XGBoost su small data
- Mantiene architettura exp_012 (provata vincente)

**Priorità 3: Advanced Features + RF**
- exp_018: RF con advanced time-series features
- Usare `advanced_features.py` (già pronto)
- Solo RF (no XGB - exp_006 fallì con features)
- Feature selection per evitare overfitting

**NON PROVARE:**
- ❌ Stacking più complesso (multi-level)
- ❌ Meta-learner più potente (XGBoost meta)
- ❌ Più base models (troppa complessità)

---

**Status:** ❌ FALLITO - exp_012 (12.16010) rimane BEST

**Conclusione:** Stacking troppo complesso per dataset piccolo. Simple weighted average con pesi fissi (70/30) superiore a pesi appresi con meta-learner. La teoria non sempre batte la pratica su small data.

**NOTA TECNICA:**
Il meta-learner ha dato peso 98% a RF e solo 20% a XGB, confermando che:
1. RF è MOLTO superiore a XGBoost su questo dataset
2. XGB contribuisce poco (ma comunque positivamente con 20%)
3. La combinazione ottimale è quasi "solo RF con piccolo aggiustamento XGB"

**SE FALLISCE:** Considerare exp_014 con:
- LightGBM invece di XGBoost come secondo base model
- Più base models (RF + XGB + LightGBM + Extra Trees)
- Meta-learner più complesso (XGBoost invece di Ridge)
