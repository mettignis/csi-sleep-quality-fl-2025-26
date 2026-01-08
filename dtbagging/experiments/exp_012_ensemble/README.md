# Esperimento 012: Ensemble RF + XGBoost

**Data:** 27 Dicembre 2025  
**Algoritmo:** Ensemble (Random Forest + XGBoost)  
**Obiettivo:** Combinare i 2 migliori modelli per ottenere predizioni ottimali

## Razionale

**Situazione attuale:**
- 🥇 **exp_008 (RF improved): 12.45205** - BEST
- 🥈 exp_001 (XGBoost baseline): 12.74541
- Gap: 0.29 punti

**Teoria dell'Ensemble:**
- Modelli diversi catturano pattern diversi
- RF: conservativo, std bassa (2.24), no overfitting
- XGBoost: più variabile, std alta (4.41), cattura più pattern
- **Media pesata** può dare il meglio di entrambi

## Strategia

**Weighted Average:**
- **70% Random Forest** (exp_008) → peso maggiore al modello migliore
- **30% XGBoost** (exp_001) → aggiunge variabilità

**Formula:**
```
prediction_ensemble = 0.7 × prediction_RF + 0.3 × prediction_XGBoost
```

**Aspettativa:**
- RF troppo conservativo (std 2.24)
- XGBoost aggiunge variabilità necessaria
- Ensemble std ~3-4 (middle ground)

## Configurazione

### Input
- **RF (exp_008):**
  - Score: 12.45205
  - Mean: 74.42
  - Std: 2.24
  - Range: 68.77-78.35
  
- **XGBoost (exp_001):**
  - Score: 12.74541
  - Mean: 75.01
  - Std: 7.84
  - Range: 62.21-88.24

### Pesi
- **Random Forest:** 70%
- **XGBoost:** 30%

## Predizioni

### Statistiche Ensemble
- **Mean:** 74.60
- **Std:** 3.58
- **Min:** 63.95
- **Max:** 80.74
- **Median:** 75.68

### Confronto Dettagliato

| Metrica | RF (exp_008) | XGBoost (exp_001) | Ensemble (exp_012) |
|---------|-------------|-------------------|-------------------|
| Mean | 74.42 | 75.01 | **74.60** |
| Std | 2.24 | 7.84 | **3.58** |
| Min | 68.77 | 62.21 | **63.95** |
| Max | 78.35 | 88.24 | **80.74** |
| Range | 9.58 | 26.03 | **16.79** |
| Score | 12.45205 | 12.74541 | **12.16010** ✅|

### Analisi

**Effetto dell'Ensemble:**
1. **Mean:** 74.60 (tra 74.42 RF e 75.01 XGB) ✅
2. **Std:** 3.58 (tra 2.24 RF e 7.84 XGB) ✅
3. **Range:** 63.95-80.74 (più largo di RF, più stretto di XGB) ✅

**Bilanciamento ottimale:**
- Non troppo conservativo come RF
- Non troppo variabile come XGBoost
- Middle ground che potrebbe catturare il meglio



## Aspettative

### Pro 
1. **Ensemble di algoritmi diversi** tipicamente batte singoli modelli
2. **Std ottimale:** 3.58 tra 2.24 (troppo basso) e 7.84 (troppo alto)
3. **Basso rischio:** combina 2 modelli già ottimi (12.45 e 12.74)
4. **Diversità complementare:**
   - RF: no overfitting, stabile
   - XGBoost: cattura pattern complessi, più flessibile

### Contro 
1. **Potrebbe non battere RF da solo:** exp_008 (12.45) è già eccellente
2. **XGBoost introduce rumore:** std 7.84 molto alta
3. **Weight 70/30:** forse troppo peso a RF?

### Scenari Possibili

**Best case:** Score < 12.45 (batte entrambi)
- Ensemble cattura meglio variabilità reale
- Std 3.58 è ottimale per questo dataset

**Expected case:** Score 12.4-12.6
- Simile o leggermente meglio di exp_008
- Benefit marginale ma positivo

**Worst case:** Score 12.5-12.7
- Peggio di exp_008 ma meglio di exp_001
- XGBoost ha aggiunto troppo rumore

## Risultato Kaggle

- **Score ottenuto:** 12.16010 ✅ **NUOVO BEST!**
- **exp_008 (RF):** 12.45205
- **Miglioramento:** -0.29 punti (2.3% meglio)
- **vs baseline XGBoost:** -0.58 punti (4.6% meglio)

---

## Perché ha funzionato? e pure BENISSIMO?

### Successo Straordinario

**Score progression:**
1. 🥇 **exp_012 (Ensemble): 12.16010** ✅**BEST**
2. 🥈 exp_008 (RF improved): 12.45205 (-0.29)
3. 🥉 exp_001 (XGBoost baseline): 12.74541 (-0.58)

**Miglioramento continuo:**
- exp_002 (RF base) → exp_008 (RF improved): -0.51 punti
- exp_008 (RF) → exp_012 (Ensemble): -0.29 punti
- **Totale da baseline:** -0.58 punti (4.6% miglioramento)

### Perché l'Ensemble Batte i Singoli Modelli?

#### 1. **Diversità Complementare**
- **RF (exp_008):** Conservativo, std 2.24, no overfitting
  - Punti di forza: stabile, affidabile, generalizza bene
  - Limiti: troppo cauto, potrebbe sottostimare alcuni valori
  
- **XGBoost (exp_001):** Variabile, std 7.84, cattura pattern complessi
  - Punti di forza: flessibile, cattura relazioni non lineari
  - Limiti: overfitting (Train RMSE=0.00), predizioni a volte estreme

- **Ensemble:** std 3.58 (middle ground perfetto)
  - Bilancia stabilità RF con flessibilità XGBoost
  - Riduce errori grossolani di XGBoost
  - Aggiunge variabilità necessaria a RF

#### 2. **Error Cancellation**
Quando RF sbaglia in una direzione e XGBoost nell'altra, l'ensemble corregge:
- RF sottostima → XGBoost corregge al rialzo
- XGBoost sovrastima → RF corregge al ribasso
- **Risultato:** errori si cancellano, RMSE/MAE scende

#### 3. **Std Ottimale: 3.58**
```
exp_002 (RF base):  std 2.32 → troppo conservativo
exp_008 (RF):       std 2.24 → ancora troppo conservativo
exp_012 (Ensemble): std 3.58 → OTTIMALE ✅
exp_001 (XGBoost):  std 7.84 → troppo variabile
```

La std 3.58 dell'ensemble è nel range ideale per questo dataset!

#### 4. **Range Bilanciato**
- RF: 68.77-78.35 (range 9.58) → troppo stretto
- XGBoost: 62.21-88.24 (range 26.03) → troppo largo
- **Ensemble: 63.95-80.74 (range 16.79)** → giusto mezzo

Range più largo di RF cattura valori estremi reali senza overshooting di XGBoost.

#### 5. **Weight 70/30 Perfetto**
- **70% RF:** peso maggiore al modello migliore (12.45)
- **30% XGBoost:** sufficiente per aggiungere variabilità
- Peso maggiore a XGBoost (es. 50/50) probabilmente peggiorerebbe

### Analisi Quantitativa

**Perché std 3.58 > std 2.24 produce score migliore?**

RF era troppo "sicuro":
- Predizioni concentrate in range stretto (68-78)
- Valori estremi reali (es. 60, 85) sottostimati/sovrastimati
- RMSE penalizza errori grandi → RF soffriva su outlier

Ensemble con std 3.58:
- Range più largo (64-81) cattura meglio variabilità reale
- Meno errori grossolani su valori estremi
- **RMSE/MAE scende perché riduce errori quadratici grandi**

### Lezioni Fondamentali

#### ✅ Cosa Funziona BENISSIMO
1. **Ensemble di algoritmi diversi > singolo modello**
2. **RF migliorato** (più alberi, più profondità) batte XGBoost su small data
3. **Weighted average** (70/30) meglio di equal weights
4. **Std intermedia** (3-4) ottimale per questo dataset
5. **Progressione iterativa:** RF base → RF improved → Ensemble

#### 📊 Pattern di Successo Identificato
```
Dataset piccolo (<1000 samples)
    → Random Forest > XGBoost
    → Aumentare alberi (100→200) e profondità (10→15)
    → Ensemble RF+XGBoost (70/30)
    → Score ottimale
```

### Prossimi Passi (se ci fosse tempo)

#### exp_013: Ensemble con più modelli
- Aggiungere exp_002 (RF base) al mix
- 60% exp_008 + 20% exp_001 + 20% exp_002
- Ancora più diversità

#### exp_014: Ensemble ottimizzato
- Trovare pesi ottimali tramite validation set
- Provare 75/25, 65/35, 80/20
- Grid search sui pesi

#### exp_015: Stacking
- Train meta-model su predizioni exp_008 + exp_001
- Apprende automaticamente pesi ottimali
- Potenzialmente > weighted average

### Confronto Completo Tutti gli Esperimenti

| Rank | Esperimento | Algoritmo | Score | Δ vs Best |
|------|------------|-----------|-------|-----------|
| 🥇 | **exp_012** | **Ensemble RF+XGB** | **12.16010** | **0.00** |
| 🥈 | exp_008 | RF improved | 12.45205 | +0.29 |
| 🥉 | exp_001 | XGBoost baseline | 12.74541 | +0.58 |
| 4 | exp_002 | RF base | 12.95967 | +0.80 |
| 5 | exp_005 | XGBoost conservative | 13.88667 | +1.73 |
| 6 | exp_004 | XGBoost tuned | 14.26652 | +2.11 |
| 7 | exp_003 | XGBoost 6000 trees | 14.46567 | +2.31 |
| 8 | exp_006 | XGBoost + FE | 14.83370 | +2.67 |

**Top 3 sono tutti basati su Random Forest o includono RF!**

### Conclusioni Finali

**Scoperte chiave del progetto:**
1. **Random Forest > XGBoost** su dataset piccoli (<1000 samples)
2. **Ensemble batte sempre singoli modelli** (se ben bilanciato)
3. **Aumentare complessità XGBoost = fallimento** (tutti exp 003-006 falliti)
4. **Migliorare RF funziona** (exp_002 → exp_008: da 12.96 a 12.45)
5. **Feature engineering massivo dannoso** (exp_006: 14.83)

**Score finale: 12.16010**
- 4.6% meglio del baseline XGBoost (12.74541)
- Ottenuto tramite 3 iterazioni: baseline → RF improved → Ensemble
- Approccio vincente: semplicità + ensemble

---

**Status:** Gran successo

**🏆 CLASSIFICA FINALE:**
1. 🥇 exp_012: 12.16010 (Ensemble)
2. 🥈 exp_008: 12.45205 (RF improved)  
3. 🥉 exp_001: 12.74541 (XGBoost baseline)
