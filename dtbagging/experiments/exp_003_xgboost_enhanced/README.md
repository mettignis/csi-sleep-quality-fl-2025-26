# Esperimento 003: XGBoost Potenziato

**Data:** 22 Dicembre 2025  
**Algoritmo:** XGBoost (Federated Learning)  
**Obiettivo:** Migliorare Exp 001 aumentando round e alberi

## Configurazione

### Parametri Training
- **Round FL:** 30 (era 10 in exp_001)
- **Alberi per client:** 200 (era 100 in exp_001)
- **Alberi totali:** 6000 (30 round × 9 clients × 200 trees / round)
- **Max depth:** 6
- **Learning rate:** 0.3
- **Subsample:** 0.8
- **Colsample_bytree:** 0.8

### Dataset
- **Training:** 697 campioni totali (9 gruppi)
- **Test:** 189 campioni
- **Features:** 40

## Training Details

```
🚀 XGBOOST POTENZIATO - ESPERIMENTO 003
Round FL: 30 (era 10)
Alberi per client: 200 (era 100)
Alberi totali finali: 6,000
```

### Performance Training (ultimo round - round 30)
- Client 0: Test RMSE 13.91
- Client 1: Test RMSE 13.79
- Client 2: Test RMSE 25.25
- Client 3: Test RMSE 10.21
- Client 4: Test RMSE 18.32
- Client 5: Test RMSE 15.52
- Client 6: Test RMSE 13.69
- Client 7: Test RMSE 17.19
- Client 8: Test RMSE 14.87

Training RMSE: 0.00 (perfetto fit sui dati di training)

## Predizioni

### Statistiche Output
- **Min:** 51.10
- **Max:** 92.99
- **Mean:** 77.74
- **Std:** 6.58

### Confronto con altri esperimenti

| Esperimento | Min | Max | Mean | Std | Kaggle Score |
|------------|-----|-----|------|-----|--------------|
| exp_001 (XGBoost baseline) | 62.21 | 88.24 | 75.12 | 4.41 | **12.74541** ✅ |
| exp_002 (Random Forest) | 66.58 | 79.00 | 72.90 | 2.32 | 12.95967 |
| **exp_003 (XGBoost enhanced)** | 51.10 | 92.99 | 77.74 | 6.58 | **14.46567** ❌ |

## File Prodotti

- `final_model_exp003.json` - Modello XGBoost con 6000 alberi
- `submission_exp003.csv` - Predizioni per Kaggle

## Note

### Miglioramenti rispetto a exp_001
- **3x più round:** 30 vs 10 → Più cicli di apprendimento federato
- **2x più alberi per client:** 200 vs 100 → Modelli più complessi
- **Risultato:** Range predizioni più ampio (51-93 vs 62-88)
- **Std maggiore:** 6.58 vs 4.41 → Predizioni più variegate

### Aspettative
- Modello più complesso dovrebbe catturare pattern più sottili
- Range più ampio potrebbe essere positivo se cattura valori estremi reali
- Std più alta: può essere meglio se RF era troppo conservativo (std 2.32)

### Risultato Kaggle
- **Score ottenuto:** 14.46567 ❌
- **Baseline exp_001:** 12.74541 
- **Peggioramento:** +1.72 punti (13.6% peggio)

---

## Perché ha fallito?

### Problema: OVERFITTING SEVERO
1. **Train RMSE = 0.00** → Memorizzazione perfetta dei dati di training
2. **Test RMSE alto** → Scarsa generalizzazione
3. **Range troppo ampio** (51-93) → Predizioni estreme non realistiche

### Cause
- **Troppi alberi (6000)** senza regolarizzazione adeguata
- **Learning rate alto (0.3)** → apprendimento troppo aggressivo
- **Nessun controllo overfitting** (min_child_weight=1)

### Imparato che
**Più parametri ≠ Modello migliore**  
**Regolarizzazione è fondamentale** su dataset piccoli (697 campioni)  
**Train RMSE = 0.00 è red flag** → Overfitting garantito

### Soluzione: Esperimento 004
- Learning rate ↓ (0.3 → 0.1)
- Max depth ↑ (6 → 8) + regolarizzazione
- Min child weight ↑ (1 → 3)

---

**Status:** Esperimento fallito - Non usare questo modello
