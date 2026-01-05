# Esperimenti Centralizzati (Non Federated)

**Motivo spostamento:** Questi esperimenti violano i requisiti di Federated Learning della challenge.

## Problema

Gli esperimenti in questa cartella caricano dati da **TUTTI i 9 clients contemporaneamente** con un loop `for client_id in range(9):`, concatenando tutti i dati prima del training. Questo è **apprendimento centralizzato**, non federated learning.

### Codice problematico:
```python
# ❌ CENTRALIZZATO - SBAGLIATO
for client_id in range(9):
    data_path = Path("clients_data") / f"group{client_id}"
    # ... carica dati ...
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)  # Tutti i 999 samples insieme!
```

### Codice corretto (Federated):
```python
# ✅ FEDERATED - CORRETTO
def train_client(partition_id):
    # Carica SOLO i dati del client specifico (5 users del suo gruppo)
    data_path = Path("clients_data") / f"group{partition_id}"
    # ... train solo su questi dati ...
    return model

# Poi aggrega i modelli (FedAvg, bagging, etc.)
```

## Esperimenti in questa cartella

### exp_013_stacking
- **Score Kaggle:** 12.81492 MAE
- **Problema:** `train_stacking_exp013.py` linea 36: `for client_id in range(9)`
- **Impatto:** Stacking meta-learner trainato su tutti i 999 samples centralizzati

### exp_016_three_way_ensemble
- **Score Kaggle:** 11.73074 MAE (sarebbe stato ottimo!)
- **Problema:** `train_lightgbm_exp016.py` linea 34: `for client_id in range(9)`
- **Impatto:** LightGBM trainato su tutti i 999 samples centralizzati

## Esperimenti derivati (anche eliminati)

Gli esperimenti che dipendono da questi sono anche invalidi:
- **exp_017:** Ensemble che usa exp_016 (centralizzato)
- **exp_020:** Weight optimization che carica tutti i 9 clients
- **exp_021:** Stacking meta-learner che carica tutti i 9 clients

## Note

Federated Learning richiede che:
1. Ogni client alleni **SOLO sui propri dati** (5 users del suo gruppo)
2. I modelli vengono **aggregati** (FedAvg, bagging, ensemble)
3. **NON** si concatenano dati di più clients prima del training

Questi esperimenti sono conservati per documentazione, ma **NON sono validi** per la challenge.
