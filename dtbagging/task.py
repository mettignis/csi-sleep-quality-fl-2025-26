"""Task per caricare e preprocessare i dati."""
import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_data(partition_id:  int, test_fraction: float, data_folder: str = "dtbagging"):
    """
    Carica i dati per un client specifico.
    
    Args:
        partition_id:  ID del client (0-8)
        test_fraction:  Frazione di dati per il test
        data_folder:  Cartella contenente la sottocartella ready_for_flwr
    
    Returns:
        train_dmatrix, valid_dmatrix, num_train, num_test
    """
    # Se data_folder è relativo, usa il percorso del file corrente
    if not os.path.isabs(data_folder):
        # Determina il percorso assoluto relativo a questo file
        base_path = Path(__file__).parent
        data_file = base_path / "ready_for_flwr" / f"client_{partition_id}.csv"
    else:
        data_file = Path(data_folder) / "ready_for_flwr" / f"client_{partition_id}.csv"
    
    if not data_file.exists():
        raise FileNotFoundError(f"File non trovato: {data_file}")
    
    # Carica i dati (già uniti e preprocessati)
    data = pd.read_csv(data_file)
    
    # Rimuovi colonne completamente vuote PRIMA di dropna
    data = data.dropna(axis=1, how='all')
    
    # Seleziona le feature numeriche (escludi day, label e le colonne _series se presenti)
    feature_cols = [col for col in data.columns 
                    if col not in ['day', 'label'] 
                    and not col.endswith('_time_series')
                    and data[col].dtype in ['int64', 'float64']]
    
    X = data[feature_cols]
    y = data['label']
    
    # Rimuovi righe con valori mancanti nelle feature selezionate o nella label
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Split temporale (non shufflare per preservare l'ordine temporale)
    split_idx = int(len(X) * (1 - test_fraction))
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    num_train = len(X_train)
    num_test = len(X_test)
    
    # Crea DMatrix per XGBoost
    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    valid_dmatrix = xgb. DMatrix(X_test, label=y_test)
    
    return train_dmatrix, valid_dmatrix, num_train, num_test


def replace_keys(input_dict, match="-", target="_"):
    """Sostituisce ricorsivamente caratteri nelle chiavi del dizionario."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict