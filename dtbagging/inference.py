"""
Script per generare predizioni sul test set per submission Kaggle
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def preprocess_timeseries_columns(data):
    """
    Preprocessa colonne time-series estraendo statistiche.
    Identico al preprocessing usato per il setup del dataset.
    """
    
    timeseries_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
    
    for col in timeseries_cols:
        if col not in data.columns:
            continue
            
        prefix = col.replace('_time_series', '')
        
        # Converti stringhe a liste di numeri
        def parse_series(x):
            if pd.isna(x):
                return []
            try:
                # Rimuovi parentesi quadre e splitta
                s = str(x).strip('[]')
                # Splitta e converti a float, ignorando valori -1 o -2
                values = [float(v) for v in s.split(',') if v.strip() and float(v.strip()) >= 0]
                return values
            except:
                return []
        
        series = data[col].apply(parse_series)
        
        # Estrai statistiche
        data[f'{prefix}_mean'] = series.apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_min'] = series.apply(lambda x: np.min(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_max'] = series.apply(lambda x: np.max(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_std'] = series.apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
        
        # Rimuovi colonna originale
        data.drop(columns=[col], inplace=True)
    
    return data

def load_and_preprocess_test(test_path):
    # Prova diversi separatori
    try:
        data = pd.read_csv(test_path, sep=';')
    except:
        try:
            data = pd.read_csv(test_path, sep=',')
        except Exception as e:
            print(f"Errore nel caricamento: {e}")
            sys.exit(1)
        
    # Salva gli ID per la submission
    test_ids = data['id'].copy()
    
    # Preprocessa time-series
    data = preprocess_timeseries_columns(data)
    
    # Rimuovi colonna id (non è una feature)
    if 'id' in data.columns:
        data.drop(columns=['id'], inplace=True)
    
    # Aggiungi colonna Unnamed: 0 (era nei dati di training come indice)
    if 'Unnamed: 0' not in data.columns:
        data.insert(0, 'Unnamed: 0', range(len(data)))
    
    # Rimuovi colonne completamente vuote
    data = data.dropna(axis=1, how='all')
        
    return data, test_ids

def generate_predictions(model_path, test_path, output_path):    
    # Carica il modello
    model = xgb.Booster()
    model.load_model(str(model_path))
    
    # Carica e preprocessa test set
    X_test, test_ids = load_and_preprocess_test(test_path)
    
    # Verifica compatibilità features
    if len(X_test.columns) != model.num_features():
        # Mostra le prime feature per debug
        print(f"Prime 10 feature del test set:")
        for i, col in enumerate(X_test.columns[:10], 1):
            print(f"{i}. {col}")
        sys.exit(1)
    
    # Riordina le colonne per rispettare l'ordine del modello
    model_features = model.feature_names
    if model_features and list(X_test.columns) != model_features:
        try:
            X_test = X_test[model_features]
        except KeyError as e:
            print(f"Errore: feature mancante nel test set - {e}")
            sys.exit(1)
    
    # Crea DMatrix per XGBoost
    dtest = xgb.DMatrix(X_test)
    
    # Genera predictions
    predictions = model.predict(dtest)
    
    print(f"Predizioni generate: {len(predictions)}")
    print(f"Statistiche predizioni:")
    print(f"Min: {predictions.min():.2f}")
    print(f"Max: {predictions.max():.2f}")
    print(f"Mean: {predictions.mean():.2f}")
    print(f"Std: {predictions.std():.2f}")
    
    # Crea DataFrame per submission
    submission = pd.DataFrame({
        'id': test_ids,
        'label': predictions
    })
    
    # Salva submission file
    submission.to_csv(output_path, index=False)
    print(f"\Submission salvata: {output_path}")
    
if __name__ == "__main__":    
    # Prepara i path
    project_dir = Path(__file__).parent
    model_path = project_dir / "final_model.json"
    test_path = project_dir / "x_test.csv"
    output_path = project_dir / "submission.csv"
    
    # Verifica esistenza file
    if not model_path.exists():
        print(f"Modello non trovato: {model_path}")
        sys.exit(1)
    
    if not test_path.exists():
        print(f"Test set non trovato: {test_path}")
        sys.exit(1)
    
    # Genera prediction
    generate_predictions(model_path, test_path, output_path)
