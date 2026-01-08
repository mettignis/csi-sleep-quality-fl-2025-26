"""Inference per Esperimento 004 - XGBoost Tuned"""
print("⏳ Avvio script... (caricamento librerie)")

import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path

def load_test_data():
    """Carica e prepara i dati di test"""
    print("\n📂 Caricamento dati di test...")
    
    # Percorso al file di test
    test_path = Path(__file__).parent / "x_test.csv"
    
    if not test_path.exists():
        raise FileNotFoundError(f"File di test non trovato: {test_path}")
    
    # Carica dati 
    data = pd.read_csv(test_path, sep=';')
    print(f"   Shape iniziale: {data.shape}")
    
    # Salva ID per submission
    test_ids = data['id'].values
    
    # Time-series preprocessing
    print("   📊 Preprocessing time-series...")
    timeseries_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
    
    for col in timeseries_cols:
        if col not in data.columns:
            continue
            
        prefix = col.replace('_time_series', '')
        
        def parse_series(x):
            if pd.isna(x):
                return []
            try:
                s = str(x).strip('[]')
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
    
    # Rimuovi colonna id
    if 'id' in data.columns:
        data.drop(columns=['id'], inplace=True)
    
    print(f"   Shape dopo preprocessing: {data.shape}")
    
    # Aggiungi Unnamed: 0 se mancante
    if 'Unnamed: 0' not in data.columns:
        data.insert(0, 'Unnamed: 0', range(len(data)))
        print(f"   ✓ Aggiunta colonna 'Unnamed: 0'")
    
    return data, test_ids

def main():
    print("="*70)
    print("🔮 INFERENCE - ESPERIMENTO 004")
    print("="*70)
    
    # Carica modello da experiments/
    model_path = Path(__file__).parent / "experiments" / "exp_004_xgboost_tuned" / "final_model_exp004.json"
    print(f"\n📥 Caricamento modello da: experiments/exp_004_xgboost_tuned/{model_path.name}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_path}")
    
    model = xgb.Booster()
    model.load_model(str(model_path))
    
    print(f"   Alberi: {model.num_boosted_rounds()}")
    print(f"   Features: {model.num_features()}")
    
    # Carica dati di test
    X_test, test_ids = load_test_data()
    
    # Verifica e riordina colonne
    model_features = model.feature_names
    print(f"\n🔍 Verifica features...")
    print(f"   Model features: {len(model_features)}")
    print(f"   Test features: {len(X_test.columns)}")
    
    missing_cols = set(model_features) - set(X_test.columns)
    extra_cols = set(X_test.columns) - set(model_features)
    
    if missing_cols:
        print(f"   ⚠️  Missing in test: {missing_cols}")
    if extra_cols:
        print(f"   ⚠️  Extra in test: {extra_cols}")
    
    # Riordina per matching esatto
    X_test = X_test[model_features]
    print(f"   ✓ Colonne riordinate")
    
    # Converti a DMatrix
    dtest = xgb.DMatrix(X_test)
    
    # Predizioni
    print(f"\n🎯 Generazione predizioni per {len(X_test)} campioni...")
    predictions = model.predict(dtest)
    
    # Statistiche predizioni
    print(f"\n📊 Statistiche predizioni:")
    print(f"   Min: {predictions.min():.2f}")
    print(f"   Max: {predictions.max():.2f}")
    print(f"   Mean: {predictions.mean():.2f}")
    print(f"   Std: {predictions.std():.2f}")
    
    # Crea submission
    submission = pd.DataFrame({
        'id': test_ids,
        'label': predictions
    })
    
    # Salva submission in experiments/
    exp_dir = Path(__file__).parent / "experiments" / "exp_004_xgboost_tuned"
    output_path = exp_dir / "submission_exp004.csv"
    submission.to_csv(output_path, index=False)
    
    print(f"\n💾 Submission salvata: experiments/exp_004_xgboost_tuned/{output_path.name}")
    print(f"   Righe: {len(submission)}")
    print(f"\n✅ INFERENCE COMPLETATA!")
    print("="*70)
    print("\n🚀 Carica submission_exp004.csv su Kaggle per vedere il punteggio!")

if __name__ == "__main__":
    main()
