"""Inference per Esperimento 006 - Feature Engineering"""
import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Importa feature engineering
sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import add_engineered_features


def parse_series(x):
    """Parse time-series string to list of floats"""
    if pd.isna(x):
        return []
    try:
        s = str(x).strip('[]')
        values = [float(v) for v in s.split(',') if v.strip() and float(v.strip()) >= 0]
        return values
    except:
        return []


def load_and_preprocess_test(test_path: str):
    """
    Carica e preprocessa test set con feature engineering.
    DEVE FARE ESATTAMENTE LE STESSE TRASFORMAZIONI DEL TRAINING!
    """
    print(f"📂 Caricamento test set: {test_path}")
    
    # Carica CSV
    data = pd.read_csv(test_path, sep=';')
    print(f"   Campioni originali: {len(data)}")
    print(f"   Colonne originali: {len(data.columns)}")
    
    # Preprocessing time-series (IDENTICO AL TRAINING)
    timeseries_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
    
    for col in timeseries_cols:
        if col not in data.columns:
            print(f"   ⚠️  Colonna {col} non trovata!")
            continue
        
        prefix = col.replace('_time_series', '')
        series = data[col].apply(parse_series)
        
        # Statistiche base
        data[f'{prefix}_mean'] = series.apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_min'] = series.apply(lambda x: np.min(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_max'] = series.apply(lambda x: np.max(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_std'] = series.apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
        
        data.drop(columns=[col], inplace=True)
    
    print(f"   Feature dopo preprocessing: {len(data.columns)}")
    
    # Aggiungi colonne mancanti dal training set
    if 'Unnamed: 0' not in data.columns:
        data.insert(0, 'Unnamed: 0', range(len(data)))
    
    if 'day' not in data.columns:
        # Genera valori day fittizi (il modello li ha visti nel training)
        data.insert(1, 'day', range(len(data)))
    
    # FEATURE ENGINEERING! 🔧
    original_features = len(data.columns)
    data = add_engineered_features(data)
    new_features = len(data.columns)
    print(f"   📊 Feature engineering: {original_features} → {new_features} (+{new_features - original_features})")
    
    print(f"   Feature finali: {len(data.columns)}")
    
    return data


def generate_submission(model_path: str, test_path: str, output_path: str):
    """
    Genera submission.csv per Kaggle.
    """
    print("="*70)
    print("🎯 INFERENCE - ESPERIMENTO 006 (Feature Engineering)")
    print("="*70)
    
    # Carica modello
    print(f"\n📦 Caricamento modello: {model_path}")
    model = xgb.Booster()
    model.load_model(model_path)
    print(f"   ✓ Modello caricato!")
    print(f"   Alberi: {model.num_boosted_rounds()}")
    print(f"   Feature attese: {model.num_features()}")
    print(f"   Feature names: {len(model.feature_names)}")
    
    # Carica e preprocessa test set CON feature engineering
    test_data = load_and_preprocess_test(test_path)
    
    # Riordina colonne per matchare esattamente il modello
    print(f"\n🔄 Matching feature con modello...")
    model_features = model.feature_names
    print(f"   Modello richiede {len(model_features)} features")
    print(f"   Test set ha {len(test_data.columns)} features")
    
    # Verifica che tutte le feature del modello siano presenti
    missing_features = [f for f in model_features if f not in test_data.columns]
    if missing_features:
        print(f"\n❌ ERROR: Feature mancanti nel test set!")
        print(f"   {missing_features[:5]}...")
        return
    
    # Riordina secondo l'ordine del modello
    test_features = test_data[model_features]
    print(f"   ✓ Feature riordinate: {list(test_features.columns)[:5]}...")
    
    # Predizioni
    print(f"\n🤖 Generazione predizioni...")
    dtest = xgb.DMatrix(test_features)
    predictions = model.predict(dtest)
    
    # Statistiche predizioni
    print(f"\n📊 Statistiche predizioni:")
    print(f"   Mean: {predictions.mean():.2f}")
    print(f"   Std:  {predictions.std():.2f}")
    print(f"   Min:  {predictions.min():.2f}")
    print(f"   Max:  {predictions.max():.2f}")
    print(f"   Median: {np.median(predictions):.2f}")
    
    # Crea submission.csv
    submission = pd.DataFrame({
        'id': range(len(predictions)),
        'label': predictions
    })
    
    submission.to_csv(output_path, index=False)
    
    print(f"\n✅ Submission salvata: {output_path}")
    print(f"   Righe: {len(submission)}")
    print("\n🚀 PRONTO PER KAGGLE!")
    print("="*70)


if __name__ == "__main__":
    # Percorsi
    exp_dir = Path(__file__).parent / "experiments" / "exp_006_feature_engineering"
    model_path = exp_dir / "final_model_exp006.json"
    test_path = Path(__file__).parent / "x_test.csv"
    output_path = exp_dir / "submission_exp006.csv"
    
    # Genera submission
    generate_submission(str(model_path), str(test_path), str(output_path))
