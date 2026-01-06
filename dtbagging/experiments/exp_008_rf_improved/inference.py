"""Inference per Esperimento 008 - Random Forest Migliorato"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle


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
    """Carica e preprocessa test set."""
    print(f"📂 Caricamento test set: {test_path}")
    
    data = pd.read_csv(test_path, sep=';')
    print(f"   Campioni originali: {len(data)}")
    print(f"   Colonne originali: {len(data.columns)}")
    
    # Preprocessing time-series
    timeseries_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
    
    for col in timeseries_cols:
        if col not in data.columns:
            print(f"   ⚠️  Colonna {col} non trovata!")
            continue
        
        prefix = col.replace('_time_series', '')
        series = data[col].apply(parse_series)
        
        data[f'{prefix}_mean'] = series.apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_min'] = series.apply(lambda x: np.min(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_max'] = series.apply(lambda x: np.max(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_std'] = series.apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
        
        data.drop(columns=[col], inplace=True)
    
    print(f"   Feature dopo preprocessing: {len(data.columns)}")
    
    # Aggiungi colonne mancanti
    if 'Unnamed: 0' not in data.columns:
        data.insert(0, 'Unnamed: 0', range(len(data)))
    
    if 'day' not in data.columns:
        data.insert(1, 'day', range(len(data)))
    
    print(f"   Feature finali: {len(data.columns)}")
    
    # Rimuovi colonna 'id' se presente (non serve per predizione)
    if 'id' in data.columns:
        data = data.drop(columns=['id'])
        print(f"   Rimossa colonna 'id', feature finali: {len(data.columns)}")
    
    return data


def generate_submission(model_path: str, test_path: str, output_path: str):
    """Genera submission.csv per Kaggle usando ensemble di RF."""
    print("="*70)
    print("🎯 INFERENCE - ESPERIMENTO 008 (RF Improved)")
    print("="*70)
    
    # Carica modelli
    print(f"\n📦 Caricamento modelli: {model_path}")
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
    
    print(f"   ✓ {len(models)} modelli caricati!")
    print(f"   Alberi per modello: {models[0].n_estimators}")
    print(f"   Alberi totali: {len(models) * models[0].n_estimators}")
    
    # Carica test set
    test_data = load_and_preprocess_test(test_path)
    
    # Converti a array numpy
    X_test = test_data.values
    
    print(f"\n🤖 Generazione predizioni...")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_test.shape[1]}")
    
    # Predizioni da tutti i modelli (ensemble)
    all_predictions = []
    
    for i, model in enumerate(models):
        pred = model.predict(X_test)
        all_predictions.append(pred)
    
    # Media delle predizioni (ensemble averaging)
    predictions = np.mean(all_predictions, axis=0)
    
    # Statistiche
    print(f"\n📊 Statistiche predizioni:")
    print(f"   Mean: {predictions.mean():.2f}")
    print(f"   Std:  {predictions.std():.2f}")
    print(f"   Min:  {predictions.min():.2f}")
    print(f"   Max:  {predictions.max():.2f}")
    print(f"   Median: {np.median(predictions):.2f}")
    
    # Confronto con exp_002
    print(f"\n📊 Confronto con exp_002 (RF baseline):")
    print(f"   exp_002 mean: 72.90 | exp_008 mean: {predictions.mean():.2f}")
    print(f"   exp_002 std:  2.32  | exp_008 std:  {predictions.std():.2f}")
    print(f"   exp_002 range: 66-79 | exp_008 range: {predictions.min():.0f}-{predictions.max():.0f}")
    print(f"   exp_002 score: 12.95967 | exp_008 score: ???")
    
    # Crea submission
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
    exp_dir = Path(__file__).parent / "experiments" / "exp_008_rf_improved"
    model_path = exp_dir / "final_model_rf_exp008.pkl"
    test_path = Path(__file__).parent / "x_test.csv"
    output_path = exp_dir / "submission_exp008.csv"
    
    generate_submission(str(model_path), str(test_path), str(output_path))
