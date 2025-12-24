"""Script per generare predizioni sul test set per submission Kaggle."""
print("⏳ Avvio script... (caricamento librerie in corso, attendere ~10-30 secondi)")

import xgboost as xgb
import pandas as pd
import numpy as np
from pathlib import Path
import sys

print("✓ Librerie caricate!")

def preprocess_timeseries_columns(data):
    """
    Preprocessa colonne time-series estraendo statistiche.
    Identico al preprocessing usato in setup_dataset.py
    """
    print("📊 Preprocessing colonne time-series...")
    
    timeseries_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
    
    for col in timeseries_cols:
        if col not in data.columns:
            continue
            
        print(f"   Processando {col}...")
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
    """Carica e preprocessa il file di test."""
    print(f"📂 Caricamento {test_path}...")
    
    # Prova diversi separatori
    try:
        data = pd.read_csv(test_path, sep=';')
    except:
        try:
            data = pd.read_csv(test_path, sep=',')
        except Exception as e:
            print(f"❌ Errore nel caricamento: {e}")
            sys.exit(1)
    
    print(f"   ✓ Caricati {len(data)} campioni con {len(data.columns)} colonne")
    
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
        print(f"   ✓ Aggiunta colonna 'Unnamed: 0' (indice)")
    
    # Rimuovi colonne completamente vuote
    data = data.dropna(axis=1, how='all')
    
    print(f"   ✓ Dopo preprocessing: {len(data.columns)} features")
    
    return data, test_ids

def generate_predictions(model_path, test_path, output_path):
    """Genera predizioni e salva submission file."""
    
    # Carica il modello
    print("🤖 Caricamento modello...")
    model = xgb.Booster()
    model.load_model(str(model_path))
    print(f"   ✓ Modello caricato: {model.num_boosted_rounds()} alberi, {model.num_features()} features")
    
    # Carica e preprocessa test set
    X_test, test_ids = load_and_preprocess_test(test_path)
    
    # Verifica compatibilità features
    print(f"\n🔍 Verifica compatibilità features:")
    print(f"   Modello richiede: {model.num_features()} features")
    print(f"   Test set ha: {len(X_test.columns)} features")
    
    if len(X_test.columns) != model.num_features():
        print(f"\n⚠️  ATTENZIONE: Numero di features diverso!")
        print(f"   Il modello è stato trainato con feature diverse")
        
        # Mostra le prime feature per debug
        print(f"\n   Prime 10 feature del test set:")
        for i, col in enumerate(X_test.columns[:10], 1):
            print(f"      {i}. {col}")
        sys.exit(1)
    
    # IMPORTANTE: Riordina le colonne per matchare l'ordine del modello
    model_features = model.feature_names
    if model_features and list(X_test.columns) != model_features:
        print(f"\n🔧 Riordino colonne per matchare il modello...")
        try:
            X_test = X_test[model_features]
            print(f"   ✓ Colonne riordinate")
        except KeyError as e:
            print(f"   ❌ Errore: feature mancante nel test set - {e}")
            sys.exit(1)
    
    # Crea DMatrix per XGBoost
    dtest = xgb.DMatrix(X_test)
    
    # Genera predizioni
    print(f"\n🎯 Generazione predizioni...")
    predictions = model.predict(dtest)
    
    print(f"   ✓ Predizioni generate: {len(predictions)}")
    print(f"   Statistiche predizioni:")
    print(f"      Min: {predictions.min():.2f}")
    print(f"      Max: {predictions.max():.2f}")
    print(f"      Mean: {predictions.mean():.2f}")
    print(f"      Std: {predictions.std():.2f}")
    
    # Crea DataFrame per submission
    submission = pd.DataFrame({
        'id': test_ids,
        'label': predictions
    })
    
    # Salva submission file
    submission.to_csv(output_path, index=False)
    print(f"\n💾 Submission salvata: {output_path}")
    print(f"   Formato: id,label")
    print(f"   Righe: {len(submission)}")
    
    # Mostra prime righe
    print(f"\n📋 Prime 5 predizioni:")
    print(submission.head().to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ Inference completata!")
    print("="*70)
    print(f"\n📤 Ora puoi caricare '{output_path}' su Kaggle!")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 INFERENCE SCRIPT - GENERAZIONE PREDIZIONI KAGGLE")
    print("="*70)
    print()
    
    # Percorsi
    project_dir = Path(__file__).parent
    model_path = project_dir / "final_model.json"
    test_path = project_dir / "test_data/x_test.csv"
    output_path = project_dir / "submission.csv"
    
    print("📁 Configurazione:")
    print(f"   Modello:     {model_path.name}")
    print(f"   Test set:    {test_path.name}")
    print(f"   Output:      {output_path.name}")
    print()
    
    # Verifica esistenza file
    if not model_path.exists():
        print(f"❌ Modello non trovato: {model_path}")
        print("   Esegui prima il training!")
        sys.exit(1)
    
    if not test_path.exists():
        print(f"❌ Test set non trovato: {test_path}")
        print(f"   Assicurati che il file sia in: {test_path}")
        sys.exit(1)
    
    # Genera predizioni
    generate_predictions(model_path, test_path, output_path)
