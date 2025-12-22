"""Inference con Random Forest per Kaggle submission."""
print("⏳ Avvio script RF... (caricamento librerie in corso)")

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle

print("✓ Librerie caricate!")

# Importa funzioni di preprocessing
import sys
sys.path.insert(0, str(Path(__file__).parent))
from inference import preprocess_timeseries_columns

def load_rf_ensemble(model_path):
    """Carica l'ensemble di Random Forest."""
    with open(model_path, 'rb') as f:
        models = pickle.load(f)
    return models

def predict_with_ensemble(models, X):
    """Genera predizioni usando tutti i modelli (ensemble averaging)."""
    predictions = []
    
    for model in models:
        pred = model.predict(X)
        predictions.append(pred)
    
    # Media delle predizioni
    return np.mean(predictions, axis=0)

def generate_predictions_rf(model_path, test_path, output_path):
    """Genera predizioni con Random Forest."""
    
    print("🤖 Caricamento ensemble Random Forest...")
    models = load_rf_ensemble(model_path)
    print(f"   ✓ Ensemble caricato: {len(models)} modelli")
    print(f"   ✓ Alberi totali: {sum(m.n_estimators for m in models)}")
    
    # Carica test set
    print(f"📂 Caricamento {test_path}...")
    try:
        data = pd.read_csv(test_path, sep=';')
    except:
        data = pd.read_csv(test_path, sep=',')
    
    print(f"   ✓ Caricati {len(data)} campioni con {len(data.columns)} colonne")
    
    # Salva ID
    test_ids = data['id'].copy()
    
    # Preprocessa
    print("📊 Preprocessing colonne time-series...")
    data = preprocess_timeseries_columns(data)
    
    if 'id' in data.columns:
        data.drop(columns=['id'], inplace=True)
    
    # Aggiungi Unnamed: 0
    if 'Unnamed: 0' not in data.columns:
        data.insert(0, 'Unnamed: 0', range(len(data)))
        print(f"   ✓ Aggiunta colonna 'Unnamed: 0'")
    
    data = data.dropna(axis=1, how='all')
    
    print(f"   ✓ Dopo preprocessing: {len(data.columns)} features")
    
    # RF non ha feature_names_in_, usa semplicemente le colonne del DataFrame
    # Assumiamo che l'ordine sia corretto (stesso preprocessing di training)
    
    # Genera predizioni
    print(f"\n🎯 Generazione predizioni (ensemble di {len(models)} modelli)...")
    X_test = data.values
    predictions = predict_with_ensemble(models, X_test)
    
    print(f"   ✓ Predizioni generate: {len(predictions)}")
    print(f"   Statistiche:")
    print(f"      Min: {predictions.min():.2f}")
    print(f"      Max: {predictions.max():.2f}")
    print(f"      Mean: {predictions.mean():.2f}")
    print(f"      Std: {predictions.std():.2f}")
    
    # Crea submission
    submission = pd.DataFrame({
        'id': test_ids,
        'label': predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"\n💾 Submission salvata: {output_path}")
    print(f"\n📋 Prime 5 predizioni:")
    print(submission.head().to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ Inference completata!")
    print("="*70)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌲 INFERENCE RANDOM FOREST - KAGGLE SUBMISSION")
    print("="*70)
    print()
    
    project_dir = Path(__file__).parent
    model_path = project_dir / "final_model_rf.pkl"
    test_path = project_dir / "x_test.csv"
    output_path = project_dir / "submission_rf.csv"
    
    print("📁 Configurazione:")
    print(f"   Modello:     {model_path.name}")
    print(f"   Test set:    {test_path.name}")
    print(f"   Output:      {output_path.name}")
    print()
    
    if not model_path.exists():
        print(f"❌ Modello non trovato: {model_path}")
        print("   Esegui prima: python train_rf.py")
        sys.exit(1)
    
    if not test_path.exists():
        print(f"❌ Test set non trovato: {test_path}")
        sys.exit(1)
    
    generate_predictions_rf(model_path, test_path, output_path)
    print(f"\n📤 Ora puoi caricare '{output_path}' su Kaggle!")
