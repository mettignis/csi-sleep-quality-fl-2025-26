#!/usr/bin/env python3
"""
ESPERIMENTO 016 - Training LightGBM

Train LightGBM su dataset completo (793 samples) con base features.
Da usare in three-way ensemble con RF advanced + XGBoost.

Strategy:
- LightGBM con GOSS (Gradient-Based One-Side Sampling)
- Base features (40) come XGBoost
- Config ottimizzato per small data
- Leaf-wise growth limitato per evitare overfitting
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import mean_absolute_error

print("="*80)
print("ESPERIMENTO 016: Training LightGBM")
print("="*80)

def load_and_preprocess_data():
    """Carica dati già preprocessati da ready_for_flwr."""
    print("\n📁 Caricamento dati da ready_for_flwr...")
    
    all_data = []
    base_path = Path(__file__).parent / "ready_for_flwr"
    
    # Carica tutti i 9 client
    for client_id in range(9):
        client_file = base_path / f"client_{client_id}.csv"
        
        if not client_file.exists():
            print(f"   ⚠️  File non trovato: {client_file}")
            continue
        
        df = pd.read_csv(client_file)
        print(f"   Client {client_id}: {len(df)} samples, {len(df.columns)} colonne")
        all_data.append(df)
    
    # Combina tutti i dati
    data = pd.concat(all_data, ignore_index=True)
    print(f"   ✅ Totale samples: {len(data)}")
    
    # Rimuovi colonne completamente vuote
    data = data.dropna(axis=1, how='all')
    
    # Rimuovi 'day' se presente
    if 'day' in data.columns:
        data = data.drop(columns=['day'])
    
    # Rimuovi 'act_activeTime' se presente (non era nel training XGBoost)
    if 'act_activeTime' in data.columns:
        data = data.drop(columns=['act_activeTime'])
    
    # Rimuovi colonne time-series se ancora presenti
    time_series_cols = [col for col in data.columns if col.endswith('_time_series')]
    if time_series_cols:
        data = data.drop(columns=time_series_cols)
    
    # Seleziona solo feature numeriche (escludi label)
    feature_cols = [col for col in data.columns 
                    if col != 'label' 
                    and data[col].dtype in ['int64', 'float64']]
    
    # Gestisci valori mancanti
    print("\n🔧 Gestione valori mancanti...")
    for col in feature_cols:
        if data[col].isna().sum() > 0:
            median_val = data[col].median()
            fill_val = median_val if not pd.isna(median_val) else 0
            data[col] = data[col].fillna(fill_val)
            print(f"   - {col}: {data[col].isna().sum()} NaN -> riempiti con {fill_val:.2f}")
    
    # Separa features e target
    X = data[feature_cols]
    y = data['label']
    
    # Rimuovi righe con label NaN
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Rinomina colonne per LightGBM (rimuovi caratteri speciali JSON)
    print("\n🔧 Rinomina colonne per LightGBM...")
    X.columns = [col.replace(':', '_').replace('[', '_').replace(']', '_').replace('"', '').replace(',', '_') 
                 for col in X.columns]
    
    print(f"\n   ✅ Features finali: {X.shape[1]}")
    print(f"   ✅ Samples finali: {len(X)}")
    
    return X, y

def train_lightgbm(X, y):
    """Train LightGBM con config ottimizzato per small data."""
    print("\n🚀 Training LightGBM...")
    
    # Config ottimizzato per small data (793 samples)
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,           # Default, buono per piccoli dati
        'max_depth': 6,             # Come XGBoost
        'learning_rate': 0.1,       # Come XGBoost
        'feature_fraction': 0.8,    # Come colsample_bytree XGBoost
        'bagging_fraction': 0.8,    # Come subsample XGBoost
        'bagging_freq': 5,          # Ogni 5 iterazioni
        'min_data_in_leaf': 20,     # Evita foglie troppo piccole
        'lambda_l1': 0.0,           # L1 regularization
        'lambda_l2': 0.0,           # L2 regularization
        'min_gain_to_split': 0.0,   # Minimum gain per split
        'verbose': -1,              # Silenzioso
        'seed': 42
    }
    
    print("   Parametri:")
    print(f"   - num_leaves: {params['num_leaves']}")
    print(f"   - max_depth: {params['max_depth']}")
    print(f"   - learning_rate: {params['learning_rate']}")
    print(f"   - feature_fraction: {params['feature_fraction']}")
    print(f"   - bagging_fraction: {params['bagging_fraction']}")
    print(f"   - min_data_in_leaf: {params['min_data_in_leaf']}")
    
    # Crea dataset LightGBM
    train_data = lgb.Dataset(X, label=y)
    
    # Train con early stopping su training set (no validation split per massimizzare dati)
    print("\n   Training...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data],
        valid_names=['train'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, verbose=False),
            lgb.log_evaluation(period=10)
        ]
    )
    
    # Valuta su training set
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    print(f"\n   📊 Training MAE: {mae:.4f}")
    
    return model

def main():
    # Carica dati
    X, y = load_and_preprocess_data()
    
    # Train LightGBM
    model = train_lightgbm(X, y)
    
    # Salva modello
    output_path = Path(__file__).parent / "experiments" / "exp_016_three_way_ensemble" / "lgbm_model_exp016.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Salvataggio modello...")
    model.save_model(str(output_path))
    print(f"   ✅ Salvato: {output_path}")
    
    # Salva anche feature names per inference
    feature_names_path = output_path.parent / "lgbm_feature_names_exp016.pkl"
    with open(feature_names_path, 'wb') as f:
        pickle.dump(list(X.columns), f)
    print(f"   ✅ Feature names salvati: {feature_names_path}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETATO")
    print("="*80)
    print(f"\nModello LightGBM pronto per ensemble exp_016")
    print(f"Features: {len(X.columns)}")
    print(f"Training samples: {len(X)}")

if __name__ == "__main__":
    main()
