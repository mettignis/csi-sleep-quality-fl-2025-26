#!/usr/bin/env python3
"""
ESPERIMENTO 034 - Training CatBoost FEDERATED

Train CatBoost PER CLIENT (9 clients) per four-way ensemble.
CatBoost è ottimo con categorical features e small data.

Strategy:
- CatBoost con ordered boosting (riduce overfitting)
- 9 modelli separati (uno per client)
- Bagging delle predizioni
- Parametri conservativi per small data
"""

from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import mean_absolute_error

print("="*80)
print("ESPERIMENTO 034: Training CatBoost FEDERATED")
print("="*80)

def load_and_preprocess_data_for_client(client_id):
    """
    Carica dati per UN SINGOLO CLIENT (FEDERATED).
    """
    print(f"\n📁 Caricamento dati per Client {client_id}...")
    
    base_path = Path(__file__).parent.parent.parent / "ready_for_flwr"
    client_file = base_path / f"client_{client_id}.csv"
    
    if not client_file.exists():
        raise FileNotFoundError(f"File non trovato: {client_file}")
    
    data = pd.read_csv(client_file)
    print(f"   Client {client_id}: {len(data)} samples, {len(data.columns)} colonne")
    
    # Rimuovi colonne completamente vuote
    data = data.dropna(axis=1, how='all')
    
    # Seleziona feature numeriche (escludi day, label, time_series, Unnamed: 0)
    feature_cols = [col for col in data.columns 
                    if col not in ['day', 'label', 'Unnamed: 0'] 
                    and not col.endswith('_time_series')
                    and data[col].dtype in ['int64', 'float64']]
    
    X = data[feature_cols]
    y = data['label']
    
    # Rimuovi righe con NaN
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"   Features: {len(feature_cols)} colonne")
    print(f"   Samples validi: {len(X)}")
    
    # Split temporale 80/20
    split_idx = int(len(X) * 0.8)
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]
    
    print(f"   Split: {len(X_train)} train / {len(X_val)} validation")
    
    return X_train, y_train, X_val, y_val

# ============================================================================
# TRAINING 9 MODELLI CATBOOST (UNO PER CLIENT)
# ============================================================================

print("\n" + "="*80)
print("TRAINING CATBOOST FEDERATED (9 clients)")
print("="*80)

# Parametri CatBoost ottimizzati per small data
params = {
    'iterations': 200,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_strength': 0.5,
    'bagging_temperature': 0.2,
    'border_count': 32,
    'loss_function': 'MAE',
    'eval_metric': 'MAE',
    'random_seed': 42,
    'verbose': False,
    'early_stopping_rounds': 20,
    'task_type': 'CPU'
}

models = []
train_metrics = []

for client_id in range(9):
    print(f"\n{'='*80}")
    print(f"CLIENT {client_id}")
    print(f"{'='*80}")
    
    # Carica dati per questo client
    X_train, y_train, X_val, y_val = load_and_preprocess_data_for_client(client_id)
    
    # Train CatBoost
    print(f"\n🚀 Training CatBoost per Client {client_id}...")
    
    model = CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False
    )
    
    # Valutazione
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    
    print(f"📊 Client {client_id} Results:")
    print(f"   Train MAE: {train_mae:.2f}")
    print(f"   Val MAE:   {val_mae:.2f}")
    print(f"   Best iteration: {model.best_iteration_}")
    
    models.append(model)
    train_metrics.append({
        'client_id': client_id,
        'train_mae': train_mae,
        'val_mae': val_mae,
        'best_iteration': model.best_iteration_
    })

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TRAINING COMPLETATO - SUMMARY")
print("="*80)

metrics_df = pd.DataFrame(train_metrics)
print("\n📊 Metriche per Client:")
print(metrics_df.to_string(index=False))

print("\n📊 Statistiche Aggregate:")
print(f"   Train MAE (avg): {metrics_df['train_mae'].mean():.2f}")
print(f"   Val MAE (avg):   {metrics_df['val_mae'].mean():.2f}")
print(f"   Train MAE (std): {metrics_df['train_mae'].std():.2f}")
print(f"   Val MAE (std):   {metrics_df['val_mae'].std():.2f}")

# ============================================================================
# SALVATAGGIO MODELLI
# ============================================================================

print("\n💾 SALVATAGGIO MODELLI...")

# Salva modelli
with open('catboost_models_federated_exp034.pkl', 'wb') as f:
    pickle.dump(models, f)
print("✅ Modelli salvati: catboost_models_federated_exp034.pkl")

# Salva metriche
with open('training_metrics_exp034.pkl', 'wb') as f:
    pickle.dump(train_metrics, f)
print("✅ Metriche salvate: training_metrics_exp034.pkl")

print("\n" + "="*80)
print("✅ TRAINING CatBoost COMPLETATO")
print("="*80)
print("\n⏭️  NEXT STEP:")
print("   python inference_exp034.py")
print("="*80)
