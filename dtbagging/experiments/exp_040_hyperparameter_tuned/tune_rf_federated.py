"""
EXP 040 - Random Forest FEDERATED Hyperparameter Tuning
Grid search ESTESA per-client (vero FL) - trova best params per ogni client
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import json
import time

print("="*80)
print("EXP 040: RF FEDERATED Grid Search (PER-CLIENT)")
print("="*80)

# Path ai dati augmented
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "ready_for_flwr_augmented"

print(f"\n📂 Dati augmented: {DATA_DIR}")
print(f"🔐 FEDERATED LEARNING: Grid search separata per ogni client\n")

# Grid search parameters ESTESA (come quella da 3500)
param_grid = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [12, 15, 18, 20, 25],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3, 4]
}

print("\n🔍 Grid Search ESTESA:")
print(f"   n_estimators: {param_grid['n_estimators']}")
print(f"   max_depth: {param_grid['max_depth']}")
print(f"   min_samples_split: {param_grid['min_samples_split']}")
print(f"   min_samples_leaf: {param_grid['min_samples_leaf']}")

combinations_per_client = (len(param_grid['n_estimators']) * 
                           len(param_grid['max_depth']) * 
                           len(param_grid['min_samples_split']) * 
                           len(param_grid['min_samples_leaf']))

print(f"   Combinazioni per client: {combinations_per_client}")
print(f"   Totale (9 clients): {combinations_per_client * 9}")

# Dizionari per salvare risultati
client_models = {}
client_best_params = {}
client_results = {}

print("="*80)
print("FEDERATED GRID SEARCH - TRAINING PER CLIENT")
print("="*80)

# Grid search PER OGNI CLIENT
for client_id in range(9):
    print(f"\n{'='*80}")
    print(f"CLIENT {client_id} - Grid Search")
    print(f"{'='*80}")
    
    # Carica dati client
    client_file = DATA_DIR / f"client_{client_id}.csv"
    if not client_file.exists():
        print(f"⚠️  Client {client_id} non trovato, skip")
        continue
    
    df_client = pd.read_csv(client_file)
    print(f"📊 Samples: {len(df_client)}")
    
    # Prepara features
    feature_cols = [col for col in df_client.columns 
                    if col not in ['day', 'label', 'Unnamed: 0'] 
                    and df_client[col].dtype in ['int64', 'float64']]
    
    X = df_client[feature_cols].fillna(df_client[feature_cols].median())
    y = df_client['label'].fillna(df_client['label'].median())
    
    # Split train/val (80/20) per questo client
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}")
    
    best_score = float('inf')
    best_params = None
    results = []
    
    print(f"\n🎯 Grid Search per Client {client_id}...\n")
    start_time = time.time()
    
    for n_est in param_grid['n_estimators']:
        for max_d in param_grid['max_depth']:
            for min_split in param_grid['min_samples_split']:
                for min_leaf in param_grid['min_samples_leaf']:
                    
                    params = {
                        'n_estimators': n_est,
                        'max_depth': max_d,
                        'min_samples_split': min_split,
                        'min_samples_leaf': min_leaf,
                        'max_features': 'sqrt',
                        'random_state': 42,
                        'n_jobs': -1
                    }
                    
                    # Train on client training set
                    rf = RandomForestRegressor(**params)
                    rf.fit(X_train, y_train)
                    
                    # Evaluate on client validation set
                    y_train_pred = rf.predict(X_train)
                    y_val_pred = rf.predict(X_val)
                    
                    train_mae = mean_absolute_error(y_train, y_train_pred)
                    val_mae = mean_absolute_error(y_val, y_val_pred)
                    
                    results.append({
                        'n_estimators': n_est,
                        'max_depth': max_d,
                        'min_samples_split': min_split,
                        'min_samples_leaf': min_leaf,
                        'train_mae': train_mae,
                        'val_mae': val_mae
                    })
                    
                    if val_mae < best_score:
                        best_score = val_mae
                        best_params = params
                        print(f"   ⭐ Client {client_id}: n_est={n_est}, depth={max_d}, split={min_split}, leaf={min_leaf} | "
                              f"Train: {train_mae:.3f}, Val: {val_mae:.3f}")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Client {client_id} completato in {elapsed/60:.1f} minuti")
    print(f"\n🏆 BEST CONFIG for Client {client_id}:")
    for key, val in best_params.items():
        if key != 'n_jobs':
            print(f"   {key}: {val}")
    print(f"📊 Best Val MAE: {best_score:.3f}")
    
    # Train final model con best params su TUTTI i dati del client
    print(f"\n🎯 Training final model per Client {client_id}...")
    rf_final = RandomForestRegressor(**best_params)
    rf_final.fit(X, y)  # Train su tutti i dati del client
    
    # Salva modello del client
    client_models[client_id] = rf_final
    client_best_params[client_id] = best_params
    client_results[client_id] = results
    
    print(f"✅ Client {client_id} completato\n")

print("\n" + "="*80)
print("FEDERATED TRAINING COMPLETATO - SALVATAGGIO MODELLI")
print("="*80)

# Salva tutti i modelli dei client
EXP_DIR = Path(__file__).parent
models_file = EXP_DIR / "rf_federated_models.pkl"
with open(models_file, 'wb') as f:
    pickle.dump(client_models, f)

print(f"\n💾 Salvati {len(client_models)} modelli RF (uno per client)")
print(f"   File: {models_file.name}")

# Salva best params per client
params_file = EXP_DIR / "rf_federated_best_params.json"
with open(params_file, 'w') as f:
    # Convert params to JSON-serializable format
    params_json = {}
    for client_id, params in client_best_params.items():
        params_json[str(client_id)] = {k: v for k, v in params.items() if k != 'n_jobs'}
    json.dump(params_json, f, indent=2)

print(f"   Best params: {params_file.name}")

# Salva statistiche
print(f"\n📊 STATISTICHE FEDERATED RF:")
val_scores = []
for client_id, results in client_results.items():
    best_val = min(r['val_mae'] for r in results)
    val_scores.append(best_val)
    print(f"   Client {client_id}: Best Val MAE = {best_val:.3f}")

print(f"\n   Average Val MAE: {np.mean(val_scores):.3f}")
print(f"   Std Val MAE: {np.std(val_scores):.3f}")

print("\n" + "="*80)
print("✅ RF FEDERATED Hyperparameter Tuning Completato")
print("="*80)
