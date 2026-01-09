"""
EXP 040 - LightGBM FEDERATED Hyperparameter Tuning
Grid search ESTESA per-client (vero FL) - trova best params per ogni client
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import pickle
import json
import time

print("="*80)
print("EXP 040: LightGBM FEDERATED Grid Search (PER-CLIENT)")
print("="*80)

# Path ai dati augmented
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "ready_for_flwr_augmented"

print(f"\n📂 Dati augmented: {DATA_DIR}")
print(f"🔐 FEDERATED LEARNING: Grid search separata per ogni client\n")

# Grid search parameters ESTESA
param_grid = {
    'num_leaves': [25, 31, 40, 50, 60],
    'learning_rate': [0.02, 0.03, 0.04, 0.05, 0.06],
    'max_depth': [6, 8, 10, 12],
    'n_estimators': [100, 200, 300, 400, 500]
}

print("\n🔍 Grid Search ESTESA:")
print(f"   num_leaves: {param_grid['num_leaves']}")
print(f"   learning_rate: {param_grid['learning_rate']}")
print(f"   max_depth: {param_grid['max_depth']}")
print(f"   n_estimators: {param_grid['n_estimators']}")

combinations_per_client = (len(param_grid['num_leaves']) * 
                           len(param_grid['learning_rate']) * 
                           len(param_grid['max_depth']) * 
                           len(param_grid['n_estimators']))

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
    
    for num_l in param_grid['num_leaves']:
        for lr in param_grid['learning_rate']:
            for max_d in param_grid['max_depth']:
                for n_est in param_grid['n_estimators']:
                    
                    params = {
                        'objective': 'regression',
                        'metric': 'mae',
                        'num_leaves': num_l,
                        'learning_rate': lr,
                        'max_depth': max_d,
                        'n_estimators': n_est,
                        'feature_fraction': 0.8,
                        'bagging_fraction': 0.8,
                        'bagging_freq': 5,
                        'verbose': -1,
                        'random_state': 42
                    }
                    
                    # Train con early stopping
                    model = lgb.LGBMRegressor(**params)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                    )
                    
                    y_val_pred = model.predict(X_val)
                    val_score = mean_absolute_error(y_val, y_val_pred)
                    
                    best_iter = model.best_iteration_
                    
                    results.append({
                        'num_leaves': num_l,
                        'learning_rate': lr,
                        'max_depth': max_d,
                        'n_estimators': n_est,
                        'best_iteration': best_iter,
                        'val_mae': val_score
                    })
                    
                    if val_score < best_score:
                        best_score = val_score
                        best_params = params.copy()
                        best_params['best_iteration'] = best_iter
                        print(f"   ⭐ Client {client_id}: leaves={num_l}, lr={lr:.2f}, depth={max_d}, n_est={n_est} | "
                              f"Iter: {best_iter}, Val: {val_score:.3f}")
    
    elapsed = time.time() - start_time
    print(f"\nClient {client_id} completato in {elapsed/60:.1f} minuti")
    print(f"\n🏆 BEST CONFIG for Client {client_id}:")
    for key, val in best_params.items():
        if key not in ['verbose', 'random_state']:
            print(f"   {key}: {val}")
    print(f"📊 Best Val MAE: {best_score:.3f}")
    
    # Train final model con best params su TUTTI i dati del client
    print(f"\n🎯 Training final model per Client {client_id}...")
    lgbm_final = lgb.LGBMRegressor(**best_params)
    lgbm_final.fit(
        X, y,  # Train su tutti i dati del client
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Salva modello del client
    client_models[client_id] = lgbm_final
    client_best_params[client_id] = best_params
    client_results[client_id] = results
    
    print(f"✅ Client {client_id} completato\n")

print("\n" + "="*80)
print("FEDERATED TRAINING COMPLETATO - SALVATAGGIO MODELLI")
print("="*80)

# Salva tutti i modelli dei client
EXP_DIR = Path(__file__).parent
models_file = EXP_DIR / "lgbm_federated_models.pkl"
with open(models_file, 'wb') as f:
    pickle.dump(client_models, f)

print(f"\n💾 Salvati {len(client_models)} modelli LightGBM (uno per client)")
print(f"   File: {models_file.name}")

# Salva best params per client
params_file = EXP_DIR / "lgbm_federated_best_params.json"
with open(params_file, 'w') as f:
    # Convert params to JSON-serializable format
    params_json = {}
    for client_id, params in client_best_params.items():
        params_json[str(client_id)] = {k: v for k, v in params.items() if k not in ['verbose', 'random_state']}
    json.dump(params_json, f, indent=2)

print(f"   Best params: {params_file.name}")

# Salva statistiche
print(f"\n📊 STATISTICHE FEDERATED LightGBM:")
val_scores = []
for client_id, results in client_results.items():
    best_val = min(r['val_mae'] for r in results)
    val_scores.append(best_val)
    print(f"   Client {client_id}: Best Val MAE = {best_val:.3f}")

print(f"\n   Average Val MAE: {np.mean(val_scores):.3f}")
print(f"   Std Val MAE: {np.std(val_scores):.3f}")

print("\n" + "="*80)
print("✅ LightGBM FEDERATED Hyperparameter Tuning Completato")
print("="*80)
