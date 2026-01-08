#!/usr/bin/env python3
"""
ESPERIMENTO 031 - Training LightGBM FEDERATED

Train LightGBM PER CLIENT (9 clients) con base features.
Da usare in three-way ensemble con RF advanced + XGBoost.

DIFFERENZA DA exp_016 (centralizzato):
- exp_016: caricava TUTTI i 9 clients insieme (loop + concat)
- exp_031: allena 9 modelli LGBM separati, uno per client
- Aggregazione: bagging dei 9 modelli (media predizioni)

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
print("ESPERIMENTO 031: Training LightGBM FEDERATED")
print("="*80)

def load_and_preprocess_data_for_client(client_id):
    """
    Carica dati per UN SINGOLO CLIENT (FEDERATED).
    
    Args:
        client_id: ID del client (0-8)
        
    Returns:
        X_train, y_train, X_val, y_val
    """
    print(f"\n📁 Caricamento dati per Client {client_id}...")
    
    base_path = Path(__file__).parent / "ready_for_flwr"
    client_file = base_path / f"client_{client_id}.csv"
    
    if not client_file.exists():
        raise FileNotFoundError(f"File non trovato: {client_file}")
    
    data = pd.read_csv(client_file)
    print(f"   Client {client_id}: {len(data)} samples, {len(data.columns)} colonne")
    
    # Rimuovi colonne completamente vuote
    data = data.dropna(axis=1, how='all')
    
    # Seleziona feature numeriche (escludi day, label, time_series)
    feature_cols = [col for col in data.columns 
                    if col not in ['day', 'label'] 
                    and not col.endswith('_time_series')
                    and data[col].dtype in ['int64', 'float64']]
    
    X = data[feature_cols]
    y = data['label']
    
    # Sanitizza nomi colonne (LightGBM non supporta caratteri speciali JSON)
    import re
    X.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in X.columns]
    
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
    
    print(f"   Train: {len(X_train)}, Validation: {len(X_val)}")
    
    return X_train, y_train, X_val, y_val


def train_lightgbm_federated():
    """
    Training FEDERATED: allena 9 modelli LightGBM, uno per client.
    """
    print("\n" + "="*80)
    print("🌳 FEDERATED LIGHTGBM TRAINING")
    print("="*80)
    print("\nStrategy:")
    print("  - 9 clients → 9 modelli LightGBM separati")
    print("  - Ogni modello allena SOLO sui dati del suo client")
    print("  - Predizione finale: media (bagging) dei 9 modelli")
    print()
    
    # Parametri LightGBM ottimizzati per small data
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'max_depth': 6,
        'min_child_samples': 10,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1
    }
    
    num_clients = 9
    all_models = []
    all_metrics = []
    
    for client_id in range(num_clients):
        print(f"\n{'='*70}")
        print(f"👤 CLIENT {client_id}")
        print(f"{'='*70}")
        
        # Carica dati del client
        X_train, y_train, X_val, y_val = load_and_preprocess_data_for_client(client_id)
        
        # Crea dataset LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train
        print("\n🏋️  Training LightGBM...")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(period=50)
            ]
        )
        
        # Valutazione
        train_pred = model.predict(X_train, num_iteration=model.best_iteration)
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        print(f"\n📊 Metriche Client {client_id}:")
        print(f"   Train MAE: {train_mae:.4f}")
        print(f"   Val MAE:   {val_mae:.4f}")
        print(f"   Best iteration: {model.best_iteration}")
        
        all_models.append(model)
        all_metrics.append({
            'client_id': client_id,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'best_iter': model.best_iteration
        })
    
    # Statistiche finali
    print("\n" + "="*80)
    print("📈 RIEPILOGO FEDERATED TRAINING")
    print("="*80)
    
    avg_train_mae = np.mean([m['train_mae'] for m in all_metrics])
    avg_val_mae = np.mean([m['val_mae'] for m in all_metrics])
    
    print(f"\nMedia Train MAE: {avg_train_mae:.4f}")
    print(f"Media Val MAE:   {avg_val_mae:.4f}")
    print(f"\nDettagli per client:")
    print(f"{'Client':<8} {'Train':<10} {'Val':<10} {'N_train':<10} {'N_val':<8} {'Iter':<6}")
    print("-" * 60)
    for m in all_metrics:
        print(f"{m['client_id']:<8} {m['train_mae']:<10.4f} {m['val_mae']:<10.4f} "
              f"{m['n_train']:<10} {m['n_val']:<8} {m['best_iter']:<6}")
    
    # Salva modelli
    exp_dir = Path(__file__).parent / "experiments" / "exp_031_three_way_federated"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = exp_dir / "lgbm_models_federated_exp031.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(all_models, f)
    
    print(f"\nSalvati {len(all_models)} modelli in: {model_file}")
    
    # Salva metriche
    metrics_file = exp_dir / "training_metrics_exp031.pkl"
    with open(metrics_file, 'wb') as f:
        pickle.dump(all_metrics, f)
    
    print(f"Salvate metriche in: {metrics_file}")
    
    return all_models, all_metrics


if __name__ == "__main__":
    models, metrics = train_lightgbm_federated()
    print("\n✅ Training completato!")
    print("\n📝 Next step: Run inference_exp031.py per generare predizioni")
