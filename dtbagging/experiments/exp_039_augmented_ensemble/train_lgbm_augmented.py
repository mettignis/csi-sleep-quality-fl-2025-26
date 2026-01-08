"""
ESPERIMENTO 039 - LightGBM su dati AUGMENTED

Training LightGBM su dati augmented (6x samples).
Stesso config di exp_031 ma con più dati.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import pickle

print("="*80)
print("EXP 039: Training LightGBM su dati AUGMENTED")
print("="*80)

# Path ai dati augmented
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "ready_for_flwr_augmented"

print(f"\n📂 Dati augmented: {DATA_DIR}")

# Parametri LightGBM (stessi di exp_031)
LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'max_depth': 8,
    'min_data_in_leaf': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'seed': 42
}

print(f"\n⚙️  Parametri LightGBM:")
for k, v in LGBM_PARAMS.items():
    if k != 'verbose':
        print(f"   {k}: {v}")

# Carica e aggrega tutti i client
print(f"\n📊 Caricamento dati da 9 client...")

all_data = []
for i in range(9):
    client_file = DATA_DIR / f"client_{i}.csv"
    if not client_file.exists():
        print(f"   ⚠️  Client {i} non trovato")
        continue
    
    df = pd.read_csv(client_file)
    print(f"   ✓ Client {i}: {len(df)} samples")
    all_data.append(df)

# 🔄 FEDERATED LEARNING: Allena 9 modelli separati (uno per client)
print(f"\n✅ Dati caricati da {len(all_data)} client")
print(f"\n🎯 Training Federated: 9 modelli separati...")

all_lgbm_models = []
train_maes = []
val_maes = []
best_iters = []

for i, df in enumerate(all_data):
    print(f"\n👤 Client {i}:")
    print(f"   Samples: {len(df)}")
    
    # Split features/target
    feature_cols = [col for col in df.columns 
                    if col not in ['day', 'label', 'Unnamed: 0'] 
                    and df[col].dtype in ['int64', 'float64']]
    
    X = df[feature_cols]
    y = df['label']
    
    # Fill NaN con median
    X = X.fillna(X.median())
    y = y.fillna(y.median())
    
    # Split train/val 80/20
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Crea dataset LightGBM
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    # Training LightGBM per questo client
    model = lgb.train(
        LGBM_PARAMS,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_train, lgb_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0)  # Silent
        ]
    )
    
    # Evaluation
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    
    print(f"   Train MAE: {train_mae:.2f}, Val MAE: {val_mae:.2f}, Best iter: {model.best_iteration}")
    
    all_lgbm_models.append(model)
    train_maes.append(train_mae)
    val_maes.append(val_mae)
    best_iters.append(model.best_iteration)

# Media delle metriche
avg_train_mae = np.mean(train_maes)
avg_val_mae = np.mean(val_maes)
avg_best_iter = np.mean(best_iters)

print(f"\n📊 RISULTATI AGGREGATI (media su 9 client):")
print(f"   Train MAE: {avg_train_mae:.2f}")
print(f"   Val MAE:   {avg_val_mae:.2f}")
print(f"   Best iteration (avg): {avg_best_iter:.1f}")

# Salva TUTTI i 9 modelli per ensemble/bagging
model_file = Path(__file__).parent / "lgbm_augmented_model.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(all_lgbm_models, f)

print(f"\n💾 Modelli salvati: {model_file.name}")
print(f"   Numero modelli FL: {len(all_lgbm_models)}")

print("\n" + "="*80)
print("✅ Training LightGBM FEDERATED completato")
print("="*80)
