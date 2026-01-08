"""
ESPERIMENTO 039 - Random Forest su dati AUGMENTED

Training RF-advanced features su dati augmented (6x samples).
Stesso config di exp_014 ma con più dati.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle

print("="*80)
print("EXP 039: Training Random Forest su dati AUGMENTED")
print("="*80)

# Path ai dati augmented
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "ready_for_flwr_augmented"

print(f"\n📂 Dati augmented: {DATA_DIR}")

# Parametri RF (stessi di exp_014)
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

print(f"\n Parametri RF:")
for k, v in RF_PARAMS.items():
    print(f"   {k}: {v}")

# Carica client
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

all_rf_models = []
train_maes = []
val_maes = []

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
    
    # Training Random Forest per questo client
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)
    
    # Evaluation
    y_train_pred = rf.predict(X_train)
    y_val_pred = rf.predict(X_val)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    
    print(f"   Train MAE: {train_mae:.2f}, Val MAE: {val_mae:.2f}")
    
    all_rf_models.append(rf)
    train_maes.append(train_mae)
    val_maes.append(val_mae)

# Media delle metriche
avg_train_mae = np.mean(train_maes)
avg_val_mae = np.mean(val_maes)

print(f"\n📊 RISULTATI AGGREGATI (media su 9 client):")
print(f"   Train MAE: {avg_train_mae:.2f}")
print(f"   Val MAE:   {avg_val_mae:.2f}")

# Salva TUTTI i 9 modelli per ensemble/bagging
model_file = Path(__file__).parent / "rf_augmented_model.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(all_rf_models, f)

print(f"\nModelli salvati: {model_file.name}")
print(f"   Numero modelli FL: {len(all_rf_models)}")
print(f"   Alberi per modello: {RF_PARAMS['n_estimators']}")
print(f"   Alberi totali: {len(all_rf_models) * RF_PARAMS['n_estimators']}")

print("\n" + "="*80)
print("Training RF FEDERATED completato")
print("="*80)
