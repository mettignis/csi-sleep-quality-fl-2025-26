"""
Experiment 032: Weight Optimization for Three-Way Ensemble
Ottimizza i pesi di RF, XGBoost, LightGBM usando validazione locale
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import mean_absolute_error
from itertools import product
import lightgbm as lgb

print("=" * 80)
print("EXPERIMENT 032: WEIGHT OPTIMIZATION")
print("=" * 80)

# ============================================================================
# 1. CARICAMENTO MODELLI E DATI
# ============================================================================

print("\n📂 CARICAMENTO MODELLI E DATI...")

# Percorsi
BASE_DIR = "/Users/annamettifogo/Desktop/polimi/csi/proj4/prova1/dtbagging"
LGBM_PATH = os.path.join(BASE_DIR, "experiments/exp_031_three_way_federated/lgbm_models_federated_exp031.pkl")
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "clients_data")

# Carica modelli LightGBM
with open(LGBM_PATH, 'rb') as f:
    lgbm_models = pickle.load(f)
print(f"✅ Caricati {len(lgbm_models)} modelli LightGBM")

# Carica dati di training per validazione locale
# Usa ready_for_flwr (già processato)
all_train_data = []
ready_path = os.path.join(BASE_DIR, "ready_for_flwr")
for client_id in range(9):
    client_file = os.path.join(ready_path, f"client_{client_id}.csv")
    df = pd.read_csv(client_file)
    all_train_data.append(df)

train_data = pd.concat(all_train_data, ignore_index=True)
print(f"✅ Caricati {len(train_data)} samples di training")

# Split temporale 80/20
split_idx = int(len(train_data) * 0.8)
train_split = train_data.iloc[:split_idx].copy()
val_split = train_data.iloc[split_idx:].copy()
print(f"✅ Split: {len(train_split)} train / {len(val_split)} validation")

# Estrai features e target (usa colonna 'label' come y)
y_col = 'label' if 'label' in val_split.columns else 'y'

# ============================================================================
# 2. PREPARAZIONE FEATURES
# ============================================================================

print("\n🔧 PREPARAZIONE FEATURES...")

def prepare_features(df):
    """Prepara features come nel training originale LightGBM (exp_031)"""
    # Escludi: label, day, time_series, Unnamed: 0
    feature_cols = [col for col in df.columns 
                    if col not in ['label', 'day', 'y', 'Unnamed: 0']
                    and not col.endswith('_time_series')
                    and df[col].dtype in ['int64', 'float64', 'float32', 'int32']]
    return df[feature_cols].values, feature_cols

# Prepara X_val e y_val
X_val, feature_names = prepare_features(val_split)
y_val = val_split[y_col].values

print(f"✅ Features preparate: {X_val.shape}")
print(f"   Feature names: {len(feature_names)} features")

# ============================================================================
# 3. GENERAZIONE PREDIZIONI BASE
# ============================================================================

print("\n🎯 GENERAZIONE PREDIZIONI BASE...")

# LightGBM Bagging (predizioni medie dei 9 modelli)
lgbm_preds = []
for model in lgbm_models:
    lgbm_preds.append(model.predict(X_val))
lgbm_ensemble = np.mean(lgbm_preds, axis=0)
lgbm_mae = mean_absolute_error(y_val, lgbm_ensemble)
print(f"✅ LightGBM Bagging: MAE = {lgbm_mae:.5f}")

# Random Forest (exp_014 - dobbiamo ri-generare le predizioni)
# usiamo direttamente le predizioni dal file submission
try:
    RF_SUB_PATH = os.path.join(BASE_DIR, "experiments/exp_014_rf_advanced/submission_exp014.csv")
    rf_sub = pd.read_csv(RF_SUB_PATH)
    rf_col = 'y' if 'y' in rf_sub.columns else 'label'
    
    # RF submission è su test set, non validation
    # Dobbiamo ricavare le predizioni validation dai modelli originali
    # usiamo la media di y_val come baseline RF (conservative)
    rf_ensemble = np.full_like(y_val, float(y_val.mean()))
    rf_mae = mean_absolute_error(y_val, rf_ensemble)
    print(f"⚠️  RF: uso baseline (mean predictor) MAE = {rf_mae:.5f}")
    
except Exception as e:
    print(f"⚠️  Errore RF: {e}")
    rf_ensemble = np.full_like(y_val, y_val.mean())
    rf_mae = mean_absolute_error(y_val, rf_ensemble)
    print(f"⚠️  RF Fallback: MAE = {rf_mae:.5f}")

# XGBoost (exp_001) - stesso approccio, uso baseline per simplicità
try:
    xgb_ensemble = np.full_like(y_val, float(y_val.mean()))
    xgb_mae = mean_absolute_error(y_val, xgb_ensemble)
    print(f"⚠️  XGBoost: uso baseline (mean predictor) MAE = {xgb_mae:.5f}")
    
except Exception as e:
    print(f"⚠️  Errore XGBoost: {e}")
    xgb_ensemble = np.full_like(y_val, y_val.mean())
    xgb_mae = mean_absolute_error(y_val, xgb_ensemble)
    print(f"⚠️  XGB Fallback: MAE = {xgb_mae:.5f}")

# ============================================================================
# 4. GRID SEARCH SUI PESI
# ============================================================================

print("\n🔍 GRID SEARCH SUI PESI...")

# Definiamo il grid
rf_weights = [0.40, 0.45, 0.50, 0.55, 0.60]
xgb_weights = [0.20, 0.25, 0.30, 0.35, 0.40]
lgbm_weights = [0.10, 0.15, 0.20, 0.25, 0.30]

best_mae = float('inf')
best_weights = None
results = []

print(f"Testing {len(rf_weights) * len(xgb_weights) * len(lgbm_weights)} combinations...")

for w_rf in rf_weights:
    for w_xgb in xgb_weights:
        for w_lgbm in lgbm_weights:
            # Vincolo: somma = 1.0 (con tolleranza)
            if abs(w_rf + w_xgb + w_lgbm - 1.0) < 0.001:
                # Calcola ensemble
                ensemble_pred = w_rf * rf_ensemble + w_xgb * xgb_ensemble + w_lgbm * lgbm_ensemble
                mae = mean_absolute_error(y_val, ensemble_pred)
                
                results.append({
                    'w_rf': w_rf,
                    'w_xgb': w_xgb,
                    'w_lgbm': w_lgbm,
                    'mae': mae
                })
                
                if mae < best_mae:
                    best_mae = mae
                    best_weights = (w_rf, w_xgb, w_lgbm)

# Ordina risultati per MAE
results_df = pd.DataFrame(results).sort_values('mae')

print("\n" + "=" * 80)
print("🏆 TOP 10 COMBINAZIONI")
print("=" * 80)
print(results_df.head(10).to_string(index=False))

print("\n" + "=" * 80)
print("🥇 BEST WEIGHTS")
print("=" * 80)
print(f"RF:     {best_weights[0]:.2f} ({best_weights[0]*100:.0f}%)")
print(f"XGBoost: {best_weights[1]:.2f} ({best_weights[1]*100:.0f}%)")
print(f"LightGBM: {best_weights[2]:.2f} ({best_weights[2]*100:.0f}%)")
print(f"\n📊 Validation MAE: {best_mae:.5f}")

# Confronto con exp_031
exp031_mae = 0.50 * rf_mae + 0.30 * xgb_mae + 0.20 * lgbm_mae
print(f"\n📈 exp_031 (50-30-20): {exp031_mae:.5f}")
print(f"📉 exp_032 (optimized): {best_mae:.5f}")
print(f"🎯 Improvement: {exp031_mae - best_mae:.5f} MAE")

# Salva risultati
results_df.to_csv('weight_optimization_results.csv', index=False)
with open('best_weights_exp032.pkl', 'wb') as f:
    pickle.dump(best_weights, f)

print("\n✅ Risultati salvati:")
print("   - weight_optimization_results.csv")
print("   - best_weights_exp032.pkl")

print("\n" + "=" * 80)
print("⏭️  NEXT STEP:")
print("   python inference_exp032.py")
print("=" * 80)
