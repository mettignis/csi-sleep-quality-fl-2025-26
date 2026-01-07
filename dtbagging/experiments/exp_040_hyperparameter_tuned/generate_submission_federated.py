#!/usr/bin/env python3
"""
EXP 040 - Generate Submission FEDERATED
Usa modelli FL tuned (uno per client) e fa bagging predictions
"""

import pandas as pd
import numpy as np
import pickle
import ast
from pathlib import Path

def parse_series(s):
    """Parse time series string to numeric array, filtering out None/NaN values"""
    if pd.isna(s):
        return []
    try:
        if isinstance(s, str):
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                # Filter out None values and convert to float, handle any conversion errors
                result = []
                for x in parsed:
                    try:
                        if x is not None and not pd.isna(x):
                            result.append(float(x))
                    except (ValueError, TypeError):
                        continue
                return result
        return []
    except:
        return []

print("="*80)
print("EXP 040: Generazione submission FEDERATED (BAGGING)")
print("="*80)

# Path
BASE_DIR = Path(__file__).parent.parent.parent
EXP_DIR = Path(__file__).parent

# Load FEDERATED models (9 RF models + 9 LightGBM models)
print(f"\n📂 Caricamento modelli FEDERATED...")

with open(EXP_DIR / "rf_federated_models.pkl", 'rb') as f:
    rf_models = pickle.load(f)
print(f"✅ Caricati {len(rf_models)} modelli RF (uno per client)")

with open(EXP_DIR / "lgbm_federated_models.pkl", 'rb') as f:
    lgbm_models = pickle.load(f)
print(f"✅ Caricati {len(lgbm_models)} modelli LightGBM (uno per client)")

# Load test data
test_file = BASE_DIR / "x_test.csv"
print(f"\n📂 Caricamento test data: {test_file.name}")
df_test = pd.read_csv(test_file, sep=';')
print(f"✅ Test data: {len(df_test)} samples, {len(df_test.columns)} colonne")

# Parse time series function
def parse_series(s):
    """Parse time series string to list of floats"""
    if pd.isna(s) or s == '':
        return []
    if isinstance(s, str):
        try:
            return ast.literal_eval(s)
        except:
            return []
    return []

# Extract time series statistics (CRITICAL - must match training!)
print(f"\n🔄 Preprocessing test data...")
test_data = df_test.copy()

def safe_stat(arr, func):
    """Safely compute statistic, handling empty arrays"""
    if len(arr) == 0:
        return np.nan
    try:
        return func(arr)
    except:
        return np.nan

for col in ['hr_time_series', 'resp_time_series', 'stress_time_series']:
    if col in test_data.columns:
        prefix = col.replace('_time_series', '')
        print(f"   Estraendo statistiche da {col}...")
        
        series = test_data[col].apply(parse_series)
        
        # Compute statistics safely
        test_data[f'{prefix}_mean'] = series.apply(lambda x: safe_stat(x, np.mean))
        test_data[f'{prefix}_min'] = series.apply(lambda x: safe_stat(x, np.min))
        test_data[f'{prefix}_max'] = series.apply(lambda x: safe_stat(x, np.max))
        test_data[f'{prefix}_std'] = series.apply(lambda x: safe_stat(x, np.std))
        
        test_data = test_data.drop(columns=[col])

# Drop non-feature columns
drop_cols = ['day', 'Unnamed: 0']
for col in drop_cols:
    if col in test_data.columns:
        test_data = test_data.drop(columns=[col])

# Handle NaN
if test_data.isnull().any().any():
    print(f"⚠️  Trovati NaN, filling con median...")
    test_data = test_data.fillna(test_data.median())

X_test = test_data
print(f"✅ X_test preparato: {X_test.shape}")

# FEDERATED BAGGING: Average predictions from all client models
print(f"\n🎯 FEDERATED BAGGING - Predizioni da 9 modelli per algoritmo...")

# RF predictions (bagging)
print(f"\n📊 Random Forest Bagging (9 client models):")
rf_predictions = []
for client_id, model in rf_models.items():
    # Align features
    model_features = model.feature_names_in_
    missing_features = set(model_features) - set(X_test.columns)
    if missing_features:
        for feat in missing_features:
            X_test[feat] = 0
    
    X_test_aligned = X_test[model_features]
    pred = model.predict(X_test_aligned)
    rf_predictions.append(pred)
    print(f"   Client {client_id}: mean={pred.mean():.2f}, std={pred.std():.2f}")

pred_rf_bagging = np.mean(rf_predictions, axis=0)
print(f"   🔹 RF BAGGING: mean={pred_rf_bagging.mean():.2f}, std={pred_rf_bagging.std():.2f}")

# LightGBM predictions (bagging)
print(f"\n📊 LightGBM Bagging (9 client models):")
lgbm_predictions = []
for client_id, model in lgbm_models.items():
    # Align features
    model_features = model.feature_name_
    missing_features = set(model_features) - set(X_test.columns)
    if missing_features:
        for feat in missing_features:
            X_test[feat] = 0
    
    X_test_aligned = X_test[model_features]
    pred = model.predict(X_test_aligned)
    lgbm_predictions.append(pred)
    print(f"   Client {client_id}: mean={pred.mean():.2f}, std={pred.std():.2f}")

pred_lgbm_bagging = np.mean(lgbm_predictions, axis=0)
print(f"   🔹 LGBM BAGGING: mean={pred_lgbm_bagging.mean():.2f}, std={pred_lgbm_bagging.std():.2f}")

# Final ensemble: combine RF bagging + LGBM bagging
pred_ensemble = 0.65 * pred_rf_bagging + 0.35 * pred_lgbm_bagging

print(f"\n📊 FINAL ENSEMBLE (65% RF + 35% LGBM):")
print(f"   Mean: {pred_ensemble.mean():.2f}")
print(f"   Std:  {pred_ensemble.std():.2f}")
print(f"   Min:  {pred_ensemble.min():.2f}")
print(f"   Max:  {pred_ensemble.max():.2f}")

# Create submission
submission = pd.DataFrame({
    'id': range(len(pred_ensemble)),
    'label': pred_ensemble
})

submission_file = EXP_DIR / "submission_exp040_federated.csv"
submission.to_csv(submission_file, index=False)
print(f"\n✅ Submission salvata: {submission_file.name}")

print("\n" + "="*80)
print("📊 CONFRONTO CON ESPERIMENTI PRECEDENTI")
print("="*80)
print(f"\n🔸 exp_031 (3-way FL): 11.92674 MAE")
print(f"🔸 exp_039 (augmented): 11.89067 MAE")
print(f"\n🔸 exp_040 (FEDERATED TUNED): Mean {pred_ensemble.mean():.2f}, Std {pred_ensemble.std():.2f}")
print(f"   Atteso: 11.60-11.80 MAE")
print(f"   Grid search ESTESA per-client (400 RF + 500 LGBM × 9 clients)")
print(f"   Vero Federated Learning con bagging!")

print("\n" + "="*80)
print("✅ SUBMISSION FEDERATED PRONTA")
print("="*80)
