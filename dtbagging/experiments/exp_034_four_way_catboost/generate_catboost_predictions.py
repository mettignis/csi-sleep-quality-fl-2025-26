#!/usr/bin/env python3
"""
Generate REAL CatBoost predictions for test set
Processa x_test.csv come i client e applica i 9 modelli CatBoost
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

print("="*80)
print("GENERAZIONE PREDIZIONI CATBOOST REALI PER TEST SET")
print("="*80)

# ============================================================================
# 1. CARICAMENTO MODELLI CATBOOST
# ============================================================================

print("\n📂 Caricamento modelli CatBoost...")
with open('catboost_models_federated_exp034.pkl', 'rb') as f:
    catboost_models = pickle.load(f)
print(f"✅ Caricati {len(catboost_models)} modelli CatBoost")

# ============================================================================
# 2. PROCESSAMENTO X_TEST
# ============================================================================

print("\n📂 Caricamento e processamento x_test.csv...")

BASE_DIR = Path(__file__).parent.parent.parent
test_file = BASE_DIR / "x_test.csv"

# Carica test data (semicolon-separated!)
test_data = pd.read_csv(test_file, sep=';')
print(f"✅ Test data caricato: {len(test_data)} samples, {len(test_data.columns)} colonne")

# Processa come nel training
# Rimuovi colonne vuote
test_data = test_data.dropna(axis=1, how='all')

# Seleziona feature numeriche (same logic as training)
feature_cols = [col for col in test_data.columns 
                if col not in ['day', 'label', 'Unnamed: 0', 'id'] 
                and not col.endswith('_time_series')
                and test_data[col].dtype in ['int64', 'float64']]

print(f"✅ Feature selezionate: {len(feature_cols)} colonne")

# Extract features
X_test = test_data[feature_cols]

# Handle NaN (fill with median from training if needed)
if X_test.isna().any().any():
    print(f"⚠️  Trovati NaN, filling con median...")
    X_test = X_test.fillna(X_test.median())

print(f"✅ X_test preparato: {X_test.shape}")
print(f"   Samples: {len(X_test)}")
print(f"   Features: {X_test.shape[1]}")

# ============================================================================
# 3. GENERAZIONE PREDIZIONI (BAGGING 9 MODELLI)
# ============================================================================

print("\n🎯 Generazione predizioni CatBoost (bagging)...")

# IMPORTANTE: Get feature names from the first model to ensure correct order
expected_features = catboost_models[0].feature_names_
print(f"   Modelli si aspettano {len(expected_features)} features")
print(f"   Test set ha {X_test.shape[1]} features")

# Reorder/align test features to match model training
missing_features = set(expected_features) - set(X_test.columns)
if missing_features:
    print(f"⚠️  Mancano {len(missing_features)} features, le aggiungo con valore 0")
    for feat in missing_features:
        X_test[feat] = 0

# Riordina colonne nello stesso ordine del training
X_test = X_test[expected_features]
print(f"✅ X_test allineato: {X_test.shape}")

catboost_preds = []
for i, model in enumerate(catboost_models):
    pred = model.predict(X_test)
    catboost_preds.append(pred)
    print(f"   Model {i}: mean={np.mean(pred):.2f}, std={np.std(pred):.2f}")

# Bagging: media delle predizioni
catboost_ensemble = np.mean(catboost_preds, axis=0)

print(f"\n✅ CatBoost Bagging (REAL):")
print(f"   Mean: {np.mean(catboost_ensemble):.2f}")
print(f"   Std:  {np.std(catboost_ensemble):.2f}")
print(f"   Min:  {np.min(catboost_ensemble):.2f}")
print(f"   Max:  {np.max(catboost_ensemble):.2f}")

# ============================================================================
# 4. SALVATAGGIO
# ============================================================================

print("\n💾 Salvataggio predizioni CatBoost...")

catboost_sub = pd.DataFrame({
    'id': range(len(catboost_ensemble)),
    'y': catboost_ensemble
})

catboost_sub.to_csv('catboost_predictions_exp034.csv', index=False)
print(f"✅ Salvato: catboost_predictions_exp034.csv")
print(f"   Samples: {len(catboost_sub)}")

print("\n" + "="*80)
print("✅ PREDIZIONI CATBOOST REALI GENERATE")
print("="*80)
print("\n⏭️  Usa questo file per il grid search sui pesi")
