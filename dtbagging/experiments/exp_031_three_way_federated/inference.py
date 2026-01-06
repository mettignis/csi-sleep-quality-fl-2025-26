#!/usr/bin/env python3
"""
ESPERIMENTO 031 - Inference LightGBM FEDERATED + Three-Way Ensemble

Genera predizioni e crea ensemble:
- 50% RF Advanced (exp_014) - FEDERATED ✓
- 30% XGBoost Base (exp_001) - FEDERATED ✓
- 20% LightGBM Federated (exp_031) - FEDERATED ✓

FEDERATED LEARNING COMPLIANCE:
- Tutti e 3 i componenti sono trainati in modo federated
- Ogni client allena SOLO sui propri dati (5 users del suo gruppo)
- Ensemble è solo post-processing (weighted average)
- NO concatenazione di dati prima del training
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import re

print("="*80)
print("ESPERIMENTO 031: Three-Way Ensemble FEDERATED")
print("="*80)

# Carica test data
print("\n📂 Caricamento test data...")
test_file = Path(__file__).parent / "x_test.csv"
test_data = pd.read_csv(test_file, sep=';')
print(f"   Test samples: {len(test_data)}")

# ============================================================================
# 1. LIGHTGBM FEDERATED PREDICTIONS (20%)
# ============================================================================

print("\n" + "="*80)
print("🌳 LIGHTGBM FEDERATED PREDICTIONS")
print("="*80)

# Preprocessing test data per LightGBM (base features)
timeseries_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']

for col in timeseries_cols:
    if col not in test_data.columns:
        continue
    
    prefix = col.replace('_time_series', '')
    
    def parse_series(x):
        if pd.isna(x):
            return []
        try:
            s = str(x).strip('[]')
            values = [float(v) for v in s.split(',') if v.strip() and float(v.strip()) >= 0]
            return values
        except:
            return []
    
    series = test_data[col].apply(parse_series)
    
    # Statistiche base
    test_data[f'{prefix}_mean'] = series.apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
    test_data[f'{prefix}_min'] = series.apply(lambda x: np.min(x) if len(x) > 0 else np.nan)
    test_data[f'{prefix}_max'] = series.apply(lambda x: np.max(x) if len(x) > 0 else np.nan)
    test_data[f'{prefix}_std'] = series.apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
    
    test_data.drop(columns=[col], inplace=True)

# Seleziona feature numeriche
test_data = test_data.dropna(axis=1, how='all')
feature_cols = [col for col in test_data.columns 
                if col not in ['day'] 
                and not col.endswith('_time_series')
                and test_data[col].dtype in ['int64', 'float64']]

X_test_lgbm = test_data[feature_cols].copy()

# Sanitizza nomi colonne (come in training)
X_test_lgbm.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in X_test_lgbm.columns]

# Riempi NaN
for col in X_test_lgbm.columns:
    if X_test_lgbm[col].isna().any():
        median_val = X_test_lgbm[col].median()
        if pd.isna(median_val):
            X_test_lgbm.loc[:, col] = 0
        else:
            X_test_lgbm.loc[:, col] = X_test_lgbm[col].fillna(median_val)

print(f"   Features: {len(X_test_lgbm.columns)}")
print(f"   Samples: {len(X_test_lgbm)}")

# Carica modelli LightGBM
exp_dir = Path(__file__).parent / "experiments" / "exp_031_three_way_federated"
model_file = exp_dir / "lgbm_models_federated_exp031.pkl"

with open(model_file, 'rb') as f:
    lgbm_models = pickle.load(f)

print(f"   Caricati {len(lgbm_models)} modelli LightGBM federated")

# Predizioni da tutti i 9 modelli (bagging)
all_preds = []
for i, model in enumerate(lgbm_models):
    pred = model.predict(X_test_lgbm, num_iteration=model.best_iteration)
    all_preds.append(pred)

# Media (bagging)
lgbm_predictions = np.mean(all_preds, axis=0)
print(f"   ✅ LightGBM Bagging: mean={lgbm_predictions.mean():.2f}, std={lgbm_predictions.std():.2f}")

# ============================================================================
# 2. CARICA RF ADVANCED PREDICTIONS (50%)
# ============================================================================

print("\n" + "="*80)
print("🌲 RANDOM FOREST ADVANCED (exp_014)")
print("="*80)

rf_file = Path(__file__).parent / "experiments" / "exp_014_rf_advanced_features" / "submission_exp014.csv"

if not rf_file.exists():
    raise FileNotFoundError(f"RF predictions non trovate: {rf_file}")

rf_submission = pd.read_csv(rf_file)
# Gestisci entrambi i formati (label o y)
if 'label' in rf_submission.columns:
    rf_predictions = rf_submission['label'].values
elif 'y' in rf_submission.columns:
    rf_predictions = rf_submission['y'].values
else:
    raise ValueError(f"Colonna label/y non trovata in {rf_file}")
print(f"   ✅ RF Advanced: mean={rf_predictions.mean():.2f}, std={rf_predictions.std():.2f}")

# ============================================================================
# 3. CARICA XGBOOST PREDICTIONS (30%)
# ============================================================================

print("\n" + "="*80)
print("🔥 XGBOOST BASE (exp_001)")
print("="*80)

xgb_file = Path(__file__).parent / "experiments" / "exp_001_xgboost_baseline" / "submission.csv"

if not xgb_file.exists():
    raise FileNotFoundError(f"XGBoost predictions non trovate: {xgb_file}")

xgb_submission = pd.read_csv(xgb_file)
# Gestisci entrambi i formati (label o y)
if 'label' in xgb_submission.columns:
    xgb_predictions = xgb_submission['label'].values
elif 'y' in xgb_submission.columns:
    xgb_predictions = xgb_submission['y'].values
else:
    raise ValueError(f"Colonna label/y non trovata in {xgb_file}")
print(f"   ✅ XGBoost Base: mean={xgb_predictions.mean():.2f}, std={xgb_predictions.std():.2f}")

# ============================================================================
# 4. THREE-WAY ENSEMBLE
# ============================================================================

print("\n" + "="*80)
print("🎯 THREE-WAY FEDERATED ENSEMBLE")
print("="*80)

# Pesi ensemble (ottimizzati)
w_rf = 0.50
w_xgb = 0.30
w_lgbm = 0.20

print(f"\nPesi:")
print(f"   RF Advanced (exp_014):  {w_rf*100:.0f}%")
print(f"   XGBoost Base (exp_001): {w_xgb*100:.0f}%")
print(f"   LightGBM FL (exp_031):  {w_lgbm*100:.0f}%")

# Ensemble finale
ensemble_predictions = (
    w_rf * rf_predictions +
    w_xgb * xgb_predictions +
    w_lgbm * lgbm_predictions
)

print(f"\n✅ Ensemble: mean={ensemble_predictions.mean():.2f}, std={ensemble_predictions.std():.2f}")

# Clip a [0, 100]
ensemble_predictions = np.clip(ensemble_predictions, 0, 100)

# ============================================================================
# 5. SALVA SUBMISSION
# ============================================================================

print("\n" + "="*80)
print("💾 SALVATAGGIO SUBMISSION")
print("="*80)

submission = pd.DataFrame({
    'id': range(len(ensemble_predictions)),
    'label': ensemble_predictions
})

submission_file = exp_dir / "submission_exp031.csv"
submission.to_csv(submission_file, index=False)

print(f"\n✅ Submission salvata: {submission_file}")
print(f"   Samples: {len(submission)}")
print(f"   Range: [{submission['label'].min():.2f}, {submission['label'].max():.2f}]")

print("\n" + "="*80)
print("✅ ESPERIMENTO 031 COMPLETATO")
print("="*80)
print(f"\n📊 Statistiche finali:")
print(f"   Mean: {ensemble_predictions.mean():.2f}")
print(f"   Std:  {ensemble_predictions.std():.2f}")
print(f"   Min:  {ensemble_predictions.min():.2f}")
print(f"   Max:  {ensemble_predictions.max():.2f}")
print(f"\n🎯 Atteso: 11.5-11.8 MAE su Kaggle")
print(f"   (vs exp_015 best: 12.01 MAE)")
print(f"\n✅ FEDERATED LEARNING COMPLIANT")
