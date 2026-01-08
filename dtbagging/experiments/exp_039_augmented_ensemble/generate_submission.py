"""
ESPERIMENTO 039 - Ensemble finale con modelli AUGMENTED

Genera submission con RF-augmented + LightGBM-augmented.
Pesi: 65% RF + 35% LGBM
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import mean_absolute_error

print("="*80)
print("EXP 039: Generazione submission con modelli AUGMENTED")
print("="*80)

# Carica modelli
print("\n📂 Caricamento modelli augmented...")

model_dir = Path(__file__).parent

with open(model_dir / "rf_augmented_model.pkl", 'rb') as f:
    rf_models = pickle.load(f)  # Lista di 9 modelli FL

if isinstance(rf_models, list):
    print(f"✅ RF models caricati: {len(rf_models)} modelli FL (bagging)")
else:
    print("⚠️  Caricato singolo modello RF (non FL), converto a lista")
    rf_models = [rf_models]

with open(model_dir / "lgbm_augmented_model.pkl", 'rb') as f:
    lgbm_models = pickle.load(f)  # Lista di 9 modelli FL o singolo
    
if isinstance(lgbm_models, list):
    print(f"✅ LightGBM models caricati: {len(lgbm_models)} modelli FL (bagging)")
else:
    print("⚠️  Caricato singolo modello LGBM (non FL), converto a lista")
    lgbm_models = [lgbm_models]

# Carica test data
BASE_DIR = model_dir.parent.parent
test_file = BASE_DIR / "x_test.csv"

print(f"\n📂 Caricamento test data: {test_file}")
df_test = pd.read_csv(test_file, sep=';')
print(f"✅ Test data: {len(df_test)} samples, {len(df_test.columns)} colonne")

# Preprocessing (come training) 
print("\n🔄 Preprocessing test data...")

# Estrai statistiche dalle time series (come exp_031)
timeseries_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']

for col in timeseries_cols:
    if col not in df_test.columns:
        continue
    
    prefix = col.replace('_time_series', '')
    print(f"   Estraendo statistiche da {col}...")
    
    def parse_series(x):
        if pd.isna(x):
            return []
        try:
            s = str(x).strip('[]')
            values = [float(v) for v in s.split(',') if v.strip() and float(v.strip()) >= 0]
            return values
        except:
            return []
    
    series = df_test[col].apply(parse_series)
    
    # Statistiche base (identico a exp_031)
    df_test[f'{prefix}_mean'] = series.apply(lambda x: np.mean(x) if len(x) > 0 else 0)
    df_test[f'{prefix}_min'] = series.apply(lambda x: np.min(x) if len(x) > 0 else 0)
    df_test[f'{prefix}_max'] = series.apply(lambda x: np.max(x) if len(x) > 0 else 0)
    df_test[f'{prefix}_std'] = series.apply(lambda x: np.std(x) if len(x) > 0 else 0)
    
    df_test = df_test.drop(columns=[col])

# Rimuovi colonne vuote
df_test = df_test.dropna(axis=1, how='all')

# Seleziona feature numeriche
feature_cols = [col for col in df_test.columns 
                if col not in ['day', 'label', 'Unnamed: 0', 'id'] 
                and not col.endswith('_time_series')
                and df_test[col].dtype in ['int64', 'float64']]

X_test = df_test[feature_cols]

# Handle NaN
if X_test.isna().any().any():
    print("⚠️  Trovati NaN, filling con median...")
    X_test = X_test.fillna(X_test.median())

print(f"✅ X_test preparato: {X_test.shape}")

# Get feature names from primo modello RF
rf_features = rf_models[0].feature_names_in_
print(f"   RF si aspetta {len(rf_features)} features")

# Align features
missing_features = set(rf_features) - set(X_test.columns)
if missing_features:
    print(f"⚠️  Mancano {len(missing_features)} features, aggiungo con 0")
    for feat in missing_features:
        X_test[feat] = 0

# Riordina colonne
X_test = X_test[rf_features]
print(f"✅ X_test allineato: {X_test.shape}")

# Predizioni FEDERATED (Bagging)
print("\n🎯 Generazione predizioni FEDERATED (bagging)...")

# RF: Media delle predizioni di 9 modelli
rf_preds = []
for i, rf_model in enumerate(rf_models):
    pred = rf_model.predict(X_test)
    rf_preds.append(pred)
    print(f"   RF client {i}: mean={pred.mean():.2f}, std={pred.std():.2f}")

rf_pred = np.mean(rf_preds, axis=0)  # Bagging: media
print(f"   RF ENSEMBLE: mean={rf_pred.mean():.2f}, std={rf_pred.std():.2f}")

# LightGBM: Media delle predizioni di 9 modelli
lgbm_preds = []
for i, lgbm_model in enumerate(lgbm_models):
    pred = lgbm_model.predict(X_test)
    lgbm_preds.append(pred)
    print(f"   LGBM client {i}: mean={pred.mean():.2f}, std={pred.std():.2f}")

lgbm_pred = np.mean(lgbm_preds, axis=0)  # Bagging: media
print(f"   LGBM ENSEMBLE: mean={lgbm_pred.mean():.2f}, std={lgbm_pred.std():.2f}")

print(f"   RF:      mean={rf_pred.mean():.2f}, std={rf_pred.std():.2f}")
print(f"   LightGBM: mean={lgbm_pred.mean():.2f}, std={lgbm_pred.std():.2f}")

# Ensemble (65% RF + 35% LGBM)
w_rf = 0.65
w_lgbm = 0.35

ensemble = w_rf * rf_pred + w_lgbm * lgbm_pred

print(f"\n📊 ENSEMBLE ({w_rf*100:.0f}% RF + {w_lgbm*100:.0f}% LGBM):")
print(f"   Mean: {ensemble.mean():.2f}")
print(f"   Std:  {ensemble.std():.2f}")
print(f"   Min:  {ensemble.min():.2f}")
print(f"   Max:  {ensemble.max():.2f}")

# Salva submission
submission = pd.DataFrame({
    'id': range(len(ensemble)),
    'label': ensemble
})

submission.to_csv('submission_exp039.csv', index=False)
print(f"\n✅ Submission salvata: submission_exp039.csv")

print("\n" + "="*80)
print("📊 CONFRONTO CON ESPERIMENTI PRECEDENTI")
print("="*80)
print("\n🔸 exp_031 (3-way): 11.92674 MAE")
print("🔸 exp_034 (4-way): 12.00102 MAE ❌")
print("🔸 exp_036 (5-way): 12.04329 MAE ❌")
print("🔸 exp_038 (2-way): 12.05000 MAE ❌")
print(f"\n🔸 exp_039 (AUGMENTED): Mean {ensemble.mean():.2f}, Std {ensemble.std():.2f}")
print("   Atteso: 11.40-11.70 MAE")
print("   6x più dati = meno overfitting!")

print("\n" + "="*80)
print("✅ SUBMISSION PRONTA")
print("="*80)
