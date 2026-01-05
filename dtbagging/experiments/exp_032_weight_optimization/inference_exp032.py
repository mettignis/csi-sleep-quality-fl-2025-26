"""
Experiment 032: Inference con pesi ottimizzati
Genera submission usando i pesi ottimali trovati
"""

import numpy as np
import pandas as pd
import pickle
import os

print("=" * 80)
print("EXPERIMENT 032: INFERENCE WITH OPTIMIZED WEIGHTS")
print("=" * 80)

# ============================================================================
# 1. CARICAMENTO PESI OTTIMALI
# ============================================================================

print("\n📂 CARICAMENTO PESI OTTIMALI...")

with open('best_weights_exp032.pkl', 'rb') as f:
    best_weights = pickle.load(f)

w_rf, w_xgb, w_lgbm = best_weights

print(f"✅ Pesi ottimali caricati:")
print(f"   RF:      {w_rf:.2f} ({w_rf*100:.0f}%)")
print(f"   XGBoost: {w_xgb:.2f} ({w_xgb*100:.0f}%)")
print(f"   LightGBM: {w_lgbm:.2f} ({w_lgbm*100:.0f}%)")

# ============================================================================
# 2. CARICAMENTO SUBMISSION FILES
# ============================================================================

print("\n📂 CARICAMENTO SUBMISSION FILES...")

BASE_DIR = "/Users/annamettifogo/Desktop/polimi/csi/proj4/prova1/dtbagging"

# RF Advanced (exp_014)
rf_sub_path = os.path.join(BASE_DIR, "experiments/exp_014_rf_advanced/submission_exp014.csv")
rf_sub = pd.read_csv(rf_sub_path)
# Detect column name
rf_col = 'y' if 'y' in rf_sub.columns else 'label'
rf_preds = rf_sub[rf_col].values
print(f"✅ RF Advanced (exp_014): {len(rf_preds)} samples")
print(f"   Stats: mean={np.mean(rf_preds):.2f}, std={np.std(rf_preds):.2f}")

# XGBoost Base (exp_001)
xgb_sub_path = os.path.join(BASE_DIR, "experiments/exp_001_xgboost/submission_exp001.csv")
xgb_sub = pd.read_csv(xgb_sub_path)
xgb_col = 'y' if 'y' in xgb_sub.columns else 'label'
xgb_preds = xgb_sub[xgb_col].values
print(f"✅ XGBoost Base (exp_001): {len(xgb_preds)} samples")
print(f"   Stats: mean={np.mean(xgb_preds):.2f}, std={np.std(xgb_preds):.2f}")

# LightGBM FL (exp_031)
lgbm_sub_path = os.path.join(BASE_DIR, "experiments/exp_031_three_way_federated/submission_exp031.csv")
lgbm_sub = pd.read_csv(lgbm_sub_path)
lgbm_col = 'y' if 'y' in lgbm_sub.columns else 'label'
lgbm_preds = lgbm_sub[lgbm_col].values
print(f"✅ LightGBM FL (exp_031): {len(lgbm_preds)} samples")
print(f"   Stats: mean={np.mean(lgbm_preds):.2f}, std={np.std(lgbm_preds):.2f}")

# Verifica lunghezze
assert len(rf_preds) == len(xgb_preds) == len(lgbm_preds), "Lunghezze diverse!"

# ============================================================================
# 3. ENSEMBLE CON PESI OTTIMALI
# ============================================================================

print("\n🎯 ENSEMBLE CON PESI OTTIMALI...")

ensemble_preds = w_rf * rf_preds + w_xgb * xgb_preds + w_lgbm * lgbm_preds

print(f"\n📊 Statistiche Ensemble:")
print(f"   Mean:  {np.mean(ensemble_preds):.2f}")
print(f"   Std:   {np.std(ensemble_preds):.2f}")
print(f"   Min:   {np.min(ensemble_preds):.2f}")
print(f"   Max:   {np.max(ensemble_preds):.2f}")

# ============================================================================
# 4. SALVATAGGIO SUBMISSION
# ============================================================================

print("\n💾 SALVATAGGIO SUBMISSION...")

submission = pd.DataFrame({
    'id': range(len(ensemble_preds)),
    'y': ensemble_preds
})

submission.to_csv('submission_exp032.csv', index=False)

print(f"✅ Submission salvata: submission_exp032.csv")
print(f"   Samples: {len(submission)}")
print(f"   Range: [{np.min(ensemble_preds):.2f}, {np.max(ensemble_preds):.2f}]")

# ============================================================================
# 5. CONFRONTO CON EXP_031
# ============================================================================

print("\n" + "=" * 80)
print("📊 CONFRONTO CON ESPERIMENTI PRECEDENTI")
print("=" * 80)

print(f"\n🔸 exp_031 (pesi 50-30-20):")
print(f"   Formula: 0.50 × RF + 0.30 × XGB + 0.20 × LGBM")
print(f"   Kaggle MAE: 11.92674")

print(f"\n🔸 exp_032 (pesi ottimizzati {w_rf:.0%}-{w_xgb:.0%}-{w_lgbm:.0%}):")
print(f"   Formula: {w_rf:.2f} × RF + {w_xgb:.2f} × XGB + {w_lgbm:.2f} × LGBM")
print(f"   Atteso: 11.60-11.85 MAE (stima basata su validazione)")

print("\n" + "=" * 80)
print("✅ ESPERIMENTO 032 COMPLETATO")
print("=" * 80)
print("\n📤 Pronto per submission su Kaggle!")
print("📂 File: experiments/exp_032_weight_optimization/submission_exp032.csv")
