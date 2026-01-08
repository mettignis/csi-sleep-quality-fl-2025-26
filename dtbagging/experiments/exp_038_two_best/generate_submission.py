"""
ESPERIMENTO 038 - TWO BEST ENSEMBLE

Strategy: SOLO i 2 modelli con migliore validation MAE
- RF-advanced (exp_014): Val MAE ~12.20
- LightGBM (exp_031): Val MAE ~10.88

Elimina XGBoost (Val MAE 13.30) che introduce rumore.

Pesi: 65% RF-adv + 35% LGBM (più peso al modello più stabile)
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("ESPERIMENTO 038: TWO BEST ENSEMBLE")
print("="*80)

BASE_DIR = Path(__file__).parent.parent

# RF-advanced (exp_014) - Il migliore per stabilità
rf_adv = pd.read_csv(BASE_DIR / "exp_014_rf_advanced_features" / "submission_exp014.csv")
if 'label' in rf_adv.columns:
    rf_adv = rf_adv.rename(columns={'label': 'y'})
print(f"\n✅ RF-advanced: {len(rf_adv)} samples, mean={rf_adv['y'].mean():.2f}, std={rf_adv['y'].std():.2f}")

# LightGBM (exp_031) - Il migliore per MAE
lgbm = pd.read_csv(BASE_DIR / "exp_031_three_way_federated" / "submission_exp031.csv")
if 'label' in lgbm.columns:
    lgbm = lgbm.rename(columns={'label': 'y'})
print(f"✅ LightGBM:    {len(lgbm)} samples, mean={lgbm['y'].mean():.2f}, std={lgbm['y'].std():.2f}")

# Pesi: più peso al modello più stabile (RF)
w_rf = 0.65
w_lgbm = 0.35

print("\n" + "="*80)
print("PESI ENSEMBLE")
print("="*80)
print(f"RF-advanced: {w_rf:.2f} ({w_rf*100:.0f}%)")
print(f"LightGBM:    {w_lgbm:.2f} ({w_lgbm*100:.0f}%)")

# Crea ensemble
ensemble = w_rf * rf_adv['y'] + w_lgbm * lgbm['y']

print("\n" + "="*80)
print("RISULTATO ENSEMBLE")
print("="*80)
print(f"Mean: {ensemble.mean():.2f}")
print(f"Std:  {ensemble.std():.2f}")
print(f"Min:  {ensemble.min():.2f}")
print(f"Max:  {ensemble.max():.2f}")

# Salva submission
submission = pd.DataFrame({
    'id': rf_adv['id'],
    'label': ensemble
})

submission.to_csv('submission_exp038.csv', index=False)
print(f"\n✅ Submission salvata: submission_exp038.csv")
print(f"   Samples: {len(submission)}")

print("\n" + "="*80)
print("📊 CONFRONTO")
print("="*80)
print("\n🔸 exp_031 (3-way): 11.92674 MAE")
print("🔸 exp_034 (4-way): 12.00102 MAE ❌")
print("🔸 exp_036 (5-way): 12.04329 MAE ❌")
print(f"\n🔸 exp_038 (2-best): Mean {ensemble.mean():.2f}, Std {ensemble.std():.2f}")
print("   Atteso: 11.70-11.90 MAE")
print("\n💡 Meno modelli = Meno rumore")
print("   Solo i 2 migliori, niente XGBoost debole")

print("\n" + "="*80)
print("✅ SUBMISSION PRONTA")
print("="*80)
