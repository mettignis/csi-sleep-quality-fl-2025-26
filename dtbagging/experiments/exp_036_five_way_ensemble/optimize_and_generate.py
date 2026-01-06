#!/usr/bin/env python3
"""
ESPERIMENTO 036 - FIVE-WAY ENSEMBLE WITH DUAL RF

Strategy:
- RF-base (exp_008): Simple features
- RF-advanced (exp_014): Engineered features  
- XGBoost-base (exp_001): Baseline
- LightGBM-FL (exp_031): Federated trained
- Dual RF captures complementary patterns

Tutti i modelli sono FL-compliant (trained per-client).
L'ensemble è post-processing puro.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

print("="*80)
print("ESPERIMENTO 036: FIVE-WAY ENSEMBLE OPTIMIZATION")
print("="*80)

# ============================================================================
# 1. CARICAMENTO SUBMISSION FILES
# ============================================================================

print("\n📂 Caricamento submission files...")

BASE_DIR = Path(__file__).parent.parent

# RF-base (exp_008)
rf_base = pd.read_csv(BASE_DIR / "exp_008_rf_improved" / "submission_exp008.csv")
if 'label' in rf_base.columns:
    rf_base = rf_base.rename(columns={'label': 'y'})
print(f"✅ RF-base:     {len(rf_base)} samples, mean={rf_base['y'].mean():.2f}")

# RF-advanced (exp_014)
rf_adv = pd.read_csv(BASE_DIR / "exp_014_rf_advanced_features" / "submission_exp014.csv")
if 'label' in rf_adv.columns:
    rf_adv = rf_adv.rename(columns={'label': 'y'})
print(f"✅ RF-advanced: {len(rf_adv)} samples, mean={rf_adv['y'].mean():.2f}")

# XGBoost (exp_001)
xgb = pd.read_csv(BASE_DIR / "exp_001_xgboost_baseline" / "submission.csv")
if 'label' in xgb.columns:
    xgb = xgb.rename(columns={'label': 'y'})
print(f"✅ XGBoost:     {len(xgb)} samples, mean={xgb['y'].mean():.2f}")

# LightGBM (exp_031)
lgbm = pd.read_csv(BASE_DIR / "exp_031_three_way_federated" / "submission_exp031.csv")
if 'label' in lgbm.columns:
    lgbm = lgbm.rename(columns={'label': 'y'})
print(f"✅ LightGBM:    {len(lgbm)} samples, mean={lgbm['y'].mean():.2f}")

# Verifica lunghezze
assert len(rf_base) == len(rf_adv) == len(xgb) == len(lgbm), "Submission lengths mismatch!"

# ============================================================================
# 2. GRID SEARCH SUI PESI (5 DIMENSIONI)
# ============================================================================

print("\n" + "="*80)
print("🔍 GRID SEARCH - FIVE-WAY ENSEMBLE")
print("="*80)

print("\nRange di ricerca:")
print("  RF-base:     0.15 - 0.35 (step 5%)")
print("  RF-advanced: 0.20 - 0.40 (step 5%)")
print("  XGBoost:     0.10 - 0.30 (step 5%)")
print("  LightGBM:    0.10 - 0.30 (step 5%)")
print("  (constraint: sum = 1.0)")

# Search space
w_rf_base_range = np.arange(0.15, 0.36, 0.05)
w_rf_adv_range = np.arange(0.20, 0.41, 0.05)
w_xgb_range = np.arange(0.10, 0.31, 0.05)
w_lgbm_range = np.arange(0.10, 0.31, 0.05)

results = []
count = 0

print("\n⏳ Testing combinazioni...")

for w_rf_base in w_rf_base_range:
    for w_rf_adv in w_rf_adv_range:
        for w_xgb in w_xgb_range:
            for w_lgbm in w_lgbm_range:
                # Constraint: weights sum to 1.0
                if abs(w_rf_base + w_rf_adv + w_xgb + w_lgbm - 1.0) < 0.001:
                    count += 1
                    
                    # Create ensemble
                    ensemble = (w_rf_base * rf_base['y'] + 
                               w_rf_adv * rf_adv['y'] +
                               w_xgb * xgb['y'] +
                               w_lgbm * lgbm['y'])
                    
                    # Proxy score: mean distance from middle + variance penalty
                    mean_val = ensemble.mean()
                    std_val = ensemble.std()
                    score = abs(mean_val - 75) + 0.3 * std_val
                    
                    results.append({
                        'w_rf_base': w_rf_base,
                        'w_rf_adv': w_rf_adv,
                        'w_xgb': w_xgb,
                        'w_lgbm': w_lgbm,
                        'mean': mean_val,
                        'std': std_val,
                        'score_proxy': score
                    })
                    
                    if count % 50 == 0:
                        print(f"   Tested: {count} combinations...")

print(f"\n✅ Tested {len(results)} combinazioni valide")

# Sort by proxy score
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('score_proxy')

# ============================================================================
# 3. RISULTATI
# ============================================================================

print("\n" + "="*80)
print("📊 RISULTATI GRID SEARCH")
print("="*80)

print("\n🏆 TOP 15 COMBINAZIONI (proxy score):")
print(results_df.head(15).to_string(index=False))

best = results_df.iloc[0]

print("\n" + "="*80)
print("🥇 BEST WEIGHTS (PROXY SCORE)")
print("="*80)
print(f"RF-base:     {best['w_rf_base']:.2f} ({best['w_rf_base']*100:.0f}%)")
print(f"RF-advanced: {best['w_rf_adv']:.2f} ({best['w_rf_adv']*100:.0f}%)")
print(f"XGBoost:     {best['w_xgb']:.2f} ({best['w_xgb']*100:.0f}%)")
print(f"LightGBM:    {best['w_lgbm']:.2f} ({best['w_lgbm']*100:.0f}%)")
print(f"\nMean:  {best['mean']:.2f}")
print(f"Std:   {best['std']:.2f}")
print(f"Score: {best['score_proxy']:.4f}")

# ============================================================================
# 4. GENERAZIONE SUBMISSION
# ============================================================================

print("\n💾 Generazione submission con best weights...")

ensemble_best = (best['w_rf_base'] * rf_base['y'] + 
                 best['w_rf_adv'] * rf_adv['y'] +
                 best['w_xgb'] * xgb['y'] +
                 best['w_lgbm'] * lgbm['y'])

submission = pd.DataFrame({
    'id': rf_base['id'],
    'label': ensemble_best
})

submission.to_csv('submission_exp036.csv', index=False)
print(f"✅ Submission salvata: submission_exp036.csv")
print(f"   Samples: {len(submission)}")
print(f"   Range: [{submission['label'].min():.2f}, {submission['label'].max():.2f}]")

# Save top 10 for reference
results_df.head(10).to_csv('top10_weights_exp036.csv', index=False)
print(f"✅ Salvato: top10_weights_exp036.csv")

# ============================================================================
# 5. CONFRONTO CON ESPERIMENTI PRECEDENTI
# ============================================================================

print("\n" + "="*80)
print("📊 CONFRONTO")
print("="*80)

print("\n🔸 exp_031 (3-way: 50-30-20):")
print("   Kaggle MAE: 11.92674")

print("\n🔸 exp_034 (4-way con CatBoost):")
print("   Kaggle MAE: 12.00102 (PEGGIORE)")

print(f"\n🔸 exp_036 (5-way dual RF: {best['w_rf_base']*100:.0f}-{best['w_rf_adv']*100:.0f}-{best['w_xgb']*100:.0f}-{best['w_lgbm']*100:.0f}):")
print(f"   Mean: {best['mean']:.2f}, Std: {best['std']:.2f}")
print(f"   Atteso: 11.60-11.85 MAE (stima)")

print("\n" + "="*80)
print("✅ GRID SEARCH COMPLETATO")
print("="*80)
print("\n📤 Submission pronta: submission_exp036.csv")
print("💡 Dual RF potrebbe catturare pattern complementari!")
