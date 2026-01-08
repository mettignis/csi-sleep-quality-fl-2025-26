"""
exp_041: Ensemble exp_039 + exp_031
Strategy: Average predictions from exp_039 (augmented RF+LGBM) and exp_031 (non-augmented RF+XGB+LGBM)

Expected improvement: -0.02/-0.05 MAE from exp_039 (11.89067)
Diversification: Different data sources + different algorithm mix
"""

import pandas as pd
import numpy as np
import os

def main():
    print("="*70)
    print("EXP_041: ENSEMBLE exp_039 + exp_031")
    print("="*70)
    
    # Paths to submission files
    exp_039_path = "../exp_039_augmented_ensemble/submission_exp039.csv"
    exp_031_path = "../exp_031_three_way_federated/results/submission_exp031.csv"
    
    # Load submissions
    print("\n📥 Loading submissions...")
    df_039 = pd.read_csv(exp_039_path)
    df_031 = pd.read_csv(exp_031_path)
    
    print(f"  exp_039: {len(df_039)} predictions")
    print(f"  exp_031: {len(df_031)} predictions")
    
    # Verify same IDs
    assert len(df_039) == len(df_031), "Different number of predictions!"
    assert all(df_039['id'] == df_031['id']), "Mismatched IDs!"
    
    # Statistics before ensemble
    print("\n📊 Statistics BEFORE ensemble:")
    print(f"  exp_039: Mean={df_039['label'].mean():.2f}, Std={df_039['label'].std():.2f}")
    print(f"  exp_031: Mean={df_031['label'].mean():.2f}, Std={df_031['label'].std():.2f}")
    
    # Ensemble: simple average 0.5 + 0.5
    print("\n🔄 Creating ensemble (0.5 × exp_039 + 0.5 × exp_031)...")
    ensemble_predictions = 0.5 * df_039['label'] + 0.5 * df_031['label']
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'id': df_039['id'],
        'label': ensemble_predictions
    })
    
    # Statistics after ensemble
    print("\n📊 Statistics AFTER ensemble:")
    print(f"  exp_041: Mean={submission['label'].mean():.2f}, Std={submission['label'].std():.2f}")
    print(f"  Range: [{submission['label'].min():.2f}, {submission['label'].max():.2f}]")
    
    # Differences analysis
    diff_039 = np.abs(ensemble_predictions - df_039['label'])
    diff_031 = np.abs(ensemble_predictions - df_031['label'])
    print(f"\n  Avg diff from exp_039: {diff_039.mean():.2f} (max: {diff_039.max():.2f})")
    print(f"  Avg diff from exp_031: {diff_031.mean():.2f} (max: {diff_031.max():.2f})")
    
    # Save submission
    output_path = "submission_exp041.csv"
    submission.to_csv(output_path, index=False)
    print(f"\n✅ Submission saved: {output_path}")
    
    # Decision thresholds
    print("\n" + "="*70)
    print("DECISION CRITERIA:")
    print("="*70)
    print("  🎯 If Kaggle < 11.87 MAE: SUCCESS! → Use submission 2 for weight optimization")
    print("  😐 If Kaggle 11.87-11.91: MARGINAL → Save submission 2 for tomorrow")
    print("  ❌ If Kaggle > 11.91 MAE: FAILURE → Revert to exp_039, don't waste submission 2")
    print("\n  Reference: exp_039 = 11.89067 MAE, exp_031 = 11.92674 MAE")
    
if __name__ == "__main__":
    main()
