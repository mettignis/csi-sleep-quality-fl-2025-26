"""
exp_042: Weight optimization exp_039/exp_031 ensemble

Current: 0.5/0.5 → 11.79207 MAE (SUCCESS!)
Test conservative weights: 0.6/0.4, 0.4/0.6, 0.55/0.45, 0.45/0.55

Strategy: Find best local balance, submit only if clearly better
"""

import pandas as pd
import numpy as np

def main():
    print("="*70)
    print("EXP_042: WEIGHT OPTIMIZATION exp_039/exp_031")
    print("="*70)
    
    # Load base submissions
    df_039 = pd.read_csv("../exp_039_augmented_ensemble/submission_exp039.csv")
    df_031 = pd.read_csv("../exp_031_three_way_federated/results/submission_exp031.csv")
    
    print(f"\n📊 Base statistics:")
    print(f"  exp_039: Mean={df_039['label'].mean():.2f}, Std={df_039['label'].std():.2f}")
    print(f"  exp_031: Mean={df_031['label'].mean():.2f}, Std={df_031['label'].std():.2f}")
    
    # Current best
    print(f"\n🏆 Current best (exp_041):")
    print(f"  0.5/0.5 → 11.79207 MAE Kaggle")
    
    # Test weight combinations
    weights_to_test = [
        (0.6, 0.4, "More exp_039 (augmented)"),
        (0.55, 0.45, "Slightly more exp_039"),
        (0.45, 0.55, "Slightly more exp_031"),
        (0.4, 0.6, "More exp_031 (non-aug + XGB)")
    ]
    
    print(f"\n🔬 Testing {len(weights_to_test)} weight combinations:")
    print("="*70)
    
    results = []
    
    for w1, w2, desc in weights_to_test:
        # Create ensemble
        ensemble = w1 * df_039['label'] + w2 * df_031['label']
        
        # Statistics
        mean_pred = ensemble.mean()
        std_pred = ensemble.std()
        min_pred = ensemble.min()
        max_pred = ensemble.max()
        
        # Difference from 0.5/0.5
        current_ensemble = 0.5 * df_039['label'] + 0.5 * df_031['label']
        diff_from_current = np.abs(ensemble - current_ensemble).mean()
        max_diff = np.abs(ensemble - current_ensemble).max()
        
        results.append({
            'weights': f"{w1:.2f}/{w2:.2f}",
            'w1': w1,
            'w2': w2,
            'desc': desc,
            'mean': mean_pred,
            'std': std_pred,
            'min': min_pred,
            'max': max_pred,
            'diff_avg': diff_from_current,
            'diff_max': max_diff
        })
        
        print(f"\n{desc} ({w1:.2f}/{w2:.2f}):")
        print(f"  Mean: {mean_pred:.2f}, Std: {std_pred:.2f}")
        print(f"  Range: [{min_pred:.2f}, {max_pred:.2f}]")
        print(f"  Diff from 0.5/0.5: avg={diff_from_current:.3f}, max={max_diff:.2f}")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS & RECOMMENDATION:")
    print("="*70)
    
    # Find most different from current
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('diff_avg', ascending=False)
    
    print(f"\n📊 Ranked by diversification from current (0.5/0.5):")
    for idx, row in df_results.iterrows():
        print(f"  {row['weights']}: {row['desc']}")
        print(f"    → Avg diff: {row['diff_avg']:.3f}, Max diff: {row['diff_max']:.2f}")
    
    # Recommendation
    print(f"\n💡 RACCOMANDAZIONE:")
    
    # Most diverse option
    best_option = df_results.iloc[0]
    print(f"\n1️⃣ MIGLIORE OPZIONE: {best_option['weights']} - {best_option['desc']}")
    print(f"   → Diversificazione massima: {best_option['diff_avg']:.3f} avg diff")
    print(f"   → Mean: {best_option['mean']:.2f} (vs 75.24 current)")
    print(f"   → Se mean più vicino a 75.5: potrebbe migliorare")
    
    # Generate best submission
    print(f"\n📝 Generating submission with best weights: {best_option['w1']:.2f}/{best_option['w2']:.2f}")
    best_ensemble = best_option['w1'] * df_039['label'] + best_option['w2'] * df_031['label']
    
    submission = pd.DataFrame({
        'id': df_039['id'],
        'label': best_ensemble
    })
    
    output_path = f"submission_exp042_w{int(best_option['w1']*100)}{int(best_option['w2']*100)}.csv"
    submission.to_csv(output_path, index=False)
    print(f"✅ Saved: {output_path}")
    
    # Decision criteria
    print("\n" + "="*70)
    print("DECISION CRITERIA:")
    print("="*70)
    print(f"  Current best: 11.79207 MAE (exp_041 with 0.5/0.5)")
    print(f"\n  🎯 If this looks clearly better locally → SUBMIT")
    print(f"  😐 If marginal difference → SAVE for tomorrow")
    print(f"  ❌ If looks same/worse → KEEP exp_041 as best, save submission")

if __name__ == "__main__":
    main()
