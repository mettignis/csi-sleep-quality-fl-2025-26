"""Esperimento 012: Ensemble RF + XGBoost"""
import pandas as pd
import numpy as np
from pathlib import Path

def create_ensemble_submission(rf_path: str, xgb_path: str, output_path: str, 
                               rf_weight: float = 0.7):
    """
    Crea ensemble di Random Forest e XGBoost.
    
    Args:
        rf_path: Path a submission exp_008 (RF improved)
        xgb_path: Path a submission exp_001 (XGBoost baseline)
        output_path: Path output
        rf_weight: Peso per RF (default 0.7, XGBoost avrà 1-rf_weight)
    """
    print("="*70)
    print("🤝 ENSEMBLE RF + XGBOOST - ESPERIMENTO 012")
    print("="*70)
    print(f"\nCombinazione modelli:")
    print(f"   • exp_008 (RF improved): score 12.45205 ✅")
    print(f"   • exp_001 (XGBoost baseline): score 12.74541")
    print(f"\nPesi ensemble:")
    print(f"   • Random Forest: {rf_weight*100:.0f}%")
    print(f"   • XGBoost: {(1-rf_weight)*100:.0f}%")
    print()
    
    # Carica submission
    rf_sub = pd.read_csv(rf_path)
    xgb_sub = pd.read_csv(xgb_path)
    
    print(f"📂 Caricamento submission:")
    print(f"   RF:  {len(rf_sub)} righe, mean={rf_sub['label'].mean():.2f}, std={rf_sub['label'].std():.2f}")
    print(f"   XGB: {len(xgb_sub)} righe, mean={xgb_sub['label'].mean():.2f}, std={xgb_sub['label'].std():.2f}")
    
    # Verifica che abbiano stesse dimensioni
    assert len(rf_sub) == len(xgb_sub), "Submission devono avere stessa lunghezza!"
    assert all(rf_sub['id'] == xgb_sub['id']), "ID devono corrispondere!"
    
    # Ensemble: weighted average
    ensemble_predictions = (rf_weight * rf_sub['label'].values + 
                           (1 - rf_weight) * xgb_sub['label'].values)
    
    # Statistiche
    print(f"\n📊 Statistiche ensemble:")
    print(f"   Mean: {ensemble_predictions.mean():.2f}")
    print(f"   Std:  {ensemble_predictions.std():.2f}")
    print(f"   Min:  {ensemble_predictions.min():.2f}")
    print(f"   Max:  {ensemble_predictions.max():.2f}")
    print(f"   Median: {np.median(ensemble_predictions):.2f}")
    
    # Confronto
    print(f"\n📊 Confronto:")
    print(f"   RF mean:      {rf_sub['label'].mean():.2f} | std: {rf_sub['label'].std():.2f}")
    print(f"   XGB mean:     {xgb_sub['label'].mean():.2f} | std: {xgb_sub['label'].std():.2f}")
    print(f"   Ensemble mean: {ensemble_predictions.mean():.2f} | std: {ensemble_predictions.std():.2f}")
    
    # Crea submission
    submission = pd.DataFrame({
        'id': rf_sub['id'],
        'label': ensemble_predictions
    })
    
    submission.to_csv(output_path, index=False)
    
    print(f"\n✅ Ensemble salvato: {output_path}")
    print(f"\n🎯 Aspettativa: score tra 12.45 e 12.74")
    print("="*70)


if __name__ == "__main__":
    # Percorsi
    base_dir = Path(__file__).parent
    
    rf_path = base_dir / "experiments" / "exp_008_rf_improved" / "submission_exp008.csv"
    xgb_path = base_dir / "experiments" / "exp_001_xgboost_baseline" / "submission.csv"
    
    exp_dir = base_dir / "experiments" / "exp_012_ensemble"
    exp_dir.mkdir(parents=True, exist_ok=True)
    output_path = exp_dir / "submission_exp012.csv"
    
    # Crea ensemble (70% RF, 30% XGBoost)
    create_ensemble_submission(
        str(rf_path), 
        str(xgb_path), 
        str(output_path),
        rf_weight=0.7
    )
