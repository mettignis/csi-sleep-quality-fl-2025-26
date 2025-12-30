#!/usr/bin/env python3
"""
ESPERIMENTO 016 - Three-Way Ensemble Inference

Combina 3 modelli con pesi ottimizzati:
- 60% Random Forest Advanced (exp_014): 25 advanced features, 90 modelli
- 25% XGBoost Base (exp_001): 40 base features
- 15% LightGBM (exp_016): 40 base features

Expected MAE: 11.85-11.95 (miglioramento su exp_015: 12.01)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import pickle
from pathlib import Path

# Importa funzioni preprocessing da inference_exp014 (per RF advanced)
from inference_exp014 import load_and_preprocess_test as load_test_advanced

# Importa funzioni preprocessing da inference_ensemble_exp014 (per XGB base)
from inference_ensemble_exp014 import load_and_preprocess_test_base

print("="*80)
print("ESPERIMENTO 016: Three-Way Ensemble Inference")
print("="*80)

def generate_three_way_ensemble_submission():
    """Genera submission combinando RF Advanced (60%) + XGBoost (25%) + LightGBM (15%)."""
    print("\n" + "="*80)
    print("GENERAZIONE THREE-WAY ENSEMBLE SUBMISSION")
    print("="*80)
    
    # === 1. LOAD MODELLI ===
    print("\n📦 Caricamento modelli...")
    
    # RF Advanced (exp_014)
    rf_model_path = Path("experiments/exp_014_rf_advanced_features/rf_advanced_models_exp014.pkl")
    with open(rf_model_path, 'rb') as f:
        rf_data = pickle.load(f)
    
    rf_models = rf_data['models']
    rf_selector = rf_data['feature_selector']
    rf_selected_features = rf_data['selected_features']
    print(f"   ✅ RF Advanced: {len(rf_models)} modelli, {len(rf_selected_features)} features")
    
    # XGBoost (exp_001)
    xgb_model = xgb.Booster()
    xgb_model.load_model("experiments/exp_001_xgboost_baseline/model.json")
    print(f"   ✅ XGBoost loaded")
    
    # LightGBM (exp_016)
    lgbm_model_path = Path("experiments/exp_016_three_way_ensemble/lgbm_model_exp016.txt")
    lgbm_model = lgb.Booster(model_file=str(lgbm_model_path))
    
    lgbm_feature_names_path = Path("experiments/exp_016_three_way_ensemble/lgbm_feature_names_exp016.pkl")
    with open(lgbm_feature_names_path, 'rb') as f:
        lgbm_feature_names = pickle.load(f)
    print(f"   ✅ LightGBM loaded: {len(lgbm_feature_names)} features")
    
    # === 2. PREPROCESS TEST SET ===
    
    # Per RF Advanced (usa funzione da inference_exp014)
    test_data_rf, test_ids = load_test_advanced("x_test.csv")
    test_data_rf_selected = test_data_rf[rf_selected_features]
    
    # Per XGBoost & LightGBM (usa funzione da inference_ensemble_exp014)
    test_data_base, _ = load_and_preprocess_test_base("x_test.csv")
    
    # === 3. PREDIZIONI RF ADVANCED ===
    print("\n🔮 Predizioni RF Advanced (60%)...")
    rf_predictions_all = []
    for model in rf_models:
        pred = model.predict(test_data_rf_selected)
        rf_predictions_all.append(pred)
    
    rf_predictions = np.mean(rf_predictions_all, axis=0)
    print(f"   ✅ Mean: {np.mean(rf_predictions):.2f}, Std: {np.std(rf_predictions):.2f}")
    
    # === 4. PREDIZIONI XGBOOST ===
    print("\n🔮 Predizioni XGBoost (25%)...")
    
    # Prepara XGBoost data (riordina colonne)
    test_data_xgb = test_data_base.copy()
    expected_feature_names = xgb_model.feature_names
    if expected_feature_names:
        missing_cols = set(expected_feature_names) - set(test_data_xgb.columns)
        if missing_cols:
            for col in missing_cols:
                test_data_xgb[col] = 0
        test_data_xgb = test_data_xgb[expected_feature_names]
    
    dtest = xgb.DMatrix(test_data_xgb)
    xgb_predictions = xgb_model.predict(dtest)
    print(f"   ✅ Mean: {np.mean(xgb_predictions):.2f}, Std: {np.std(xgb_predictions):.2f}")
    
    # === 5. PREDIZIONI LIGHTGBM ===
    print("\n🔮 Predizioni LightGBM (15%)...")
    
    # Prepara LightGBM data (rinomina colonne come training)
    test_data_lgbm = test_data_base.copy()
    test_data_lgbm.columns = [col.replace(':', '_').replace('[', '_').replace(']', '_').replace('"', '').replace(',', '_') 
                              for col in test_data_lgbm.columns]
    
    # Riordina colonne per matchare training
    test_data_lgbm = test_data_lgbm[lgbm_feature_names]
    
    lgbm_predictions = lgbm_model.predict(test_data_lgbm)
    print(f"   ✅ Mean: {np.mean(lgbm_predictions):.2f}, Std: {np.std(lgbm_predictions):.2f}")
    
    # === 6. ENSEMBLE WEIGHTED ===
    print("\n⚖️  Combinazione three-way ensemble...")
    print(f"   Pesi: 60% RF + 25% XGB + 15% LightGBM")
    
    final_predictions = (0.60 * rf_predictions + 
                        0.25 * xgb_predictions + 
                        0.15 * lgbm_predictions)
    
    print(f"\n📊 Statistiche finali:")
    print(f"   Mean: {np.mean(final_predictions):.2f}")
    print(f"   Std: {np.std(final_predictions):.2f}")
    print(f"   Min: {np.min(final_predictions):.2f}")
    print(f"   Max: {np.max(final_predictions):.2f}")
    
    # === 7. SALVA SUBMISSION ===
    submission = pd.DataFrame({
        'id': range(len(final_predictions)),
        'label': final_predictions
    })
    
    output_path = "experiments/exp_016_three_way_ensemble/submission_exp016.csv"
    submission.to_csv(output_path, index=False)
    
    print(f"\n💾 Submission salvata: {output_path}")
    print(f"   Samples: {len(submission)}")
    
    print("\n" + "="*80)
    print("✅ THREE-WAY ENSEMBLE COMPLETATO")
    print("="*80)

if __name__ == "__main__":
    generate_three_way_ensemble_submission()
