"""Inference per Esperimento 013 - Stacking"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import xgboost as xgb


def parse_series(x):
    """Parse time-series string to list of floats"""
    if pd.isna(x):
        return []
    try:
        s = str(x).strip('[]')
        values = [float(v) for v in s.split(',') if v.strip() and float(v.strip()) >= 0]
        return values
    except:
        return []


def load_and_preprocess_test(test_path: str):
    """Carica e preprocessa test set."""
    print(f"📂 Caricamento test set: {test_path}")
    
    data = pd.read_csv(test_path, sep=';')
    print(f"   Campioni originali: {len(data)}")
    
    # Preprocessing time-series
    timeseries_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
    
    for col in timeseries_cols:
        if col not in data.columns:
            continue
        
        prefix = col.replace('_time_series', '')
        series = data[col].apply(parse_series)
        
        data[f'{prefix}_mean'] = series.apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_min'] = series.apply(lambda x: np.min(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_max'] = series.apply(lambda x: np.max(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_std'] = series.apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
        
        data.drop(columns=[col], inplace=True)
    
    # Aggiungi colonne mancanti
    if 'Unnamed: 0' not in data.columns:
        data.insert(0, 'Unnamed: 0', range(len(data)))
    
    if 'day' not in data.columns:
        data.insert(1, 'day', range(len(data)))
    
    # Rimuovi id se presente
    if 'id' in data.columns:
        data = data.drop(columns=['id'])
    
    print(f"   Feature finali: {len(data.columns)}")
    
    return data


def generate_stacking_submission(model_path: str, test_path: str, output_path: str):
    """Genera submission usando stacking model."""
    print("="*70)
    print("🎯 INFERENCE - ESPERIMENTO 013 (Stacking)")
    print("="*70)
    
    # Carica stacking model
    print(f"\n📦 Caricamento stacking model: {model_path}")
    with open(model_path, 'rb') as f:
        stacking_model = pickle.load(f)
    
    rf_models = stacking_model['rf_models']
    xgb_models = stacking_model['xgb_models']
    meta_learner = stacking_model['meta_learner']
    
    print(f"   ✓ {len(rf_models)} RF models caricati")
    print(f"   ✓ {len(xgb_models)} XGBoost models caricati")
    print(f"   ✓ Meta-learner caricato")
    
    print(f"\n📊 Pesi meta-learner:")
    print(f"   RF weight: {meta_learner.coef_[0]:.4f}")
    print(f"   XGB weight: {meta_learner.coef_[1]:.4f}")
    
    # Carica test set
    test_data = load_and_preprocess_test(test_path)
    X_test = test_data.values
    
    print(f"\n🤖 Generazione predizioni base models...")
    
    # Predizioni da tutti i RF models (media)
    rf_predictions = []
    for i, rf_model in enumerate(rf_models):
        pred = rf_model.predict(X_test)
        rf_predictions.append(pred)
    
    rf_pred_avg = np.mean(rf_predictions, axis=0)
    print(f"   RF predictions: mean={rf_pred_avg.mean():.2f}, std={rf_pred_avg.std():.2f}")
    
    # Predizioni da tutti i XGBoost models (media)
    xgb_predictions = []
    for i, xgb_model in enumerate(xgb_models):
        dtest = xgb.DMatrix(X_test)
        pred = xgb_model.predict(dtest)
        xgb_predictions.append(pred)
    
    xgb_pred_avg = np.mean(xgb_predictions, axis=0)
    print(f"   XGB predictions: mean={xgb_pred_avg.mean():.2f}, std={xgb_pred_avg.std():.2f}")
    
    # Stack predictions e applica meta-learner
    print(f"\n🧠 Applicazione meta-learner...")
    meta_features = np.column_stack([rf_pred_avg, xgb_pred_avg])
    final_predictions = meta_learner.predict(meta_features)
    
    # Statistiche finali
    print(f"\n📊 Statistiche predizioni finali:")
    print(f"   Mean: {final_predictions.mean():.2f}")
    print(f"   Std:  {final_predictions.std():.2f}")
    print(f"   Min:  {final_predictions.min():.2f}")
    print(f"   Max:  {final_predictions.max():.2f}")
    print(f"   Median: {np.median(final_predictions):.2f}")
    
    # Confronto
    print(f"\n📊 Confronto:")
    print(f"   RF avg:     {rf_pred_avg.mean():.2f} | std: {rf_pred_avg.std():.2f}")
    print(f"   XGB avg:    {xgb_pred_avg.mean():.2f} | std: {xgb_pred_avg.std():.2f}")
    print(f"   Stacking:   {final_predictions.mean():.2f} | std: {final_predictions.std():.2f}")
    print(f"\n   vs exp_012 (Ensemble): mean 74.60, std 3.58")
    print(f"   vs exp_008 (RF best):  mean 74.42, std 2.24")
    
    # Crea submission
    submission = pd.DataFrame({
        'id': range(len(final_predictions)),
        'label': final_predictions
    })
    
    submission.to_csv(output_path, index=False)
    
    print(f"\n✅ Submission salvata: {output_path}")
    print(f"\n🎯 Aspettativa: < 12.16010 (battere exp_012)")
    print("="*70)


if __name__ == "__main__":
    exp_dir = Path(__file__).parent / "experiments" / "exp_013_stacking"
    model_path = exp_dir / "stacking_model_exp013.pkl"
    test_path = Path(__file__).parent / "x_test.csv"
    output_path = exp_dir / "submission_exp013.csv"
    
    generate_stacking_submission(str(model_path), str(test_path), str(output_path))
