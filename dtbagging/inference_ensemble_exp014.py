"""Inference per Esperimento 014 - Ensemble RF Advanced + XGB Base"""
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks


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


def extract_advanced_timeseries_features(series_data, prefix):
    """Estrae feature avanzate da time-series (identico a training)."""
    if len(series_data) == 0 or all(pd.isna(series_data)):
        return {}
    
    clean_data = [x for x in series_data if not pd.isna(x)]
    if len(clean_data) == 0:
        return {}
    
    features = {}
    arr = np.array(clean_data)
    
    features[f'{prefix}_p25'] = np.percentile(arr, 25)
    features[f'{prefix}_p75'] = np.percentile(arr, 75)
    features[f'{prefix}_p90'] = np.percentile(arr, 90)
    features[f'{prefix}_iqr'] = features[f'{prefix}_p75'] - features[f'{prefix}_p25']
    
    features[f'{prefix}_skewness'] = stats.skew(arr)
    features[f'{prefix}_kurtosis'] = stats.kurtosis(arr)
    
    features[f'{prefix}_range'] = np.max(arr) - np.min(arr)
    features[f'{prefix}_cv'] = np.std(arr) / (np.mean(arr) + 1e-8)
    median_val = np.median(arr)
    features[f'{prefix}_mad'] = np.median(np.abs(arr - median_val))
    
    if len(arr) > 2:
        x = np.arange(len(arr))
        try:
            slope, _, r_value, _, _ = stats.linregress(x, arr)
            features[f'{prefix}_trend_slope'] = slope
            features[f'{prefix}_trend_r2'] = r_value ** 2
        except:
            features[f'{prefix}_trend_slope'] = 0
            features[f'{prefix}_trend_r2'] = 0
    else:
        features[f'{prefix}_trend_slope'] = 0
        features[f'{prefix}_trend_r2'] = 0
    
    if len(arr) > 1:
        diffs = np.diff(arr)
        features[f'{prefix}_mean_diff'] = np.mean(diffs)
        features[f'{prefix}_std_diff'] = np.std(diffs)
        features[f'{prefix}_max_abs_diff'] = np.max(np.abs(diffs))
    else:
        features[f'{prefix}_mean_diff'] = 0
        features[f'{prefix}_std_diff'] = 0
        features[f'{prefix}_max_abs_diff'] = 0
    
    if len(arr) > 3:
        try:
            peaks, _ = find_peaks(arr)
            features[f'{prefix}_num_peaks'] = len(peaks)
            features[f'{prefix}_peak_rate'] = len(peaks) / len(arr)
            if len(peaks) > 0:
                features[f'{prefix}_mean_peak_value'] = np.mean(arr[peaks])
            else:
                features[f'{prefix}_mean_peak_value'] = np.max(arr)
        except:
            features[f'{prefix}_num_peaks'] = 0
            features[f'{prefix}_peak_rate'] = 0
            features[f'{prefix}_mean_peak_value'] = np.max(arr)
    else:
        features[f'{prefix}_num_peaks'] = 0
        features[f'{prefix}_peak_rate'] = 0
        features[f'{prefix}_mean_peak_value'] = np.max(arr)
    
    if len(arr) > 1:
        try:
            hist, _ = np.histogram(arr, bins=min(10, len(arr)))
            hist = hist[hist > 0]
            if len(hist) > 0:
                probs = hist / len(arr)
                features[f'{prefix}_entropy'] = -np.sum(probs * np.log(probs + 1e-10))
            else:
                features[f'{prefix}_entropy'] = 0
        except:
            features[f'{prefix}_entropy'] = 0
    else:
        features[f'{prefix}_entropy'] = 0
    
    if len(arr) > 2:
        try:
            corr = np.corrcoef(arr[:-1], arr[1:])[0, 1]
            features[f'{prefix}_autocorr'] = corr if not np.isnan(corr) else 0
        except:
            features[f'{prefix}_autocorr'] = 0
    else:
        features[f'{prefix}_autocorr'] = 0
    
    features[f'{prefix}_energy'] = np.sum(arr ** 2) / len(arr)
    
    if len(arr) > 1:
        mean_centered = arr - np.mean(arr)
        zero_crossings = np.sum(np.diff(np.sign(mean_centered)) != 0)
        features[f'{prefix}_zcr'] = zero_crossings / len(arr)
    else:
        features[f'{prefix}_zcr'] = 0
    
    return features


def load_and_preprocess_test_with_advanced_features(test_path: str):
    """Carica test set e aggiunge TUTTE le feature (base + advanced)."""
    print(f"📂 Caricamento test set: {test_path}")
    
    data = pd.read_csv(test_path, sep=';')
    test_ids = data['id'].copy()
    
    print(f"   Campioni: {len(data)}")
    print(f"   Colonne originali: {len(data.columns)}")
    
    # Preprocessing time-series
    timeseries_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
    
    for col in timeseries_cols:
        if col not in data.columns:
            continue
        
        prefix = col.replace('_time_series', '')
        series = data[col].apply(parse_series)
        
        # Base statistics
        data[f'{prefix}_mean'] = series.apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_min'] = series.apply(lambda x: np.min(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_max'] = series.apply(lambda x: np.max(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_std'] = series.apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
        
        # Advanced features
        advanced_features = series.apply(lambda x: extract_advanced_timeseries_features(x, prefix))
        advanced_df = pd.DataFrame(advanced_features.tolist())
        
        for adv_col in advanced_df.columns:
            data[adv_col] = advanced_df[adv_col]
        
        data.drop(columns=[col], inplace=True)
    
    # Rimuovi id
    data.drop(columns=['id'], inplace=True)
    
    # Aggiungi Unnamed: 0 e day
    data.insert(0, 'Unnamed: 0', range(len(data)))
    data.insert(1, 'day', range(len(data)))
    
    # Riempi NaN con mediana (sicuro per test set)
    for col in data.columns:
        if data[col].dtype in ['float64', 'int64'] and data[col].isna().any():
            data[col].fillna(data[col].median(), inplace=True)
    
    print(f"   Features totali dopo preprocessing: {len(data.columns)}")
    
    return data, test_ids


def load_and_preprocess_test_base(test_path: str):
    """Carica test set con SOLO feature base (per XGBoost)."""
    print(f"📂 Caricamento test set (base features): {test_path}")
    
    data = pd.read_csv(test_path, sep=';')
    test_ids = data['id'].copy()
    
    # Preprocessing time-series BASE
    timeseries_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
    
    for col in timeseries_cols:
        if col not in data.columns:
            continue
        
        prefix = col.replace('_time_series', '')
        series = data[col].apply(parse_series)
        
        # SOLO base statistics
        data[f'{prefix}_mean'] = series.apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_min'] = series.apply(lambda x: np.min(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_max'] = series.apply(lambda x: np.max(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_std'] = series.apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
        
        data.drop(columns=[col], inplace=True)
    
    data.drop(columns=['id'], inplace=True)
    data.insert(0, 'Unnamed: 0', range(len(data)))
    
    # NON aggiungere 'day' - XGBoost non l'ha vista in training
    # Rimuovi 'act_activeTime' se esiste - XGBoost non l'ha vista
    if 'act_activeTime' in data.columns:
        data.drop(columns=['act_activeTime'], inplace=True)
    
    for col in data.columns:
        if data[col].dtype in ['float64', 'int64'] and data[col].isna().any():
            data[col].fillna(data[col].median(), inplace=True)
    
    print(f"   Features base: {len(data.columns)}")
    
    return data, test_ids


def generate_ensemble_submission(rf_model_path: str, xgb_model_path: str, test_path: str, output_path: str, rf_weight=0.7):
    """
    Genera submission con ensemble RF (advanced features) + XGB (base features).
    """
    print("="*70)
    print("🎯 ENSEMBLE RF ADVANCED + XGB BASE - ESPERIMENTO 015")
    print("="*70)
    print(f"Pesi: RF {rf_weight*100:.0f}% + XGB {(1-rf_weight)*100:.0f}%")
    print()
    
    # 1. Carica RF models con advanced features
    print("🌲 Caricamento RF models (advanced features)...")
    with open(rf_model_path, 'rb') as f:
        rf_data = pickle.load(f)
    
    rf_models = rf_data['models']
    feature_selector = rf_data['feature_selector']
    selected_features = rf_data['selected_features']
    
    print(f"   ✓ {len(rf_models)} modelli RF caricati")
    print(f"   ✓ {len(selected_features)} features selezionate")
    
    # 2. Carica XGBoost model base
    print("\n⚡ Caricamento XGBoost model (base features)...")
    xgb_model = xgb.Booster()
    xgb_model.load_model(xgb_model_path)
    print(f"   ✓ Modello XGBoost caricato")
    
    # 3. Preprocessing test set per RF (con advanced features)
    print("\n🔬 Preprocessing test set per RF (advanced features)...")
    test_data_rf, test_ids = load_and_preprocess_test_with_advanced_features(test_path)
    
    # Applica feature selection
    test_data_rf_selected = test_data_rf[selected_features]
    print(f"   ✓ Features selezionate: {test_data_rf_selected.shape[1]}")
    
    # 4. Preprocessing test set per XGB (solo base)
    print("\n📊 Preprocessing test set per XGB (base features)...")
    test_data_xgb, _ = load_and_preprocess_test_base(test_path)
    
    # Riordina colonne per matchare training XGBoost
    expected_feature_names = xgb_model.feature_names
    if expected_feature_names:
        print(f"   🔄 Riordinamento colonne per XGBoost...")
        # Assicurati che tutte le feature richieste siano presenti
        missing_cols = set(expected_feature_names) - set(test_data_xgb.columns)
        if missing_cols:
            print(f"   ⚠️  Colonne mancanti: {missing_cols}")
            for col in missing_cols:
                test_data_xgb[col] = 0
        
        # Riordina
        test_data_xgb = test_data_xgb[expected_feature_names]
    
    dtest = xgb.DMatrix(test_data_xgb)
    print(f"   ✓ Features base: {test_data_xgb.shape[1]}")
    
    # 5. Predizioni RF (media di tutti i modelli)
    print(f"\n🌲 Predizioni Random Forest...")
    rf_predictions = []
    
    for i, rf_model in enumerate(rf_models):
        pred = rf_model.predict(test_data_rf_selected)
        rf_predictions.append(pred)
    
    rf_pred_avg = np.mean(rf_predictions, axis=0)
    
    print(f"   ✓ Media di {len(rf_models)} modelli RF")
    print(f"   📊 RF stats: mean={rf_pred_avg.mean():.2f}, std={rf_pred_avg.std():.2f}")
    print(f"              min={rf_pred_avg.min():.2f}, max={rf_pred_avg.max():.2f}")
    
    # 6. Predizioni XGBoost
    print(f"\n⚡ Predizioni XGBoost...")
    xgb_predictions = xgb_model.predict(dtest)
    
    print(f"   📊 XGB stats: mean={xgb_predictions.mean():.2f}, std={xgb_predictions.std():.2f}")
    print(f"              min={xgb_predictions.min():.2f}, max={xgb_predictions.max():.2f}")
    
    # 7. Ensemble weighted average
    print(f"\n🎯 Ensemble {rf_weight*100:.0f}% RF + {(1-rf_weight)*100:.0f}% XGB...")
    ensemble_predictions = rf_weight * rf_pred_avg + (1 - rf_weight) * xgb_predictions
    
    print(f"   📊 Ensemble stats:")
    print(f"      Mean:   {ensemble_predictions.mean():.2f}")
    print(f"      Std:    {ensemble_predictions.std():.2f}")
    print(f"      Min:    {ensemble_predictions.min():.2f}")
    print(f"      Max:    {ensemble_predictions.max():.2f}")
    print(f"      Median: {np.median(ensemble_predictions):.2f}")
    
    # 8. Crea submission
    submission = pd.DataFrame({
        'id': test_ids,
        'y': ensemble_predictions
    })
    
    submission.to_csv(output_path, index=False)
    
    print(f"\n✅ Submission salvata: {output_path}")
    print(f"   {len(submission)} predizioni generate")
    print("="*70)
    
    return ensemble_predictions


if __name__ == "__main__":
    rf_model_path = Path(__file__).parent / "experiments" / "exp_014_rf_advanced_features" / "rf_advanced_models_exp014.pkl"
    xgb_model_path = Path(__file__).parent / "experiments" / "exp_001_xgboost_baseline" / "model.json"
    test_path = Path(__file__).parent / "x_test.csv"
    output_path = Path(__file__).parent / "experiments" / "exp_015_ensemble_advanced" / "submission_exp015.csv"
    
    generate_ensemble_submission(
        rf_model_path=str(rf_model_path),
        xgb_model_path=str(xgb_model_path),
        test_path=str(test_path),
        output_path=str(output_path),
        rf_weight=0.7
    )
