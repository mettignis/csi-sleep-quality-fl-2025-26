"""Inference per Esperimento 014 - RF con Advanced Features"""
import numpy as np
import pandas as pd
import pickle
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


def load_and_preprocess_test(test_path: str):
    """Carica test set e aggiunge advanced features (identico a training)."""
    print(f"📂 Caricamento test set: {test_path}")
    
    data = pd.read_csv(test_path, sep=';')
    test_ids = data['id'].copy()
    
    print(f"   Campioni: {len(data)}")
    
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
    
    # Riempi NaN (identico a training)
    for col in data.columns:
        if data[col].dtype in ['float64', 'int64'] and data[col].isna().any():
            median_val = data[col].median()
            if pd.isna(median_val):
                data[col] = data[col].fillna(0)
            else:
                data[col] = data[col].fillna(median_val)
    
    print(f"   Features totali: {len(data.columns)}")
    
    return data, test_ids


def generate_submission(model_path: str, test_path: str, output_path: str):
    """Genera submission usando solo RF models con advanced features."""
    print("="*70)
    print("🌲 RANDOM FOREST ADVANCED FEATURES - ESPERIMENTO 014")
    print("="*70)
    print()
    
    # Carica RF models
    print("🌲 Caricamento RF models...")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    rf_models = model_data['models']
    selected_features = model_data['selected_features']
    
    print(f"   ✓ {len(rf_models)} modelli RF caricati")
    print(f"   ✓ {len(selected_features)} features selezionate")
    
    # Preprocessing test set
    print("\n🔬 Preprocessing test set (advanced features)...")
    test_data, test_ids = load_and_preprocess_test(test_path)
    
    # Applica feature selection
    print(f"\n🔍 Applicazione feature selection...")
    test_data_selected = test_data[selected_features]
    print(f"   ✓ Features selezionate: {test_data_selected.shape[1]}")
    
    # Predizioni (media di tutti i 90 modelli)
    print(f"\n🌲 Predizioni (media di {len(rf_models)} modelli RF)...")
    all_predictions = []
    
    for i, rf_model in enumerate(rf_models):
        pred = rf_model.predict(test_data_selected)
        all_predictions.append(pred)
        
        if (i + 1) % 10 == 0:
            print(f"   Modello {i+1}/{len(rf_models)}...", flush=True)
    
    final_predictions = np.mean(all_predictions, axis=0)
    
    print(f"\n📊 Statistiche predizioni:")
    print(f"   Mean:   {final_predictions.mean():.2f}")
    print(f"   Std:    {final_predictions.std():.2f}")
    print(f"   Min:    {final_predictions.min():.2f}")
    print(f"   Max:    {final_predictions.max():.2f}")
    print(f"   Median: {np.median(final_predictions):.2f}")
    
    # Crea submission
    submission = pd.DataFrame({
        'id': test_ids,
        'y': final_predictions
    })
    
    submission.to_csv(output_path, index=False)
    
    print(f"\n✅ Submission salvata: {output_path}")
    print(f"   {len(submission)} predizioni generate")
    print("="*70)
    
    return final_predictions


if __name__ == "__main__":
    exp_dir = Path(__file__).parent / "experiments" / "exp_014_rf_advanced_features"
    model_path = exp_dir / "rf_advanced_models_exp014.pkl"
    test_path = Path(__file__).parent / "x_test.csv"
    output_path = exp_dir / "submission_exp014.csv"
    
    generate_submission(
        model_path=str(model_path),
        test_path=str(test_path),
        output_path=str(output_path)
    )
