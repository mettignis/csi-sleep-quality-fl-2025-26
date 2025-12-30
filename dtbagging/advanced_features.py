"""Advanced Feature Engineering dalle Time-Series - Per exp_013"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks


def extract_advanced_timeseries_features(series_data):
    """
    Estrae feature avanzate da time-series.
    
    Args:
        series_data: array di valori
        
    Returns:
        dict di feature
    """
    if len(series_data) == 0:
        return {}
    
    features = {}
    
    # 1. Statistiche base (già abbiamo, ma aggiungiamo altre)
    features['mean'] = np.mean(series_data)
    features['std'] = np.std(series_data)
    features['min'] = np.min(series_data)
    features['max'] = np.max(series_data)
    features['median'] = np.median(series_data)
    
    # 2. Percentili
    features['percentile_25'] = np.percentile(series_data, 25)
    features['percentile_75'] = np.percentile(series_data, 75)
    features['percentile_90'] = np.percentile(series_data, 90)
    features['iqr'] = features['percentile_75'] - features['percentile_25']
    
    # 3. Shape statistics
    features['skewness'] = stats.skew(series_data)
    features['kurtosis'] = stats.kurtosis(series_data)
    
    # 4. Variabilità
    features['range'] = features['max'] - features['min']
    features['cv'] = features['std'] / (features['mean'] + 1e-8)  # coefficient of variation
    features['mad'] = np.median(np.abs(series_data - features['median']))  # median absolute deviation
    
    # 5. Trend features
    if len(series_data) > 2:
        # Linear trend
        x = np.arange(len(series_data))
        slope, intercept, r_value, _, _ = stats.linregress(x, series_data)
        features['trend_slope'] = slope
        features['trend_r2'] = r_value ** 2
    else:
        features['trend_slope'] = 0
        features['trend_r2'] = 0
    
    # 6. Change rate
    if len(series_data) > 1:
        diffs = np.diff(series_data)
        features['mean_diff'] = np.mean(diffs)
        features['std_diff'] = np.std(diffs)
        features['max_diff'] = np.max(np.abs(diffs))
        
        # Acceleration (second derivative)
        if len(diffs) > 1:
            acc = np.diff(diffs)
            features['mean_acceleration'] = np.mean(acc)
            features['std_acceleration'] = np.std(acc)
        else:
            features['mean_acceleration'] = 0
            features['std_acceleration'] = 0
    else:
        features['mean_diff'] = 0
        features['std_diff'] = 0
        features['max_diff'] = 0
        features['mean_acceleration'] = 0
        features['std_acceleration'] = 0
    
    # 7. Peak detection
    if len(series_data) > 3:
        peaks, _ = find_peaks(series_data)
        valleys, _ = find_peaks(-series_data)
        
        features['num_peaks'] = len(peaks)
        features['num_valleys'] = len(valleys)
        features['peak_rate'] = len(peaks) / len(series_data)
        
        if len(peaks) > 0:
            features['mean_peak_value'] = np.mean(series_data[peaks])
        else:
            features['mean_peak_value'] = features['max']
            
        if len(valleys) > 0:
            features['mean_valley_value'] = np.mean(series_data[valleys])
        else:
            features['mean_valley_value'] = features['min']
    else:
        features['num_peaks'] = 0
        features['num_valleys'] = 0
        features['peak_rate'] = 0
        features['mean_peak_value'] = features['max']
        features['mean_valley_value'] = features['min']
    
    # 8. Entropy (misura di irregolarità)
    if len(series_data) > 1:
        # Approximate entropy
        hist, _ = np.histogram(series_data, bins=10)
        hist = hist[hist > 0]  # remove zero bins
        if len(hist) > 0:
            features['entropy'] = -np.sum((hist / len(series_data)) * np.log(hist / len(series_data)))
        else:
            features['entropy'] = 0
    else:
        features['entropy'] = 0
    
    # 9. Autocorrelation (lag-1)
    if len(series_data) > 2:
        # Pearson correlation with lag-1
        features['autocorr_lag1'] = np.corrcoef(series_data[:-1], series_data[1:])[0, 1]
        if np.isnan(features['autocorr_lag1']):
            features['autocorr_lag1'] = 0
    else:
        features['autocorr_lag1'] = 0
    
    # 10. Energy
    features['energy'] = np.sum(series_data ** 2) / len(series_data)
    
    # 11. Zero-crossing rate
    if len(series_data) > 1:
        mean_centered = series_data - features['mean']
        zero_crossings = np.sum(np.diff(np.sign(mean_centered)) != 0)
        features['zero_crossing_rate'] = zero_crossings / len(series_data)
    else:
        features['zero_crossing_rate'] = 0
    
    return features


def add_advanced_features(df):
    """
    Aggiunge feature avanzate al dataframe.
    Applica extract_advanced_timeseries_features a hr, resp, stress time-series.
    
    NOTA: Questa funzione assume che le time-series siano già state processate
    e che esistano colonne mean/min/max/std. Qui NON processiamo le time-series raw,
    ma solo aggiungiamo nuove feature basate su quelle esistenti.
    """
    print("   🔬 Advanced feature engineering in corso...")
    
    # Per ora aggiungiamo solo feature aggiuntive basate su quelle esistenti
    # (percentili, ratios, etc.) poiché non abbiamo accesso alle time-series raw qui
    
    # Advanced ratios
    for prefix in ['hr', 'resp', 'stress']:
        if f'{prefix}_mean' in df.columns and f'{prefix}_std' in df.columns:
            # Coefficient of variation
            df[f'{prefix}_cv_advanced'] = df[f'{prefix}_std'] / (df[f'{prefix}_mean'] + 1e-8)
            
            # Normalized range
            if f'{prefix}_max' in df.columns and f'{prefix}_min' in df.columns:
                df[f'{prefix}_norm_range'] = (df[f'{prefix}_max'] - df[f'{prefix}_min']) / (df[f'{prefix}_mean'] + 1e-8)
                
                # Symmetry measure
                df[f'{prefix}_symmetry'] = (df[f'{prefix}_mean'] - df[f'{prefix}_min']) / (df[f'{prefix}_max'] - df[f'{prefix}_min'] + 1e-8)
    
    # Cross-feature interactions (più sofisticate)
    if 'hr_mean' in df.columns and 'stress_mean' in df.columns:
        df['hr_stress_ratio'] = df['hr_mean'] / (df['stress_mean'] + 1e-8)
        df['hr_stress_product'] = df['hr_mean'] * df['stress_mean']
    
    if 'resp_mean' in df.columns and 'stress_mean' in df.columns:
        df['resp_stress_ratio'] = df['resp_mean'] / (df['stress_mean'] + 1e-8)
    
    # Sleep efficiency features
    if 'sleep_sleepTimeSeconds' in df.columns and 'sleep_deepSleepSeconds' in df.columns:
        df['deep_sleep_efficiency'] = df['sleep_deepSleepSeconds'] / (df['sleep_sleepTimeSeconds'] + 1e-8)
        
        if 'sleep_remSleepSeconds' in df.columns:
            df['rem_deep_ratio'] = df['sleep_remSleepSeconds'] / (df['sleep_deepSleepSeconds'] + 1e-8)
    
    # Activity features
    if 'act_totalCalories' in df.columns and 'act_activeTime' in df.columns:
        df['calorie_per_minute'] = df['act_totalCalories'] / (df['act_activeTime'] / 60 + 1e-8)
    
    if 'act_distance' in df.columns and 'act_activeTime' in df.columns:
        df['speed'] = df['act_distance'] / (df['act_activeTime'] + 1e-8)
    
    # Log transforms (per distribuzioni skewed)
    for col in ['act_totalCalories', 'act_distance', 'sleep_sleepTimeSeconds']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])
    
    # Polynomial features (interazioni quadratiche chiave)
    if 'hr_mean' in df.columns:
        df['hr_mean_squared'] = df['hr_mean'] ** 2
    
    if 'stress_mean' in df.columns:
        df['stress_mean_squared'] = df['stress_mean'] ** 2
    
    print(f"   ✅ Advanced features aggiunte: {len(df.columns)} feature totali")
    
    return df
