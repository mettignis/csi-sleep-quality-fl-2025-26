"""Training Random Forest con Advanced Features - Esperimento 014"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error
import pickle
from scipy import stats
from scipy.signal import find_peaks

print("ci sto provando rega...")

def extract_advanced_timeseries_features(series_data, prefix):
    """
    Estrae feature avanzate da time-series.
    
    Args:
        series_data: lista di valori
        prefix: prefisso per nomi feature (hr, resp, stress)
        
    Returns:
        dict di feature
    """
    if len(series_data) == 0 or all(pd.isna(series_data)):
        return {}
    
    # Rimuovi NaN
    clean_data = [x for x in series_data if not pd.isna(x)]
    if len(clean_data) == 0:
        return {}
    
    features = {}
    arr = np.array(clean_data)
    
    # 1. Percentili
    features[f'{prefix}_p25'] = np.percentile(arr, 25)
    features[f'{prefix}_p75'] = np.percentile(arr, 75)
    features[f'{prefix}_p90'] = np.percentile(arr, 90)
    features[f'{prefix}_iqr'] = features[f'{prefix}_p75'] - features[f'{prefix}_p25']
    
    # 2. Shape statistics
    features[f'{prefix}_skewness'] = stats.skew(arr)
    features[f'{prefix}_kurtosis'] = stats.kurtosis(arr)
    
    # 3. Variabilità avanzata
    features[f'{prefix}_range'] = np.max(arr) - np.min(arr)
    features[f'{prefix}_cv'] = np.std(arr) / (np.mean(arr) + 1e-8)
    median_val = np.median(arr)
    features[f'{prefix}_mad'] = np.median(np.abs(arr - median_val))
    
    # 4. Trend
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
    
    # 5. Change rate
    if len(arr) > 1:
        diffs = np.diff(arr)
        features[f'{prefix}_mean_diff'] = np.mean(diffs)
        features[f'{prefix}_std_diff'] = np.std(diffs)
        features[f'{prefix}_max_abs_diff'] = np.max(np.abs(diffs))
    else:
        features[f'{prefix}_mean_diff'] = 0
        features[f'{prefix}_std_diff'] = 0
        features[f'{prefix}_max_abs_diff'] = 0
    
    # 6. Peak detection
    if len(arr) > 3:
        try:
            peaks, _ = find_peaks(arr)
            valleys, _ = find_peaks(-arr)
            
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
    
    # 7. Entropy
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
    
    # 8. Autocorrelation
    if len(arr) > 2:
        try:
            corr = np.corrcoef(arr[:-1], arr[1:])[0, 1]
            features[f'{prefix}_autocorr'] = corr if not np.isnan(corr) else 0
        except:
            features[f'{prefix}_autocorr'] = 0
    else:
        features[f'{prefix}_autocorr'] = 0
    
    # 9. Energy
    features[f'{prefix}_energy'] = np.sum(arr ** 2) / len(arr)
    
    # 10. Zero-crossing rate
    if len(arr) > 1:
        mean_centered = arr - np.mean(arr)
        zero_crossings = np.sum(np.diff(np.sign(mean_centered)) != 0)
        features[f'{prefix}_zcr'] = zero_crossings / len(arr)
    else:
        features[f'{prefix}_zcr'] = 0
    
    return features


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


def load_data_with_advanced_features(partition_id, test_fraction=0.2):
    """Carica dati e aggiunge advanced features."""
    data_path = Path(__file__).parent / "clients_data" / f"group{partition_id}"
    
    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files:
        return None, None, None, None
    
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, sep=';')
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    
    # Preprocessing time-series BASE (mean, min, max, std)
    timeseries_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
    
    for col in timeseries_cols:
        if col not in data.columns:
            continue
        
        prefix = col.replace('_time_series', '')
        series = data[col].apply(parse_series)
        
        # Base statistics (come exp_008)
        data[f'{prefix}_mean'] = series.apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_min'] = series.apply(lambda x: np.min(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_max'] = series.apply(lambda x: np.max(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_std'] = series.apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
        
        # ADVANCED FEATURES ✨
        advanced_features = series.apply(lambda x: extract_advanced_timeseries_features(x, prefix))
        
        # Converti lista di dict in DataFrame
        advanced_df = pd.DataFrame(advanced_features.tolist())
        
        # Aggiungi al dataframe principale
        for adv_col in advanced_df.columns:
            data[adv_col] = advanced_df[adv_col]
        
        # Rimuovi colonna originale
        data.drop(columns=[col], inplace=True)
    
    # Rimuovi righe con troppi NaN
    data = data.dropna(thresh=len(data.columns) * 0.5)
    
    # Riempi NaN rimanenti con mediana (o 0 se mediana è NaN)
    for col in data.columns:
        if data[col].dtype in ['float64', 'int64'] and data[col].isna().any():
            median_val = data[col].median()
            if pd.isna(median_val):
                # Se mediana è NaN (colonna tutta NaN), usa 0
                data[col] = data[col].fillna(0)
            else:
                data[col] = data[col].fillna(median_val)
    
    # Separa features e label
    feature_cols = [col for col in data.columns if col != 'label']
    X = data[feature_cols]
    y = data['label']
    
    # Train/test split
    split_idx = int(len(data) * (1 - test_fraction))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test


def select_best_features(X_train, y_train, k=25):
    """
    Seleziona le top-k feature più importanti usando SelectKBest.
    
    Args:
        X_train: training features
        y_train: training labels
        k: numero di feature da selezionare
        
    Returns:
        selector: fitted SelectKBest object
        selected_features: lista nomi feature selezionate
    """
    print(f"\n🔍 Feature Selection: {X_train.shape[1]} → {k} features")
    
    # Usa f_regression per ranking
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X_train, y_train)
    
    # Ottieni nomi feature selezionate
    feature_mask = selector.get_support()
    selected_features = X_train.columns[feature_mask].tolist()
    
    return selector, selected_features


def train_rf_advanced(num_rounds=10, n_estimators=200, k_features=25):
    """
    Training RF con advanced features e feature selection.
    """
    print("="*70)
    print("🌲🔬 RANDOM FOREST + ADVANCED FEATURES - ESPERIMENTO 014")
    print("="*70)
    print(f"Round FL: {num_rounds}")
    print(f"Alberi per client: {n_estimators}")
    print(f"Features selezionate: top {k_features}")
    print()
    print("✨ ADVANCED FEATURES:")
    print("   • Percentili (25, 75, 90, IQR)")
    print("   • Shape: skewness, kurtosis")
    print("   • Variabilità: range, CV, MAD")
    print("   • Trend: slope, R²")
    print("   • Change: diffs, std_diff, max_abs_diff")
    print("   • Peaks: count, rate, mean_value")
    print("   • Entropy, Autocorrelation, Energy, ZCR")
    print()
    
    # Configurazione RF (identica a exp_008)
    rf_params = {
        "n_estimators": n_estimators,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": 'sqrt',
        "n_jobs": -1,
        "random_state": 42
    }
    
    all_models = []
    all_feature_names = []
    
    # Feature selection globale (su tutti i dati)
    print("🔧 Feature Selection su tutti i dati...")
    all_X_train = []
    all_y_train = []
    
    for partition_id in range(9):
        X_train, _, y_train, _ = load_data_with_advanced_features(partition_id, 0.2)
        if X_train is not None:
            all_X_train.append(X_train)
            all_y_train.append(y_train)
        print(f"   Client {partition_id}: ✓", end=" ", flush=True)
    
    X_all = pd.concat(all_X_train, ignore_index=True)
    y_all = pd.concat(all_y_train, ignore_index=True)
    
    print(f"\n📊 Dataset completo: {X_all.shape[0]} samples, {X_all.shape[1]} features")
    
    # Feature selection
    selector, selected_features = select_best_features(X_all, y_all, k=k_features)
    
    print(f"   ✅ Selected {len(selected_features)} best features\n")
    
    # Training per round
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*70}")
        print(f"🔄 ROUND {round_num}/{num_rounds}")
        print(f"{'='*70}")
        
        round_models = []
        round_rmses = []
        
        for partition_id in range(9):
            X_train, X_test, y_train, y_test = load_data_with_advanced_features(
                partition_id, 0.2
            )
            
            if X_train is None:
                continue
            
            # Applica feature selection
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
        round_rmses = []
        
        for partition_id in range(9):
            X_train, X_test, y_train, y_test = load_data_with_advanced_features(
                partition_id, 0.2
            )
            
            if X_train is None:
                continue
            
            # Applica feature selection
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
            
            # Training RF
            rf_params['random_state'] = 42 + round_num
            rf = RandomForestRegressor(**rf_params)
            rf.fit(X_train_selected, y_train)
            
            # Predizione
            y_pred = rf.predict(X_test_selected)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            round_rmses.append(rmse)
            
            round_models.append(rf)
        
        avg_rmse = np.mean(round_rmses)
        print(f"   Avg RMSE: {avg_rmse:.2f} | Trained {len(round_models)} clients")
        
        all_models.extend(round_models)
    
    print(f"\n{'='*70}")
    print(f"✅ Training completato!")
    print(f"   Modelli allenati: {len(all_models)}")
    print(f"   Features per modello: {len(selected_features)}")
    print(f"{'='*70}")
    
    # Salva modelli e selector
    output_dir = Path(__file__).parent / "experiments" / "exp_014_rf_advanced_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "rf_advanced_models_exp014.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'models': all_models,
            'feature_selector': selector,
            'selected_features': selected_features,
            'n_estimators': n_estimators,
            'num_rounds': num_rounds
        }, f)
    
    print(f"\n💾 Modelli salvati: {model_path}")
    
    return all_models, selector, selected_features


if __name__ == "__main__":
    train_rf_advanced(num_rounds=10, n_estimators=200, k_features=25)
