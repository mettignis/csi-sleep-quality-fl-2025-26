"""Training Random Forest Migliorato - Esperimento 008"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

def load_data_for_rf(partition_id, test_fraction=0.2):
    """Carica dati per Random Forest."""
    data_path = Path(__file__).parent / "clients_data" / f"group{partition_id}"
    
    # Trova tutti i CSV del gruppo
    csv_files = sorted(data_path.glob("*.csv"))
    
    if not csv_files:
        print(f"⚠️  Nessun file trovato per partition {partition_id}")
        return None, None, None, None
    
    # Carica e unisci tutti i CSV
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, sep=';')
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    
    # Preprocessing time-series
    timeseries_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
    
    for col in timeseries_cols:
        if col not in data.columns:
            continue
            
        prefix = col.replace('_time_series', '')
        
        def parse_series(x):
            if pd.isna(x):
                return []
            try:
                s = str(x).strip('[]')
                values = [float(v) for v in s.split(',') if v.strip() and float(v.strip()) >= 0]
                return values
            except:
                return []
        
        series = data[col].apply(parse_series)
        
        # Statistiche base
        data[f'{prefix}_mean'] = series.apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_min'] = series.apply(lambda x: np.min(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_max'] = series.apply(lambda x: np.max(x) if len(x) > 0 else np.nan)
        data[f'{prefix}_std'] = series.apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
        
        data.drop(columns=[col], inplace=True)
    
    # Separa features e label
    if 'label' not in data.columns:
        print(f"⚠️  Colonna 'label' non trovata!")
        return None, None, None, None
    
    X = data.drop(columns=['label'])
    y = data['label']
    
    # Split train/validation
    split_idx = int(len(X) * (1 - test_fraction))
    
    X_train = X.iloc[:split_idx].values
    X_test = X.iloc[split_idx:].values
    y_train = y.iloc[:split_idx].values
    y_test = y.iloc[split_idx:].values
    
    return X_train, X_test, y_train, y_test


def train_rf_improved(num_rounds=10, n_estimators=200, max_depth=15):
    """
    Training Random Forest MIGLIORATO.
    
    Miglioramenti vs exp_002:
    - Trees: 100 → 200 (più modelli diversi)
    - Max depth: 10 → 15 (più profondità)
    - Min samples split: 2 → 5 (più generalizzazione)
    """
    print("="*70)
    print("🌲 RANDOM FOREST MIGLIORATO - ESPERIMENTO 008")
    print("="*70)
    print(f"Round FL: {num_rounds}")
    print(f"Alberi per client: {n_estimators} (exp_002: 100)")
    print(f"Max depth: {max_depth} (exp_002: 10)")
    print(f"Min samples split: 5 (exp_002: 2)")
    print()
    print("🎯 STRATEGIA: Migliorare RF")
    print("   • exp_002 score: 12.95967 (solo 0.21 peggio del baseline!)")
    print("   • exp_002 std: 2.32 (troppo conservativo)")
    print("   • Aumentiamo alberi e profondità per più variabilità")
    print()
    
    num_clients = 9
    all_models = []
    
    for round_num in range(num_rounds):
        print(f"\n{'='*70}")
        print(f"🔄 ROUND {round_num + 1}/{num_rounds}")
        print(f"{'='*70}\n")
        
        round_models = []
        
        for client_id in range(num_clients):
            print(f"👤 Client {client_id}:")
            
            # Carica dati
            X_train, X_test, y_train, y_test = load_data_for_rf(
                partition_id=client_id,
                test_fraction=0.2
            )
            
            if X_train is None or len(X_train) == 0:
                print(f"   ⚠️  Skipping (no data)")
                continue
            
            print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Training Random Forest MIGLIORATO
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=5,  # Più generalizzazione
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42 + round_num,  # Seed diverso per round
                n_jobs=-1
            )
            
            rf.fit(X_train, y_train)
            
            # Valutazione
            train_pred = rf.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            train_mae = mean_absolute_error(y_train, train_pred)
            
            print(f"   Train RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}")
            
            if len(X_test) > 0:
                test_pred = rf.predict(X_test)
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                test_mae = mean_absolute_error(y_test, test_pred)
                print(f"   Test  RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}")
            
            round_models.append(rf)
        
        all_models.extend(round_models)
        print(f"\n✓ Round {round_num + 1} completato: {len(round_models)} modelli")
    
    # Salva tutti i modelli
    exp_dir = Path(__file__).parent / "experiments" / "exp_008_rf_improved"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = exp_dir / "final_model_rf_exp008.pkl"
    
    with open(output_path, 'wb') as f:
        pickle.dump(all_models, f)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETATO!")
    print("="*70)
    print(f"\n💾 Modelli salvati: {output_path}")
    print(f"   Numero modelli: {len(all_models)}")
    print(f"   Alberi per modello: {n_estimators}")
    print(f"   Alberi totali: {len(all_models) * n_estimators}")
    
    print(f"\n📊 Usa 'python inference_exp008.py' per generare predizioni")
    
    return all_models

if __name__ == "__main__":
    models = train_rf_improved(num_rounds=10, n_estimators=200, max_depth=15)
