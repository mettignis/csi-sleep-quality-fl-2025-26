"""Training XGBoost con Feature Engineering - Esperimento 006"""
import xgboost as xgb
import numpy as np
import pandas as pd
from pathlib import Path

# Importa funzioni
import sys
sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import add_engineered_features


def load_data_with_features(partition_id: int, test_fraction: float = 0.2):
    """
    Carica dati con feature engineering applicato.
    
    Args:
        partition_id: ID del client (0-8)
        test_fraction: Frazione per validation set
        
    Returns:
        train_dmatrix, valid_dmatrix, num_train, num_test
    """
    # Percorso ai dati preprocessati
    data_path = Path(__file__).parent / "clients_data" / f"group{partition_id}"
    
    # Trova tutti i CSV del gruppo
    csv_files = sorted(data_path.glob("*.csv"))
    
    if not csv_files:
        print(f"⚠️  Nessun file trovato per partition {partition_id}")
        return None, None, 0, 0
    
    # Carica e unisci tutti i CSV
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, sep=';')
        dfs.append(df)
    
    data = pd.concat(dfs, ignore_index=True)
    
    # Preprocessing time-series (come nel baseline)
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
    
    # FEATURE ENGINEERING! 🔧
    original_features = len(data.columns) - 1  # -1 per label
    data = add_engineered_features(data)
    new_features = len(data.columns) - 1
    print(f"   📊 Feature: {original_features} → {new_features} (+{new_features - original_features})")
    
    # Separa features e label
    if 'label' not in data.columns:
        print(f"⚠️  Colonna 'label' non trovata!")
        return None, None, 0, 0
    
    X = data.drop(columns=['label'])
    y = data['label']
    
    # Split train/validation
    split_idx = int(len(X) * (1 - test_fraction))
    
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    
    # Converti a DMatrix
    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    valid_dmatrix = xgb.DMatrix(X_test, label=y_test)
    
    return train_dmatrix, valid_dmatrix, len(X_train), len(X_test)


def train_xgb_feature_eng(num_rounds=10, n_estimators=100):
    """
    Training XGBoost con feature engineering.
    Usa stessi hyperparameter del baseline vincente.
    """
    print("="*70)
    print("🎨 XGBOOST + FEATURE ENGINEERING - ESPERIMENTO 006")
    print("="*70)
    print(f"Round FL: {num_rounds} (baseline)")
    print(f"Alberi per client: {n_estimators} (baseline)")
    print()
    print("🔧 FEATURE ENGINEERING:")
    print("   • Interazioni: hr×stress, sleep×hr, resp×stress")
    print("   • Ratios: deep/rem/light sleep, activity intensity")
    print("   • Advanced stats: range, CV, variability index")
    print("   • Transformations: log(calories), log(distance)")
    print()
    
    # Configurazione XGBoost (IDENTICA AL BASELINE)
    params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "eta": 0.3,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "rmse"
    }
    
    num_clients = 9
    global_model = None
    
    for round_num in range(num_rounds):
        print(f"\n{'='*70}")
        print(f"🔄 ROUND {round_num + 1}/{num_rounds}")
        print(f"{'='*70}\n")
        
        local_models = []
        local_weights = []
        
        for client_id in range(num_clients):
            print(f"👤 Client {client_id}:")
            
            # Carica dati CON feature engineering
            train_dmatrix, valid_dmatrix, num_train, num_test = load_data_with_features(
                partition_id=client_id,
                test_fraction=0.2
            )
            
            if num_train == 0:
                print(f"   ⚠️  Skipping (0 train samples)")
                continue
            
            print(f"   Train: {num_train}, Test: {num_test}")
            
            # Training
            if global_model is not None:
                xgb_train = xgb.train(
                    params,
                    train_dmatrix,
                    num_boost_round=n_estimators,
                    evals=[(valid_dmatrix, "validation")],
                    verbose_eval=False,
                    xgb_model=global_model
                )
            else:
                xgb_train = xgb.train(
                    params,
                    train_dmatrix,
                    num_boost_round=n_estimators,
                    evals=[(valid_dmatrix, "validation")],
                    verbose_eval=False
                )
            
            # Valutazione
            train_pred = xgb_train.predict(train_dmatrix)
            train_labels = train_dmatrix.get_label()
            train_rmse = np.sqrt(np.mean((train_pred - train_labels) ** 2))
            train_mae = np.mean(np.abs(train_pred - train_labels))
            
            print(f"   Train RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}")
            
            if num_test > 0:
                test_pred = xgb_train.predict(valid_dmatrix)
                test_labels = valid_dmatrix.get_label()
                test_rmse = np.sqrt(np.mean((test_pred - test_labels) ** 2))
                test_mae = np.mean(np.abs(test_pred - test_labels))
                print(f"   Test  RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}")
            
            local_models.append(xgb_train)
            local_weights.append(num_train)
        
        # Aggregazione
        print(f"\n🔄 Aggregazione {len(local_models)} modelli...")
        
        if len(local_models) > 0:
            global_model = local_models[0]
            
        print(f"✓ Round {round_num + 1} completato")
    
    # Salva modello finale in experiments/
    exp_dir = Path(__file__).parent / "experiments" / "exp_006_feature_engineering"
    exp_dir.mkdir(parents=True, exist_ok=True)
    output_path = exp_dir / "final_model_exp006.json"
    global_model.save_model(str(output_path))
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETATO!")
    print("="*70)
    print(f"\n💾 Modello salvato: {output_path}")
    print(f"   Alberi nel modello: {global_model.num_boosted_rounds()}")
    print(f"   Features: {global_model.num_features()}")
    
    print(f"\n📊 Usa 'python inference_exp006.py' per generare predizioni")
    
    return global_model

if __name__ == "__main__":
    model = train_xgb_feature_eng(num_rounds=10, n_estimators=100)
