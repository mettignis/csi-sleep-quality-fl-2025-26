"""Training Federated Learning con Random Forest invece di XGBoost."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

def load_data_for_rf(partition_id, test_fraction=0.2):
    """Carica dati per Random Forest (ritorna array, non DMatrix)."""
    import pandas as pd
    from pathlib import Path
    
    data_file = Path(__file__).parent / "ready_for_flwr" / f"client_{partition_id}.csv"
    
    if not data_file.exists():
        raise FileNotFoundError(f"File non trovato: {data_file}")
    
    data = pd.read_csv(data_file)
    data = data.dropna(axis=1, how='all')
    
    feature_cols = [col for col in data.columns 
                    if col not in ['day', 'label'] 
                    and not col.endswith('_time_series')
                    and data[col].dtype in ['int64', 'float64']]
    
    X = data[feature_cols]
    y = data['label']
    
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    split_idx = int(len(X) * (1 - test_fraction))
    
    X_train = X.iloc[:split_idx].values
    y_train = y.iloc[:split_idx].values
    X_test = X.iloc[split_idx:].values
    y_test = y.iloc[split_idx:].values
    
    return X_train, X_test, y_train, y_test

def train_local_fl_rf(num_rounds=10, n_estimators=100, max_depth=10):
    """
    Training Federated Learning con Random Forest.
    
    Args:
        num_rounds: Numero di round FL
        n_estimators: Numero di alberi per client
        max_depth: Profondità massima alberi
    """
    print("="*70)
    print("🌲 FEDERATED LEARNING CON RANDOM FOREST")
    print("="*70)
    print(f"Round: {num_rounds}")
    print(f"Trees per client: {n_estimators}")
    print(f"Max depth: {max_depth}")
    print()
    
    # Numero di client
    num_clients = 9
    
    # Lista per aggregare i modelli
    all_models = []
    
    for round_num in range(num_rounds):
        print(f"\n{'='*70}")
        print(f"🔄 ROUND {round_num + 1}/{num_rounds}")
        print(f"{'='*70}\n")
        
        round_models = []
        
        for client_id in range(num_clients):
            print(f"👤 Client {client_id}:")
            
            # Carica dati del client
            X_train, X_test, y_train, y_test = load_data_for_rf(
                partition_id=client_id,
                test_fraction=0.2
            )
            
            n_train = len(X_train)
            n_test = len(X_test)
            
            if n_train == 0:
                print(f"   ⚠️  Skipping (0 train samples)")
                continue
            
            print(f"   Train: {n_train}, Test: {n_test}")
            
            # Train Random Forest
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42 + round_num,  # Seed diverso per round
                n_jobs=-1  # Usa tutti i core
            )
            
            model.fit(X_train, y_train)
            
            # Valuta
            train_pred = model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            train_mae = mean_absolute_error(y_train, train_pred)
            
            print(f"   Train RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}")
            
            if n_test > 0:
                test_pred = model.predict(X_test)
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                test_mae = mean_absolute_error(y_test, test_pred)
                print(f"   Test  RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}")
            
            round_models.append(model)
        
        # Aggiungi modelli del round
        all_models.extend(round_models)
        
        print(f"\n✓ Round {round_num + 1} completato. Modelli totali: {len(all_models)}")
    
    print("\n" + "="*70)
    print("🎯 AGGREGAZIONE FINALE")
    print("="*70)
    print(f"Modelli da aggregare: {len(all_models)}")
    
    # Per RF, l'aggregazione è semplicemente tenere tutti gli alberi
    # Creiamo un ensemble che usa tutti i modelli
    print("✓ Ensemble Random Forest creato")
    
    # Salva tutti i modelli
    output_path = Path(__file__).parent / "final_model_rf.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(all_models, f)
    
    print(f"\n💾 Modello salvato: {output_path}")
    print(f"   Modelli nell'ensemble: {len(all_models)}")
    print(f"   Alberi totali: {len(all_models) * n_estimators}")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETATO!")
    print("="*70)
    
    return all_models

if __name__ == "__main__":
    import sys
    
    # Parametri configurabili
    num_rounds = 10
    n_estimators = 100  # Alberi per client
    max_depth = 10
    
    # Parsing argomenti opzionali
    if len(sys.argv) > 1:
        num_rounds = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_estimators = int(sys.argv[2])
    if len(sys.argv) > 3:
        max_depth = int(sys.argv[3])
    
    # Train
    models = train_local_fl_rf(
        num_rounds=num_rounds,
        n_estimators=n_estimators,
        max_depth=max_depth
    )
    
    print(f"\n📊 Usa 'python inference_rf.py' per generare predizioni")
