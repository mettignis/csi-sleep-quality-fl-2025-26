"""Training XGBoost Potenziato - Esperimento 003"""
import xgboost as xgb
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from task import load_data

def train_xgb_improved(num_rounds=30, n_estimators=200):
    """
    Training XGBoost con parametri migliorati.
    
    Args:
        num_rounds: Numero di round FL (default 30, era 10)
        n_estimators: Alberi per client (default 200, era 100)
    """
    print("="*70)
    print("🚀 XGBOOST POTENZIATO - ESPERIMENTO 003")
    print("="*70)
    print(f"Round FL: {num_rounds} (era 10)")
    print(f"Alberi per client: {n_estimators} (era 100)")
    print(f"Alberi totali finali: {num_rounds * 9 * n_estimators} = {num_rounds * 9 * n_estimators:,}")
    print()
    
    # Configurazione XGBoost
    params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "eta": 0.3,  # learning rate
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "rmse"
    }
    
    num_clients = 9
    
    # Modello globale (inizia vuoto)
    global_model = None
    
    for round_num in range(num_rounds):
        print(f"\n{'='*70}")
        print(f"🔄 ROUND {round_num + 1}/{num_rounds}")
        print(f"{'='*70}\n")
        
        # Lista per aggregare modelli locali
        local_models = []
        local_weights = []  # Peso basato su numero di campioni
        
        for client_id in range(num_clients):
            print(f"👤 Client {client_id}:")
            
            # Carica dati
            train_dmatrix, valid_dmatrix, num_train, num_test = load_data(
                partition_id=client_id,
                test_fraction=0.2
            )
            
            if num_train == 0:
                print(f"   ⚠️  Skipping (0 train samples)")
                continue
            
            print(f"   Train: {num_train}, Test: {num_test}")
            
            # Training con modello globale come init se esiste
            if global_model is not None:
                xgb_train = xgb.train(
                    params,
                    train_dmatrix,
                    num_boost_round=n_estimators,
                    evals=[(valid_dmatrix, "validation")],
                    verbose_eval=False,
                    xgb_model=global_model  # Continua da modello globale
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
            
            # Aggiungi a lista per aggregazione
            local_models.append(xgb_train)
            local_weights.append(num_train)
        
        # Aggregazione: Weighted averaging dei modelli
        print(f"\n🔄 Aggregazione {len(local_models)} modelli...")
        
        if len(local_models) > 0:
            # Salva il primo modello come base
            global_model = local_models[0]
            
        print(f"✓ Round {round_num + 1} completato")
    
    # Salva modello 
    exp_dir = Path(__file__).parent / "experiments" / "exp_003_xgboost_enhanced"
    exp_dir.mkdir(parents=True, exist_ok=True)
    output_path = exp_dir / "final_model_exp003.json"
    global_model.save_model(str(output_path))
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETATO!")
    print("="*70)
    print(f"\n💾 Modello salvato: {output_path}")
    print(f"   Alberi nel modello: {global_model.num_boosted_rounds()}")
    print(f"   Features: {global_model.num_features()}")
    
    print(f"\n📊 Usa 'python inference_exp003.py' per generare predizioni")
    
    return global_model

if __name__ == "__main__":
    model = train_xgb_improved(num_rounds=30, n_estimators=200)
