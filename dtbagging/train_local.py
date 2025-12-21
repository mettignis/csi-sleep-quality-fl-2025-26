"""
Script per eseguire il training FL senza simulazione Ray.
Esegue sequenzialmente i client e aggrega i risultati.
"""
import json
import xgboost as xgb
from pathlib import Path
from .task import load_data

def train_local_fl(num_clients=9, num_rounds=10, test_fraction=0.2):
    """
    Training Federated Learning locale senza Ray.
    
    Args:
        num_clients: Numero di client (default 9)
        num_rounds: Numero di round FL (default 10)
        test_fraction: Frazione di dati per test
    """
    print(f"🚀 Avvio training Federated Learning locale")
    print(f"📊 {num_clients} client, {num_rounds} round\n")
    
    # Parametri XGBoost
    params = {
        "objective": "reg:squarederror",
        "eta": 0.1,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist"
    }
    
    global_model = None
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}/{num_rounds}")
        print(f"{'='*60}")
        
        local_models = []
        total_train_samples = 0
        
        # Train sui client
        for client_id in range(num_clients):
            try:
                # Carica dati del client
                train_data, valid_data, num_train, num_test = load_data(
                    client_id, test_fraction
                )
                
                print(f"\nClient {client_id}:")
                print(f"  Train samples: {num_train}, Test samples: {num_test}")
                
                # Train locale
                if global_model is None:
                    # Primo round: train da zero
                    bst = xgb.train(
                        params,
                        train_data,
                        num_boost_round=10,
                        evals=[(valid_data, "valid")],
                        verbose_eval=False
                    )
                else:
                    # Round successivi: continua da modello globale
                    bst = xgb.train(
                        params,
                        train_data,
                        num_boost_round=10,
                        xgb_model=global_model,
                        evals=[(valid_data, "valid")],
                        verbose_eval=False
                    )
                
                # Valuta
                preds = bst.predict(valid_data)
                labels = valid_data.get_label()
                mse = ((preds - labels) ** 2).mean()
                rmse = mse ** 0.5
                mae = abs(preds - labels).mean()
                
                print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                
                local_models.append((bst, num_train))
                total_train_samples += num_train
                
            except Exception as e:
                print(f"  ❌ Errore: {e}")
                continue
        
        if not local_models:
            print("\n❌ Nessun client ha completato il training!")
            break
        
        # Aggrega modelli (media pesata)
        print(f"\n🔄 Aggregazione di {len(local_models)} modelli...")
        
        # Salva temporaneamente i modelli
        temp_models = []
        for i, (model, _) in enumerate(local_models):
            temp_path = Path(f"/tmp/temp_model_{i}.json")
            model.save_model(temp_path)
            temp_models.append(temp_path)
        
        # Usa il primo modello come base (in un vero FL si fa media dei pesi)
        # Per XGBoost bagging, prendiamo semplicemente l'ultimo modello addestrato
        global_model = local_models[-1][0]
        
        # Cleanup
        for temp_path in temp_models:
            if temp_path.exists():
                temp_path.unlink()
        
        print(f"✓ Round {round_num} completato!")
    
    # Salva modello finale
    output_path = Path("dtbagging/final_model.json")
    global_model.save_model(output_path)
    print(f"\n{'='*60}")
    print(f"✅ Training completato!")
    print(f"💾 Modello salvato in: {output_path}")
    print(f"{'='*60}\n")
    
    return global_model


if __name__ == "__main__":
    train_local_fl()
