"""Training XGBoost con Meno Round - Esperimento 007"""
import xgboost as xgb
import numpy as np
import pandas as pd
from pathlib import Path
from task import load_data


def train_xgb_reduced_rounds(num_rounds=5, n_estimators=100):
    """
    Training XGBoost con MENO round per ridurre overfitting.
    Strategia: Fermare prima che il modello memorizzi troppo.
    """
    print("="*70)
    print("⏱️  XGBOOST - RIDUZIONE ROUND - ESPERIMENTO 007")
    print("="*70)
    print(f"Round FL: {num_rounds} (baseline: 10, riduzione 50%)")
    print(f"Alberi per client: {n_estimators} (baseline: 100)")
    print(f"Alberi totali: {num_rounds * 9 * n_estimators} = {num_rounds * 9 * n_estimators} (baseline: 1000)")
    print()
    print("🎯 STRATEGIA: RIDURRE complessità")
    print("   • 5 round invece di 10 → 500 alberi totali vs 1000")
    print("   • Fermare training prima dell'overfitting")
    print("   • Stessi hyperparameter vincenti del baseline")
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
            
            # Carica dati (SENZA feature engineering)
            train_dmatrix, valid_dmatrix, num_train, num_test = load_data(
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
    
    # Salva modello finale 
    exp_dir = Path(__file__).parent / "experiments" / "exp_007_reduced_rounds"
    exp_dir.mkdir(parents=True, exist_ok=True)
    output_path = exp_dir / "final_model_exp007.json"
    global_model.save_model(str(output_path))
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETATO!")
    print("="*70)
    print(f"\n💾 Modello salvato: {output_path}")
    print(f"   Alberi nel modello: {global_model.num_boosted_rounds()}")
    print(f"   Features: {global_model.num_features()}")
    print(f"\n⏱️  Riduzione: {num_rounds} round vs 10 baseline → {(1 - num_rounds/10)*100:.0f}% meno alberi")
    
    print(f"\n📊 Usa 'python inference_exp007.py' per generare predizioni")
    
    return global_model

if __name__ == "__main__":
    model = train_xgb_reduced_rounds(num_rounds=5, n_estimators=100)
