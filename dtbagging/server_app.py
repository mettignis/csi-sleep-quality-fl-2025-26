"""Server Flower per Federated Learning."""
import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, Context
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedXgbBagging, FedXgbCyclic
import warnings

from .task import replace_keys

warnings.filterwarnings("ignore", category=DeprecationWarning)

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main del server."""
    # Configurazione
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    train_method = context.run_config["train-method"]
    
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]
    
    # Modello iniziale vuoto
    global_model = b""
    arrays = ArrayRecord([np.frombuffer(global_model, dtype=np.uint8)])
    
    # Strategia
    if train_method == "bagging":
        strategy = FedXgbBagging(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
        )
    else:
        strategy = FedXgbCyclic()
    
    # Esegui training federato
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )
    
    # Salva modello finale
    bst = xgb.Booster(params=params)
    global_model = bytearray(result.arrays["0"].numpy().tobytes())
    bst.load_model(global_model)
    
    print("\n✅ Salvataggio modello finale...")
    bst.save_model("final_model.json")
    print("📊 Modello salvato in:  final_model.json")