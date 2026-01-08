"""
Client Flower per Federated Learning con XGBoost
"""
import warnings
import numpy as np
import xgboost as xgb
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from sklearn.metrics import mean_squared_error

from .task import load_data, replace_keys

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

app = ClientApp()

def _local_boost(bst_input, num_local_round, train_dmatrix, train_method):
    # Boosting locale
    for i in range(num_local_round):
        bst_input.update(train_dmatrix, bst_input.num_boosted_rounds())
    
    if train_method == "bagging": 
        bst = bst_input[
            bst_input.num_boosted_rounds() - num_local_round :  
            bst_input.num_boosted_rounds()
        ]
    else: 
        bst = bst_input
    
    return bst


@app.train()
def train(msg: Message, context: Context) -> Message:
    # Configurazione
    partition_id = context.node_config["partition-id"]
    num_local_round = context.run_config["local-epochs"]
    train_method = context.run_config["train-method"]
    test_fraction = context.run_config["test-fraction"]
    
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]
    
    # Carica dati
    train_dmatrix, _, num_train, _ = load_data(
        partition_id=partition_id,
        test_fraction=test_fraction
    )
    
    global_round = msg.content["config"]["server-round"]
    
    if global_round == 1:
        # Primo round: training da zero
        bst = xgb.train(
            params,
            train_dmatrix,
            num_boost_round=num_local_round,
        )
    else:
        # Round successivi: continua dal modello globale
        bst = xgb. Booster(params=params)
        global_model = bytearray(msg.content["arrays"]["0"]. numpy().tobytes())
        bst.load_model(global_model)
        bst = _local_boost(bst, num_local_round, train_dmatrix, train_method)
    
    # Serializza il modello
    local_model = bst.save_raw("json")
    model_np = np.frombuffer(local_model, dtype=np.uint8)
    
    # Prepara la risposta
    model_record = ArrayRecord([model_np])
    metrics = {"num-examples": num_train}
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics":  metric_record})
    
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    partition_id = context.node_config["partition-id"]
    test_fraction = context.run_config["test-fraction"]
    
    cfg = replace_keys(unflatten_dict(context.run_config))
    params = cfg["params"]
    
    # Carica dati di test
    _, valid_dmatrix, _, num_val = load_data(
        partition_id=partition_id,
        test_fraction=test_fraction
    )
    
    # Carica modello globale
    bst = xgb. Booster(params=params)
    global_model = bytearray(msg.content["arrays"]["0"].numpy().tobytes())
    bst.load_model(global_model)
    
    # Prediction and Validation
    y_pred = bst.predict(valid_dmatrix)
    y_test = valid_dmatrix.get_label()
    mse = mean_squared_error(y_test, y_pred)
    
    # Prepara risposta
    metrics = {
        "mse": mse,
        "num-examples": num_val,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    
    return Message(content=content, reply_to=msg)