# RNN Federated Learning Experiment

## Configuration
- **Model**: LSTM Sequential (64→32 units with dropout 0.2, Dense 16 units with dropout 0.3)
- **Clients**: 9
- **Rounds**: 15
- **Local Epochs**: 10
- **Batch Size**: 16
- **Learning Rate**: 0.001
- **Sequence Length**: 7 days
- **Features**: 39

## Results
- **Final Val Loss**: 123.50632036940952
- **Final Val MAE**: 8.200467856219465

## Files
- `rnn_federated_model_20251228_164701.keras`: Trained model (Keras format)
- `rnn_federated_weights_20251228_164701.pkl`: Model weights
- `training_history_20251228_164701.json`: Training history
- `config_20251228_164701.json`: Configuration

## Training Date
20251228_164701
