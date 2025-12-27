# RNN Federated Learning for Sleep Quality Prediction

This implementation uses LSTM (Long Short-Term Memory) networks in a federated learning setting to predict sleep quality from time series physiological data.

## Overview

### Architecture
- **Model**: Bidirectional LSTM with 2 layers (64, 32 units)
- **Task**: Regression to predict sleep quality scores
- **Input**: Sequential time series data (7 days by default)
- **Features**: 40+ physiological metrics including:
  - Heart rate statistics (mean, min, max, std)
  - Respiration data
  - Stress levels
  - Sleep patterns
  - Activity metrics

### Federated Learning
- **Algorithm**: FedAvg (Federated Averaging)
- **Clients**: 9 clients (one per group)
- **Privacy**: Each client's raw data stays local
- **Aggregation**: Weighted average based on dataset sizes

## Files

### Training
- `train_rnn_fl.py` - Main training script with federated learning implementation

### Inference
- `inference_rnn.py` - Generate predictions using trained model

### Data
- `ready_for_flwr/client_*.csv` - Preprocessed client data (9 files)
- `test_data/x_test.csv` - Test data for predictions

## Usage

### 1. Training

Basic training with default parameters:
```bash
python -m dtbagging.train_rnn_fl
```

The script will:
- Load data from 9 clients
- Train for 15 rounds
- Perform 10 local epochs per round per client
- Aggregate models using FedAvg
- Save the final model and training history

**Output files** (in `experiments/exp_rnn_federated/`):
- `rnn_federated_model_*/` - Trained model (TensorFlow SavedModel format)
- `rnn_federated_weights_*.npy` - Model weights
- `training_history_*.json` - Training metrics per round
- `config_*.json` - Training configuration
- `README.md` - Experiment summary

### 2. Inference

Generate predictions for test data:
```bash
python -m dtbagging.inference_rnn \
    --model experiments/exp_rnn_federated/rnn_federated_model_* \
    --input test_data/x_test.csv \
    --output submission_rnn.csv
```

Or run without arguments to auto-detect the latest model:
```bash
python -m dtbagging.inference_rnn
```

## Configuration

Default hyperparameters in `train_rnn_fl.py`:
```python
num_clients = 9          # Number of federated clients
num_rounds = 15          # FL rounds
local_epochs = 10        # Epochs per client per round
batch_size = 16          # Training batch size
learning_rate = 0.001    # Adam optimizer learning rate
sequence_length = 7      # Days of history for LSTM
test_fraction = 0.2      # Validation split
```

You can modify these in the `main()` function.

## How It Works

### 1. Data Processing
Each client's data is loaded and preprocessed:
- Features are normalized per-client (for privacy)
- Sequential windows of `sequence_length` days are created
- Data is split into train/validation sets

### 2. Model Architecture
```
Input (sequence_length, num_features)
    ↓
LSTM(64 units, dropout=0.2, return_sequences=True)
    ↓
LSTM(32 units, dropout=0.2)
    ↓
Dense(16, relu)
    ↓
Dropout(0.3)
    ↓
Dense(1) → Output (sleep quality prediction)
```

### 3. Federated Training Loop
For each round:
1. **Local Training**: Each client trains on their local data
   - Initialize with global model weights
   - Train for `local_epochs`
   - Return updated weights and metrics
   
2. **Aggregation**: Server performs FedAvg
   - Weighted average of client weights
   - Weights proportional to dataset sizes
   - Update global model

3. **Evaluation**: Global model evaluated on all validation data

### 4. Privacy Benefits
- Raw data never leaves clients
- Only model updates are shared
- Each client uses local normalization
- No data pooling required

## Advantages of RNN for This Task

1. **Temporal Dependencies**: Captures patterns across multiple days
2. **Sequential Nature**: Sleep quality depends on previous days' data
3. **Context Awareness**: LSTM memory cells retain important information
4. **Feature Interactions**: Learns complex relationships between physiological signals

## Expected Performance

The LSTM model should:
- Learn temporal patterns in sleep quality
- Improve over rounds as more clients contribute
- Achieve MAE (Mean Absolute Error) around 5-10 points

Metrics tracked:
- **Training Loss/MAE**: Per-client performance
- **Validation Loss/MAE**: Global model performance
- **Convergence**: Typically stabilizes after 10-15 rounds

## Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce `batch_size` or `sequence_length`

### Issue: Model not learning
**Solution**: Try:
- Increase `learning_rate` to 0.01
- Increase `local_epochs`
- Check data quality (missing values)

### Issue: Overfitting
**Solution**: 
- Increase dropout rates
- Reduce model complexity
- Use more regularization

## Comparison with XGBoost

| Aspect | RNN/LSTM | XGBoost |
|--------|----------|---------|
| Temporal modeling | ✅ Excellent | ❌ Limited |
| Feature engineering | ❌ Minimal needed | ✅ Extensive required |
| Training time | ⚠️ Slower | ✅ Fast |
| Interpretability | ❌ Black box | ✅ Feature importance |
| Small data | ⚠️ May overfit | ✅ Robust |
| Sequences | ✅ Native support | ❌ Manual features |

## Future Improvements

1. **Bidirectional LSTM**: Process sequences both ways
2. **Attention Mechanism**: Focus on important time steps
3. **Per-client personalization**: Maintain client-specific layers
4. **Differential privacy**: Add noise to updates
5. **Secure aggregation**: Encrypt model updates
6. **Adaptive learning rates**: Per-client optimization

## References

- [Federated Learning: Collaborative Machine Learning without Centralized Training Data](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## License

This implementation is for educational and research purposes.
