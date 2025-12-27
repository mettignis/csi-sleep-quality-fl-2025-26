# RNN Federated Learning Implementation - Quick Start Guide

## 📋 What Was Created

I've created a complete federated learning system using RNN (LSTM) for sleep quality prediction from time series data:

### Files Created:
1. **`train_rnn_fl.py`** (18KB) - Main training script with federated learning
2. **`inference_rnn.py`** (9.5KB) - Inference script for predictions
3. **`RNN_FL_README.md`** - Comprehensive documentation
4. **`install_rnn_deps.sh`** - Dependency installation script

## 🚀 Getting Started

### Step 1: Install Dependencies

```bash
cd /home/gab/CSI/prova1
./install_rnn_deps.sh
```

This will install:
- TensorFlow 2.x
- NumPy, Pandas, Scikit-learn
- Matplotlib (for visualization)

### Step 2: Train the Model

```bash
# Activate virtual environment (if not already active)
source flvenv/bin/activate

# Run federated training
python3 -m dtbagging.train_rnn_fl
```

**What happens during training:**
- Loads data from 9 clients (ready_for_flwr/client_*.csv)
- Trains LSTM model for 15 rounds
- Each round: 10 epochs per client
- Aggregates models using FedAvg
- Saves model to `experiments/exp_rnn_federated/`

**Expected time:** ~5-15 minutes per round (depends on hardware)

### Step 3: Generate Predictions

```bash
# Automatic (finds latest model)
python3 -m dtbagging.inference_rnn

# Or specify model explicitly
python -m dtbagging.inference_rnn \
    --model experiments/exp_rnn_federated/rnn_federated_model_20251227_* \
    --input test_data/x_test.csv \
    --output submission_rnn.csv
```

## 🏗️ Architecture Overview

### Federated Learning Setup
```
┌─────────────────────────────────────────────────┐
│           Global Server (Aggregator)            │
│  - Maintains global LSTM model                  │
│  - Performs FedAvg aggregation                  │
└─────────────────┬───────────────────────────────┘
                  │
         ┌────────┴────────┐
         ▼                 ▼
    ┌─────────┐      ┌─────────┐
    │Client 0 │ ...  │Client 8 │
    │ (270 MB)│      │ (270 MB)│
    └─────────┘      └─────────┘
    Local data       Local data
    stays private    stays private
```

### LSTM Model Architecture
```
Input: (7 days × 40 features)
    ↓
LSTM Layer 1: 64 units (with dropout 0.2)
    ↓
LSTM Layer 2: 32 units (with dropout 0.2)
    ↓
Dense Layer: 16 units (ReLU)
    ↓
Dropout: 0.3
    ↓
Output: 1 value (sleep quality prediction)
```

## 🔑 Key Features

### 1. Temporal Modeling
- Uses 7-day sequences to capture temporal patterns
- LSTM captures long-term dependencies
- Predicts sleep quality based on recent history

### 2. Privacy-Preserving
- Raw data never leaves clients
- Only model updates are shared
- Per-client normalization

### 3. Federated Averaging (FedAvg)
- Weighted average based on dataset sizes
- Ensures fair contribution from all clients
- Standard FL aggregation method

### 4. Comprehensive Metrics
Tracks per round:
- Training Loss & MAE
- Validation Loss & MAE
- Per-client performance

## 📊 Data Flow

### Training Data
```
clients_data/group{0-8}/     →  ready_for_flwr/client_{0-8}.csv
  ├── 5 CSV files per group      ├── Unified & preprocessed
  └── Raw time series            └── Statistics extracted
```

### Features (40 total)
- **Heart Rate**: max, min, resting, avg, mean, std
- **Respiration**: lowest, highest, avg waking/sleep, mean, std
- **Stress**: max, avg, mean, min, max, std  
- **Sleep**: duration, deep/light/REM phases, wake count
- **Activity**: calories, distance, active time

### Target
- **label**: Sleep quality score (regression)

## ⚙️ Configuration Options

Edit `train_rnn_fl.py` → `main()` function:

```python
trainer = RNNFederatedLearning(
    num_clients=9,           # Number of FL clients
    num_rounds=15,           # FL rounds
    local_epochs=10,         # Epochs per client
    batch_size=16,           # Training batch size
    learning_rate=0.001,     # Adam LR
    sequence_length=7,       # Days of history
    test_fraction=0.2,       # Validation split
    random_seed=42           # Reproducibility
)
```

## 📈 Expected Results

### Performance Metrics
- **MAE**: 5-10 points (Mean Absolute Error)
- **RMSE**: 8-15 points (Root Mean Squared Error)
- **Convergence**: Typically stabilizes after 10-15 rounds

### Training Progress
```
Round 1:  High loss (model learning)
Round 5:  Improving, loss decreasing
Round 10: Near convergence
Round 15: Stable, minimal improvement
```

## 🆚 Comparison with XGBoost

| Aspect | RNN/LSTM (New) | XGBoost (Existing) |
|--------|----------------|-------------------|
| **Temporal modeling** | ✅ Excellent | ❌ Limited |
| **Sequence handling** | ✅ Native | ❌ Manual features |
| **Training time** | ⚠️ Slower | ✅ Fast |
| **Memory usage** | ⚠️ Higher | ✅ Lower |
| **Interpretability** | ❌ Black box | ✅ Feature importance |
| **Small data** | ⚠️ May overfit | ✅ Robust |

**When to use RNN:**
- Strong temporal dependencies
- Sequential patterns important
- Enough data (>1000 samples per client)

**When to use XGBoost:**
- Need fast training
- Feature interactions more important
- Small datasets
- Need interpretability

## 🐛 Troubleshooting

### Issue: TensorFlow not found
```bash
./install_rnn_deps.sh
# Or manually: pip install tensorflow
```

### Issue: Out of memory
Reduce batch size or sequence length:
```python
batch_size=8,        # Instead of 16
sequence_length=5,   # Instead of 7
```

### Issue: Model not converging
- Increase learning rate: `learning_rate=0.01`
- Increase local epochs: `local_epochs=20`
- Check for data quality issues

### Issue: NaN losses
- Check for missing values in data
- Reduce learning rate: `learning_rate=0.0001`
- Add gradient clipping

## 📁 Output Files

After training, you'll find in `experiments/exp_rnn_federated/`:

```
rnn_federated_model_20251227_123456/  ← Trained model (TensorFlow SavedModel)
rnn_federated_weights_20251227_123456.npy  ← Raw weights
training_history_20251227_123456.json  ← Metrics per round
config_20251227_123456.json  ← Configuration used
README.md  ← Experiment summary
```

## 🎯 Next Steps

1. **Train the model**: `python -m dtbagging.train_rnn_fl`
2. **Monitor progress**: Watch console output for metrics
3. **Generate predictions**: `python -m dtbagging.inference_rnn`
4. **Compare results**: Compare with XGBoost baseline
5. **Tune hyperparameters**: Adjust config for better performance

## 📚 Additional Resources

- **Full documentation**: `RNN_FL_README.md`
- **Existing FL implementation**: `train_local.py` (XGBoost version)
- **Data preprocessing**: `setup_dataset.py`
- **Task utilities**: `task.py`

## 💡 Tips for Best Results

1. **Start small**: Try 5 rounds first to verify everything works
2. **Monitor metrics**: Loss should decrease over rounds
3. **Check data balance**: Ensure all clients have similar data sizes
4. **Experiment**: Try different architectures (add layers, change units)
5. **Visualize**: Plot training history to understand convergence

## 🤝 Contributing

Feel free to:
- Experiment with different architectures (GRU, Attention, etc.)
- Try different aggregation strategies
- Add more sophisticated privacy mechanisms
- Implement client selection strategies

---

**Created**: December 27, 2025
**Purpose**: Federated Learning with RNN for Sleep Quality Prediction
**Author**: GitHub Copilot Assistant
