# 🌸 Sleep Quality Federated Learning

A **Federated Learning** project for predicting sleep quality using biometric data distributed across multiple clients.

The system uses **XGBoost** as the machine learning model and **Flower** as the federated learning framework, enabling collaborative training without sharing raw data among participants.

---

## 📋 Requirements

- **Python 3.13+**
- **Operating system**: macOS, Linux, Windows
- **Dependencies**: specified in `pyproject.toml`

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/MettiFire/prova1.git
cd prova1
```

### 2. Create and activate a virtual environment

```bash
python -m venv flvenv
source flvenv/bin/activate  # On macOS/Linux
# or
flvenv\Scripts\activate  # On Windows
```

### 3. Install dependencies

```bash
pip install -e .
```

**Main dependencies:**
- `flwr[simulation]>=1.13.0` – Federated Learning framework
- `xgboost>=2.1.0` - Gradient Boosting
- `pandas`, `numpy`, `scikit-learn` - Preprocessing and ML utilities

**Note for macOS:** If you encounter XGBoost errors related to OpenMP:
```bash
brew install libomp
```

---

## 📊 Data Preparation

### Where to place the datasets

The data is NOT included in the repository. You must download it separately and place it as follows:

```
dtbagging/
└── clients_data/
    ├── group0/
    │   ├── dataset_user_0_train.csv
    │   ├── dataset_user_9_train.csv
    │   └── ...
    ├── group1/
    │   ├── dataset_user_1_train.csv
    │   └── ...
    ├── group2/
    └── ...
    └── group8/
└── test_data/
    └── x_test.cvs/
```

### Preprocessing

Run the preprocessing script to prepare the data for Flower:

```bash
cd dtbagging
python3 setup_dataset.py
```

**What it does:**
- Merges the CSV files for each group
- Processes time-series columns (extracts statistics: mean, min, max, std)
- Handles missing values
- Saves 9 preprocessed files in `ready_for_flwr/`

**Output:** `ready_for_flwr/client_0.csv`, ..., `ready_for_flwr/client_8.csv`

---

## 🏃 Running the Training

### Option 1: Flower Simulation (with Ray)

```bash
cd dtbagging
python3 run_simulation.py
```

**Configuration:** Edit `pyproject.toml` → `[tool.flwr.federations.local-simulation]`

**Note:** On some macOS systems, Ray may have compatibility issues.

### Option 2: Sequential Local Training (without Ray), run outside `dtbagging`:

```bash
python3 -m dtbagging.train_local
```

**Advantages:**
- More stable on macOS
- Does not require Ray
- Sequential training of the 9 clients with manual aggregation

### Training Parameters

- **Clients:** 9 (one per group)
- **Rounds:** 10 (configurable)
- **Test fraction:** 0.2 (20% of data for validation)
- **Model:** XGBoost Regressor

---

## 📁 Project Structure

```
prova1/
├── pyproject.toml           # Project configuration and dependencies
├── README.md                # This file
├── .gitignore               # Git ignored files
│
├── dtbagging/               # Main package
│   ├── __init__.py
│   ├── task.py              # Data loading and utilities
│   ├── client_app.py        # Flower client logic
│   ├── server_app.py        # Flower server logic
│   ├── setup_dataset.py     # Preprocessing script
│   ├── run_simulation.py    # Flower simulation launcher
│   ├── train_local.py       # Alternative local training
│   │
│   ├── clients_data/        # Raw data (NOT in Git)
│   └── ready_for_flwr/      # Preprocessed data (NOT in Git)
│
└── flvenv/                  # Virtual environment (NOT in Git)
```

---

## 📈 Training Output

At the end of training, the following are generated:

* **`final_model.json`** – Final aggregated XGBoost model
* **Training logs** – RMSE/MAE metrics for each round

### Example usage of the model

```python
import xgboost as xgb

# Load the model
model = xgb.Booster()
model.load_model('dtbagging/final_model.json')

# Run predictions
# predictions = model.predict(dmatrix)
```

---

## 📈 Inference on Test set

Once we have the `final_model.json` we can perform Inference on the test data `test_data/x_test.csv`

run outside `dtbagging`:

```bash
python3 dtbagging/inference.py
```

At the end of the Inference, it is generated `submission.csv`, which contains the predictions on the test set.

---

## 🔧 Troubleshooting

### Issue: `"ModuleNotFoundError: No module named 'task'"`

**Solution:** Use relative imports or run as a module:
```bash
python -m dtbagging.train_local
```

### Issue: `"OSError: dlopen: cannot load libomp.dylib"`

**Solution (macOS):**
```bash
brew install libomp
```

### Issue: `"Ray ActorDiedError"`

**Solution:** Use the alternative local training (`train_local.py`)

### Issue: `"All samples are 0"`

**Cause:** Completely empty columns in the CSV files
**Solution:** Already handled in `task.py` with `dropna(axis=1, how='all')`

---

## 🤝 Contributions

To contribute:
1. Fork the repository
2. Create a branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## 📝 License

Project developed for the CSI course – Politecnico di Milano.

---

## 👥 Authors

- Anna Mettifogo

---

## 🔗 Resources

- [Flower Documentation](https://flower.ai/docs/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Federated Learning](https://en.wikipedia.org/wiki/Federated_learning)

---