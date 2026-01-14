# 🌸 Sleep Quality Prediction with Federated Learning 

**Final Score:** 11.79207 MAE  
**Competition Period:** December 2025 - January 9, 2026

---

## 📖 Abstract

This work investigates sleep quality prediction from wearable-device biometric data under strict federated learning constraints, where data privacy prohibits centralized data aggregation. The challenge is characterized by an extremely small dataset (999 training samples distributed across nine federated clients), leading to significant overfitting in baseline models. 

To address this, we propose a federated approach combining **per-client data augmentation** and **heterogeneous ensemble learning**. An ensemble of Random Forest and LightGBM models was trained on augmented data using 40 engineered statistical features extracted from time-series biometrics (heart rate, respiration, stress). 

**Key Finding:** In small federated datasets, mitigating overfitting through data augmentation (6x expansion) and architectural diversity is substantially more effective than extensive hyperparameter optimization.

**Final Result:** Through systematic experimentation (42 experiments, 18 submissions), our meta-ensemble strategy combining augmented training (Exp 039) with multi-algorithm diversity (Exp 031) achieved **11.79207 MAE**, securing 3rd place and demonstrating that data quality and ensemble robustness matter more than model complexity in privacy-constrained settings.

---

## 🎯 Problem Statement & Constraints

### Federated Learning Requirements

The competition enforces a **strict Federated Learning paradigm** with the following constraints:

- ❌ **NO centralized data aggregation**: Raw data must remain private to each client
- ❌ **NO label sharing**: Target labels cannot be aggregated across clients
- ✅ **Model/Prediction sharing**: Only model weights and predictions can be exchanged
- ✅ **Per-client augmentation**: Data synthesis allowed locally without sharing

**Evaluation:** Mean Absolute Error (MAE) computed on a centralized test set of 189 samples.

### Dataset Characteristics

**DataExperimental Journey (42 Experiments)

### Phase 1: Baseline Establishment (exp_001-005)
**Objective:** Evaluate standard federated models and establish performance baselines

**Experiments:**
- **exp_001** (Federated XGBoost): 13.98 MAE - Strongest single-model baseline
- **exp_002** (Random Forest): 15.85 MAE - Underperformed due to insufficient per-client data
- **exp_003** (Enhanced XGBoost): 13.76 MAE - Marginal improvement with conservative hyperparameters
- **exp_004-005** (RNN variants): Failed to converge on small sequential data

**Key Findings:**
- XGBoost provided best baseline with sequential training (shallow trees, low learning rate)
- Early hyperparameter tuning yielded only marginal gains (<2% improvement)
- Train-validation gap indicated severe overfitting (Train MAE: 3-7, Val MAE: 10-14)
- **Root cause identified:** Dataset too small (~110 samples/client)

---

### Phase 2: Feature Engineering Breakthrough (exp_006)
**Objective:** Extract domain-informed features from physiological time series

**Approach:** Transform intraday time-series arrays into compact statistical descriptors:

```python
for col in ['hr_time_series', 'resp_time_series', 'stress_time_series']:
    df[f'{col}_mean'] = df[col].apply(lambda x: np.mean(x))
    df[f'{col}_min'] = df[col].apply(lambda x: np.min(x))
    df[f'{col}_max'] = df[col].apply(lambda x: np.max(x))
    df[f'{col}_std'] = df[col].apply(lambda x: np.std(x))
```

**Advanced features tested:**
- Higher-order statistics: percentiles (25th, 50th, 75th), skewness, kurtosis
- Entropy measures: Shannon entropy, sample entropy
- Temporal trends: linear regression slopes, autocorrelation

**Results:**
- **exp_006**: 12.90 MAE ✅ **-7.7% improvement** (13.98 → 12.90)
- Feature count: 27 → 40 (added 13 time-series statistics)
- RF performance improved most significantly (15.85 → 12.45 MAE)
- Validated that domain knowledge can partially compensate for limited data

---

### Phase 3: Federated Ensemble Optimization (exp_007-030)
**Objective:** Combine heterogeneous models to reduce variance and improve generalization

**Strategy:** Federated bagging with diverse algorithms (RF, LightGBM, XGBoost, CatBoost)

**Ensemble Configurations Tested:**
- **2-way ensembles**: RF+XGBoost, RF+LGBM, XGBoost+LGBM
- **3-way ensembles**: RF+XGBoost+LGBM (best configuration)
- **4-way/5-way ensembles**: Added CatBoost, but degraded performance (noise amplification)

**Weight Optimization:**
- Grid search on ensemble weights (exp_012-030)
- Tested combinations: 50/50, 60/40, 45/30/25, etc.
- Validation-based weight tuning showed marginal gains but high overfitting risk

**Results:**
- **exp_031**: 11.92674 MAE ✅ **-14.7% from baseline** (13.98 → 11.93)
- Formula: `0.45×RF + 0.30×XGB + 0.25×LGBM`
- Confirmed: Ensemble diversity > individual model strength in FL
- **Critical finding:** More than 3 models caused overfitting, not improvement

---

### Phase 4: Data Augmentation Revolution (exp_039)
**Objective:** Address data scarcity through per-client synthetic data generation

**Hypothesis:** Model plateau caused by insufficient training data, not suboptimal architectures

**Augmentation Pipeline (6x expansion):**

1. **Gaussian Noise Injection** (2 variants):
   ```python
   augmented = original + np.random.normal(0, 0.03 * original.std(), original.shape)
   ```
   Simulates natural sensor variability and measurement noise

2. **Magnitude Scaling** (1 variant):
   ```python
   scale_factor = np.random.uniform(0.97, 1.03)
   augmented = original * scale_factor
   ```
   Models inter-individual physiological differences (metabolism, body composition)

3. **Linear Interpolation** (1 variant):
   ```python
   augmented = 0.5 * sample[i] + 0.5 * sample[i+1]
   ```
   Creates intermediate daily states, smoothing transitions

4. **Mixup** (1 variant):
   ```python
   alpha = 0.3
   augmented = alpha * sample[i] + (1-alpha) * sample[j]
   ```
   Regularization through soft blending of consecutive observations

**Data Expansion:**
- Original: ~110 samples/client → Augmented: ~650 samples/client (~6x)
- Total dataset: 999 → ~5,904 samples
- **FL Compliance:** All augmentation performed locally before training (no data sharing)

**Results:**
- **exp_039**: 11.89067 MAE ✅ **-14.9% from baseline**
- RF validation MAE: 12.20 → 8.79 (-28% improvement)
- LGBM validation MAE: 10.88 → 8.58 (-21% improvement)
- Train-validation gap significantly reduced (better generalization)
- Ensemble formula: `0.65×RF_aug + 0.35×LGBM_aug`

**Impact:** Proved that in small federated datasets, data quality/quantity >> model complexity

---

### Phase 5: Meta-Ensemble Optimization (exp_041-042) 🏆

**Objective:** Combine augmentation robustness with architectural diversity

**Experiment 041 - Winning Strategy:**

Instead of refining a single model, Exp 041 introduced a **meta-ensemble** combining two distinct federated methodologies:

```python
# Component 1: Augmented Training (Exp 039)
exp_039_pred = 0.65 × RF_augmented + 0.35 × LGBM_augmented

# Component 2: Architecture Diversity (Exp 031)  
exp_031_pred = 0.45 × RF + 0.30 × XGB + 0.25 × LGBM

# Final Meta-Ensemble
final_pred = 0.5 × exp_039_pred + 0.5 × exp_031_pred
```

**Rationale:**
- **Exp 039 (Augmented):** Provides data-driven robustness, low variance
- **Exp 031 (Diverse):** Provides structural diversity, algorithm heterogeneity
- **Equal weighting (0.5/0.5):** Balances both strengths without overfitting to validation

**Results:**
- **exp_041**: 11.79207 MAE ✅ **BEST SCORE - 3rd Place**
- Improvement: -0.098 from exp_039, -0.134 from exp_031
- Deadline breakthrough: Submitted Jan 9, 6:00 PM (5.5 hours before deadline)

**Experiment 042 - Weight Exploration:**
- Tested non-equal weights: 60/40, 55/45, 45/55
- Local validation suggested marginal gains with 60% augmented bias
- **Decision:** Maintained 50/50 for robustness (validation set too small to trust)
- exp_042 not submitted (risk of overfitting.5%)**
- Formula: `0.45×RF + 0.30×XGB + 0.25×LGBM`
- Strategia: Diversificazione algoritmica per ridurre correlation

### Phase 5: Data Augmentation Revolution (exp_039)
**Insight:** Il problema non era il modello ma la scarsità di dati

- **Augmentation 6x**: da 999 a ~6000 samples (per client)
  - Gaussian noise injection (±3%)
  - Magnitude scaling (97-103%)
  - Linear interpolation
  - Mixup (alpha=0.3)
- **exp_039**: 11.89067 MAE ✅ **Nuovo record (-12.30 MAE)**
- Validation MAE ridotta del 28% (RF) e 21% (LGBM)

### Phase 6: Final Optimization (exp_041) 🏆
**Winning Strategy:** Ensemble di ensemble con pesi ottimizzati

- **exp_041**: 11.79207 MAE ✅ **BEST SCORE - 3rd Place**
- Formula: `0.5×exp_039 + 0.5×exp_031`
- Insight: Combinare data augmentation (exp_039) con diversificazione algoritmica (exp_031)
- Deadline day breakthrough (ore 18:00 del 9 gennaio)

---

## 🛠️ Machine Learning Models & Technologies

### Random Forest
**Role:** High-variance backbone of the federated ensemble

**Architecture:** Federated bagging paradigm with independent forest training per client
- Tree-structured classifiers: `{h(x,θ_k), k=1,...,n}` where θ_k are i.i.d. random vectors
- Each tree casts unit vote, final prediction by majority/average
- **Strength:** Learns complex non-linear relationships without strong bias

**Configuration:**
```python
n_estimators = 200      # 200 trees per forest
max_depth = 15          # Enabled by augmented data (was 8 for original)
min_samples_split = 5   # Conservative to prevent overfitting
random_state = 42
```

**Performance Evolution:**
- Baseline (original data): 12.45 MAE
- With advanced features: 12.20 MAE (-2.0%)
- With augmentation: 8.79 MAE (-28% ✨)

**Key Insight:** Scaled effectively with augmented data (6x samples), supporting deeper trees without overfitting

---

### LightGBM (Light Gradient Boosting Machine)
**Role:** Fast gradient boosting with leaf-wise expansion

**Technical Advantage:** Speeds up training by >20x vs conventional GBDT while maintaining accuracy
- Leaf-wise tree growth (vs level-wise in XGBoost)
- Handles categorical features natively
- Efficient histogram-based learning

**Configuration:**
```python
num_leaves = 31
learning_rate = 0.05
max_depth = 8
n_estimators = 200
early_stopping_rounds = 50
```

**Performance:**
- Original data: 10.88 MAE (high train-val gap → overfitting)
- Augmented data: 8.58 MAE (-21% improvement)
- Best iteration: 87/200 (early stopping prevented overtraining)

**Ensemble Contribution:** Provided non-linear interaction modeling, complementing RF's variance reduction

---

### XGBoost (Extreme Gradient Boosting)
**Role:** High-bias, low-variance stabilizer for ensemble regularization

**Architecture:** Sequential gradient boosting with L1/L2 regularization
- Conservative hyperparameters: shallow trees, low learning rate
- Designed to dampen instability from noisy client data

**Configuration:**
```python
max_depth = 4           # Shallow trees (high bias)
learning_rate = 0.03    # Slow convergence
n_estimators = 150
reg_alpha = 0.1         # L1 regularization
reg_lambda = 1.0        # L2 regularization
```

**Performance:**
- Single-model baseline: 13.98 MAE (strongest initial performer)
- In ensemble (exp_031): 30% weight (bias-variance balance)

**Strategic Role:** Offset variance-heavy RF and LGBM, prevented ensemble from chasing noise

**Note:** Excluded from final augmentation experiments (exp_039) due to high training cost relative to LGBM

---

### CatBoost
**Status:** Tested but excluded from final ensemble

**Rationale for Testing:** Known for robust small-dataset performance and efficient categorical handling

**Results:**
- **exp_034** (4-way ensemble with CatBoost): 12.00 MAE
- Higher validation error than RF/LGBM
- Increased ensemble noise without diversity benefit

**Decision:** Excluded to maintain ensemble stability (3 models optimal)

---

### LSTM (Long Short-Term Memory) Networks
**Status:** Explored but not used in final solution

**Architecture Tested:**
- Bidirectional LSTM layers (forward/backward sequence processing)
- Unidirectional LSTM layers
- Dropout + Batch Normalization for regularization
- Federated Averaging for weight aggregation

**Hypothesis:** Historical day sequences contain predictive signals for future sleep quality

**Challenges:**
- Extremely short sequences (~27 days/user)
- High variance across users (different sleep patterns)
- Computational cost vs marginal gains

**Outcome:** Underperformed tree-based methods on this small, heterogeneous dataset

---

### Federated Learning Framework
**Technology:** Flower (flwr) v1.13.0+

**Orchestration:**
- **FedAvg (Federated Averaging):** Weighted average of client predictions
- **Client-side training:** Each of 9 clients trains independently on local data
- **Server aggregation:** Combines predictions only (no raw data/labels)

**FL Compliance Strategy:**
- ✅ Per-client augmentation (local synthetic data generation)
- ✅ Ensemble of predictions (aggregation at prediction level)
- ✅ Decentralized validation (each client validates locally)
- ❌ Never aggregated: raw features, labels, centralized hyperparameter tuning

**Key Advantage:** Maintained strict privacy while achieving competitive performance

---

## 📊 Key Results & Learnings

### Competition Statistics
- **Total Experiments:** 42
- **Submissions Used:** 18/35
- **Best Score:** 11.79207 MAE (exp_041)
- **Final Rank:** 3rd place 🥉
- **Improvement from baseline:** -15.6% (-2.19 MAE)

### Technical Innovations
1. **Time series feature engineering**: +13 features statistiche
2. **6x data augmentation**: Risoluzione del problema di scarsity
3. **Hierarchical ensemble**: Ensemble di ensemble per massimizzare diversità
4. **FL-compliant augmentation**: Augmentation per-client senza data sharing

### Critical Turning Points
1. **exp_006**: Feature engineering (+4 nuove features time series)
2. **exp_031**: Three-way ensemble eterogeneo
3. **exp_039**: Data augmentation 6x
4. **exp_041**: Ensemble optimization deadline day

### What Worked
✅ Data augmentation per ridurre overfitting  
✅ Ensemble eterogeneo (RF+LGBM+XGB)  
✅ Feature engineering su time series  
✅ Weight optimization basato su validation MAE  
✅ Equal weighting (50/50) per ensemble finali

### What Didn't Work
❌ CatBoost (troppo lento, performance insufficienti)  
❌ Complex weight optimization (overfitting, 50/50 spesso migliore)  
❌ 4-way/5-way ensemble (troppo correlati, no improvement)  
❌ Meta-learning without augmentation (specialization problem)

---

## 📋 Requirements

- **Python 3.13+**
- **Key Dependencies**: `flwr`, `xgboost`, `lightgbm`, `scikit-learn`, `pandas`, `numpy`
- **Hardware**: Standard CPU (no GPU required)

---

## �️ Repository Structure

```
prova1/
├── README.md                    # Project overview e competition journey
├── pyproject.toml               # Dependencies e configurazione
│
├── dtbagging/
│   ├── PROGRESS_REPORT.md       # Detailed experiment log (42 experiments)
│   │
│   ├── setup_dataset.py         # Data preprocessing pipeline
│   ├── setup_dataset_augmented_enhanced.py  # 6x augmentation pipeline
│   ├── feature_engineering.py   # Time series feature extraction
│   ├── advanced_features.py     # Advanced feature engineering
│   │
│   ├── client_app.py            # Flower client logic
│   ├── server_app.py            # Flower server logic
│   ├── run_simulation.py        # FL simulation launcher
│   ├── task.py                  # Data loading utilities
│   │
│   ├── experiments/
│   │   ├── exp_001_xgboost_baseline/
│   │   ├── exp_006_feature_engineering/    # Feature engineering breakthrough
│   │   ├── exp_031_three_way_federated/    # Three-way ensemble (11.93 MAE)
│   │   ├── exp_039_augmented_ensemble/     # Data augmentation (11.89 MAE)
│   │   ├── exp_041_ensemble/               # 🏆 WINNER (11.79 MAE)
│   │   └── ... (38 other experiments)
│   │
│   ├── clients_data/            # Raw data (9 groups, 45 users)
│   ├── ready_for_flwr/          # Preprocessed data (9 clients)
│   └── ready_for_flwr_augmented/  # Augmented data (6x samples)
│
└── flvenv/                      # Virtual environment
```

**Key Files:**
- **[PROGRESS_REPORT.md](dtbagging/PROGRESS_REPORT.md)**: Detailed log of all 42 experiments with metrics, reasoning, and lessons learned
- **exp_041**: Final winning solution code and submission
- **exp_039**: Data augmentation implementation (6x samples)
- **exp_031**: Three-way federated ensemble baseline

---

## 🔧 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/MettiFire/prova1.git
cd prova1

# Create virtual environment
python -m venv flvenv
source flvenv/bin/activate  # macOS/Linux
# flvenv\Scripts\activate   # Windows

# Install dependencies
pip install -e .
```

### Data Setup

Place training data in `dtbagging/clients_data/` with structure:
```
clients_data/
├── group0/ (dataset_user_0_train.csv, dataset_user_9_train.csv, ...)
├── group1/ (dataset_user_1_train.csv, dataset_user_10_train.csv, ...)
└── ... (group2-group8)
```

### Preprocessing

```bash
cd dtbagging

# Standard preprocessing
python setup_dataset.py

# With 6x augmentation (recommended)
python setup_dataset_augmented_enhanced.py
```

### Reproduce Winning Solution (exp_041)

```bash
cd dtbagging/experiments/exp_041_ensemble

# Train exp_039 (augmented ensemble)
cd ../exp_039_augmented_ensemble
python train_rf_augmented.py
python train_lgbm_augmented.py

# Train exp_031 (three-way federated)
cd ../exp_031_three_way_federated
python train_rf_lgbm_xgb_weighted.py

# Generate exp_041 submission
cd ../exp_041_ensemble
python optimize_weights.py
python generate_submission.py
```

**Expected Result:** submission_exp041.csv with ~11.79 MAE on Kaggle

---

## 📁 Struttura del Progetto

```
prova1/
├── pyproject.toml           # Configurazione progetto e dipendenze
├── README.md                # Questo file
├── .gitignore               # File da ignorare in Git
│
├── � Key Insights & Design Decisions

### Why Data Augmentation Was Critical
**Problem:** 999 samples / 9 clients = ~110 samples/client → severe overfitting  
**Solution:** 6x augmentation → ~650 samples/client  
**Impact:** Validation MAE dropped from 12.20 to 8.79 (RF), from 10.88 to 8.58 (LGBM)

### Why Ensemble of Ensembles Works
**Insight:** Different ensemble strategies capture different aspects of data
- **exp_039**: Augmentation-based robustness (low variance)
- **exp_031**: Algorithm diversity (RF+LGBM+XGB)
- **exp_041 = 0.5×exp_039 + 0.5×exp_031**: Best of both worlds

### FL Compliance Strategy
- ✅ **Augmentation per-client**: No data sharing needed
- ✅ **Ensemble of predictions**: Server aggregates predictions only, no labels
- ✅ **Local training**: Each client trains on own augmented data
- ❌ **Avoided**: Centralized training, label sharing, meta-learning with labels

### Weight Optimization Philosophy
**Finding:** Simple equal weighting (50/50) often outperforms complex optimization
- exp_041: 50/50 weights → 11.79 MAE ✅
- exp_042: 60/40 weights → Not better, not submitted
- **Lesson:** Overfitting to validation set is real, keep it simple

---

## 📈 Performance Metrics

### Experiment Evolution
| Experiment | MAE | Strategy | Improvement |
|------------|-----|----------|-------------|
| exp_001 | 13.98 | XGBoost baseline | - |
| exp_006 | 12.90 | Feature engineering | -7.7% |
| exp_031 | 11.93 | Three-way ensemble | -14.7% |
| exp_039 | 11.89 | Data augmentation | -14.9% |
| **exp_041** | **11.79** | **Ensemble optimization** | **-15.6%** 🏆 |

### Model Contributions (exp_041)
- **50% exp_039**: RF (65%) + LGBM (35%) on 6x augmented data
- **50% exp_031**: RF (45%) + XGB (30%) + LGBM (25%) on original data
- **Final blend**: Optimal balance between robustness and diversity

---

## 🔍 Technical Deep Dive

### Feature Engineering Pipeline
```python
# Time series → Statistical features
for col in ['hr_time_series', 'resp_time_series', 'stress_time_series']:
    df[f'{col}_mean'] = df[col].apply(lambda x: np.mean(x))
    df[f'{col}_min'] = df[col].apply(lambda x: np.min(x))
    df[f'{col}_max'] = df[col].apply(lambda x: np.max(x))
    df[f'{col}_std'] = df[col].apply(lambda x: np.std(x))
```
**Impact:** +13 features, +7.7% improvement (exp_006)

### Data Augmentation Strategy
```python
# 6x augmentation mix
augmented_data = [
    original_data,                          # 1x
    gaussian_noise(data, std=0.03),        # 2x (two variants)
    magnitude_scale(data, range=0.97-1.03), # 1x
    linear_interpolation(data, pairs),      # 1x
    mixup(data, alpha=0.3)                  # 1x
]
```
**Impact:** -28% validation MAE (RF), -21% (LGBM)

### Ensemble Architecture (exp_041)
```python
# Hierarchical ensemble
exp📚 Documentation

- **[PROGRESS_REPORT.md](dtbagging/PROGRESS_REPORT.md)**: Complete experiment log with detailed analysis of all 42 experiments
- **Experiment READMEs**: Each experiment folder contains specific implementation details
- **Code comments**: Inline documentation explaining design decisions

---

## 🎯 Future Work & Open Questions

### Potential Improvements
1. **Advanced augmentation**: GAN-based synthetic data generation
2. **Neural networks**: Test deep learning with augmented data
3. **Federated hyperparameter optimization**: Coordinate HP tuning across clients
4. **Dynamic ensemble weighting**: Adaptive weights based on client data distribution

### Open Research Questions
1. Can meta-learning work in true FL without label sharing?
2. How to handle client heterogeneity better (some clients have better data)?
3. What's the theoretical limit of FL vs centralized for this dataset?

---

## 👥 Author

**Anna Mettifogo**  
Politecnico di Milano - CSI (Cyber Security and Intelligence)

---

## 🏆 Competition Timeline

- **Start:** December 2025
- **Phase 1 (Baseline):** exp_001-005 (Dec 28-29)
- **Phase 2 (Feature Engineering):** exp_006 (Dec 30) ✨
- **Phase 3 (Ensemble Experiments):** exp_007-030 (Dec 31 - Jan 5)
- **Phase 4 (Three-Way Federated):** exp_031 (Jan 6) ✨
- **Phase 5 (Augmentation):** exp_039 (Jan 6) ✨
- **Phase 6 (Final Optimization):** exp_041 (Jan 9, 18:00) 🏆
- **Deadline:** January 9, 2026 (23:59)
- **Result:** 3rd Place 🥉

---

## 🔗 Resources

- [Flower Framework](https://flower.ai/docs/) - Federated Learning orchestration
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient Boosting
- [LightGBM](https://lightgbm.readthedocs.io/) - Fast gradient boosting
- [Scikit-learn](https://scikit-learn.org/) - Random Forest & utilities
- [Data Augmentation for ML](https://arxiv.org/abs/1904.12848) - Augmentation techniques
- [Federated Learning Survey](https://arxiv.org/abs/1912.04977) - FL overview

---

## 📄 License

Academic project for Politecnico di Milano - CSI Course

---

## 🙏 Acknowledgments

Grazie al corso CSI del Politecnico di Milano per l'opportunità di partecipare a questa competizione e approfondire il Federated Learning in un contesto pratico e competitivo.

**Competition Stats:**
- 42 experiments documented
- 18 Kaggle submissions (out of 35 allowed)
- 5 breakthrough moments
- 1 week of intense iteration
- 🥉 3rd place finish

**Key Takeaway:** In machine learning competitions, data quality and quantity often matter more than model complexity. Our winning strategy prioritized data augmentation (6x samples) and ensemble diversity over hyperparameter tuning.

---

## 🛠️ Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'task'"**
```bash
python -m dtbagging.script_name  # Use module syntax
```

**"OSError: dlopen: cannot load libomp.dylib" (macOS)**
```bash
brew install libomp
```

**"Ray ActorDiedError" (Flower simulation)**
```bash
# Use sequential training instead
cd dtbagging/experiments/exp_XXX
python train_script.py
```

**"All samples are 0"**
- Already handled in preprocessing (`dropna(axis=1, how='all')`)
- Check data format: time series should be comma-separated strings
3. Commit delle modifiche (`git commit -m 'Aggiunge nuova funzionalità'`)
4. Push al branch (`git push origin feature/nuova-funzionalita`)
5. Apri una Pull Request

---

## 📝 Licenza

Progetto sviluppato per il corso CSI - Politecnico di Milano.

---

## 👥 Autori

- Anna Mettifogo

---

## 🔗 Risorse

- [Flower Documentation](https://flower.ai/docs/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Federated Learning](https://en.wikipedia.org/wiki/Federated_learning)
