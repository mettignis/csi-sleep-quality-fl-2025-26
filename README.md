# 🌸 Sleep Quality Federated Learning

Progetto di **Federated Learning** per la predizione della qualità del sonno utilizzando dati biometrici distribuiti tra più client.

Il sistema utilizza **XGBoost** come modello di machine learning e **Flower** come framework di federazione, permettendo il training collaborativo senza condividere i dati grezzi tra i partecipanti.

---

## 📋 Requisiti

- **Python 3.13+**
- **Sistema operativo**: macOS, Linux, Windows
- **Dipendenze**: specificate in `pyproject.toml`

---

## 🚀 Installazione

### 1. Clona la repository

```bash
git clone https://github.com/MettiFire/prova1.git
cd prova1
```

### 2. Crea e attiva un ambiente virtuale

```bash
python -m venv flvenv
source flvenv/bin/activate  # Su macOS/Linux
# oppure
flvenv\Scripts\activate  # Su Windows
```

### 3. Installa le dipendenze

```bash
pip install -e .
```

**Dipendenze principali:**
- `flwr[simulation]>=1.13.0` - Framework Federated Learning
- `xgboost>=2.1.0` - Gradient Boosting
- `pandas`, `numpy`, `scikit-learn` - Preprocessing e ML utilities

**Nota per macOS:** Se riscontri errori con XGBoost relativi a OpenMP:
```bash
brew install libomp
```

---

## 📊 Preparazione dei Dati

### Dove posizionare i dataset

I dati NON sono inclusi nella repository. Devi scaricarli separatamente e posizionarli come segue:

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
```

**Struttura prevista:**
- 9 gruppi (`group0` - `group8`)
- Ogni gruppo contiene 5 file CSV con dati di utenti diversi
- Totale: 45 utenti distribuiti

### Preprocessing

Esegui lo script di preprocessing per preparare i dati per Flower:

```bash
cd dtbagging
python setup_dataset.py
```

**Cosa fa:**
- Unisce i file CSV di ogni gruppo
- Processa le colonne time-series (estrae statistiche: mean, min, max, std)
- Gestisce valori mancanti
- Salva 9 file preprocessati in `ready_for_flwr/`

**Output:** `ready_for_flwr/client_0.csv`, ..., `ready_for_flwr/client_8.csv`

---

## 🏃 Esecuzione del Training

### Opzione 1: Simulazione Flower (con Ray)

```bash
cd dtbagging
python run_simulation.py
```

**Configurazione:** Modifica `pyproject.toml` → `[tool.flwr.federations.local-simulation]`

**Nota:** Su alcuni sistemi macOS, Ray potrebbe avere problemi di compatibilità.

### Opzione 2: Training Locale Sequenziale (senza Ray)

```bash
cd dtbagging
python -m dtbagging.train_local
```

**Vantaggi:**
- Più stabile su macOS
- Non richiede Ray
- Training sequenziale dei 9 client con aggregazione manuale

### Parametri di Training

- **Client:** 9 (uno per gruppo)
- **Round:** 10 (configurabile)
- **Test fraction:** 0.2 (20% dati per validazione)
- **Modello:** XGBoost Regressor

---

## 📁 Struttura del Progetto

```
prova1/
├── pyproject.toml           # Configurazione progetto e dipendenze
├── README.md                # Questo file
├── .gitignore               # File da ignorare in Git
│
├── dtbagging/               # Package principale
│   ├── __init__.py
│   ├── task.py              # Caricamento dati e utilità
│   ├── client_app.py        # Logica client Flower
│   ├── server_app.py        # Logica server Flower
│   ├── setup_dataset.py     # Script preprocessing
│   ├── run_simulation.py    # Launcher simulazione Flower
│   ├── train_local.py       # Training locale alternativo
│   │
│   ├── clients_data/        # Dati grezzi (NON in Git)
│   └── ready_for_flwr/      # Dati preprocessati (NON in Git)
│
└── flvenv/                  # Virtual environment (NON in Git)
```

---

## 📈 Output del Training

Al termine del training, viene generato:

- **`final_model.json`** - Modello XGBoost aggregato finale
- **Log di training** - Metriche RMSE/MAE per ogni round

### Esempio di utilizzo del modello

```python
import xgboost as xgb

# Carica il modello
model = xgb.Booster()
model.load_model('dtbagging/final_model.json')

# Esegui predizioni
# predictions = model.predict(dmatrix)
```

---

## 🔧 Troubleshooting

### Problema: "ModuleNotFoundError: No module named 'task'"

**Soluzione:** Usa import relativi o esegui come modulo:
```bash
python -m dtbagging.train_local
```

### Problema: "OSError: dlopen: cannot load libomp.dylib"

**Soluzione (macOS):**
```bash
brew install libomp
```

### Problema: "Ray ActorDiedError"

**Soluzione:** Usa il training locale alternativo (`train_local.py`)

### Problema: "All samples are 0"

**Causa:** Colonne completamente vuote nei CSV
**Soluzione:** Già gestito in `task.py` con `dropna(axis=1, how='all')`

---

## 🤝 Contributi

Per contribuire:
1. Fork della repository
2. Crea un branch (`git checkout -b feature/nuova-funzionalita`)
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
