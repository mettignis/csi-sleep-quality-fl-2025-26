"""Esperimento 013: Stacking con Meta-Learner"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
import pickle


def create_stacking_model(train_data_path: str, exp_dir: Path):
    """
    Crea modello stacking con cross-validation.
    
    Steps:
    1. Load training data
    2. Generate out-of-fold predictions da RF e XGBoost
    3. Train meta-learner (Ridge) su queste predictions
    4. Save tutto
    """
    print("="*70)
    print("🏗️  STACKING META-LEARNER - ESPERIMENTO 013")
    print("="*70)
    print("\n📊 Strategia:")
    print("   1. Base models: RF (exp_008 config) + XGBoost (exp_001 config)")
    print("   2. Cross-validation: 5-fold per out-of-fold predictions")
    print("   3. Meta-learner: Ridge regression (impara pesi ottimali)")
    print("   4. Previene overfitting tramite OOF predictions")
    print()
    
    # Load training data
    print("📂 Caricamento dati training...")
    all_train_data = []
    
    for client_id in range(9):
        data_path = Path(train_data_path) / "clients_data" / f"group{client_id}"
        csv_files = sorted(data_path.glob("*.csv"))
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, sep=';')
            all_train_data.append(df)
    
    train_df = pd.concat(all_train_data, ignore_index=True)
    print(f"   ✓ Caricati {len(train_df)} campioni")
    
    # Preprocessing time-series
    print("   🔧 Preprocessing time-series...")
    timeseries_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
    
    for col in timeseries_cols:
        if col not in train_df.columns:
            continue
            
        prefix = col.replace('_time_series', '')
        
        def parse_series(x):
            if pd.isna(x):
                return []
            try:
                s = str(x).strip('[]')
                values = [float(v) for v in s.split(',') if v.strip() and float(v.strip()) >= 0]
                return values
            except:
                return []
        
        series = train_df[col].apply(parse_series)
        
        train_df[f'{prefix}_mean'] = series.apply(lambda x: np.mean(x) if len(x) > 0 else np.nan)
        train_df[f'{prefix}_min'] = series.apply(lambda x: np.min(x) if len(x) > 0 else np.nan)
        train_df[f'{prefix}_max'] = series.apply(lambda x: np.max(x) if len(x) > 0 else np.nan)
        train_df[f'{prefix}_std'] = series.apply(lambda x: np.std(x) if len(x) > 0 else np.nan)
        
        train_df.drop(columns=[col], inplace=True)
    
    # Separate features and labels
    if 'label' not in train_df.columns:
        raise ValueError("Label column not found!")
    
    X = train_df.drop(columns=['label'])
    y = train_df['label'].values
    
    print(f"   ✓ Features: {X.shape[1]}, Samples: {len(X)}")
    
    # 5-Fold Cross-Validation per OOF predictions
    print("\n🔄 Generazione Out-Of-Fold predictions (5-fold CV)...")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_rf = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    
    rf_models = []
    xgb_models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n   Fold {fold}/5:")
        
        X_train_fold = X.iloc[train_idx].values
        y_train_fold = y[train_idx]
        X_val_fold = X.iloc[val_idx].values
        
        # Train Random Forest (exp_008 config)
        print(f"      Training RF...")
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42 + fold,
            n_jobs=-1
        )
        rf.fit(X_train_fold, y_train_fold)
        oof_rf[val_idx] = rf.predict(X_val_fold)
        rf_models.append(rf)
        
        # Train XGBoost (exp_001 config)
        print(f"      Training XGBoost...")
        dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dval = xgb.DMatrix(X_val_fold)
        
        params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "eta": 0.3,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "rmse"
        }
        
        xgb_model = xgb.train(params, dtrain, num_boost_round=100)
        oof_xgb[val_idx] = xgb_model.predict(dval)
        xgb_models.append(xgb_model)
        
        # Performance su validation
        from sklearn.metrics import mean_squared_error
        rf_rmse = np.sqrt(mean_squared_error(y[val_idx], oof_rf[val_idx]))
        xgb_rmse = np.sqrt(mean_squared_error(y[val_idx], oof_xgb[val_idx]))
        print(f"      RF RMSE: {rf_rmse:.2f}, XGB RMSE: {xgb_rmse:.2f}")
    
    # Train meta-learner
    print("\n🧠 Training meta-learner (Ridge regression)...")
    
    # Stack predictions come features
    meta_features = np.column_stack([oof_rf, oof_xgb])
    
    # Ridge regression (con regolarizzazione)
    meta_learner = Ridge(alpha=1.0)
    meta_learner.fit(meta_features, y)
    
    # Performance meta-learner
    meta_pred = meta_learner.predict(meta_features)
    meta_rmse = np.sqrt(mean_squared_error(y, meta_pred))
    
    print(f"   ✓ Meta-learner trained!")
    print(f"   OOF RMSE - RF: {np.sqrt(mean_squared_error(y, oof_rf)):.2f}")
    print(f"   OOF RMSE - XGB: {np.sqrt(mean_squared_error(y, oof_xgb)):.2f}")
    print(f"   Meta RMSE: {meta_rmse:.2f}")
    
    print(f"\n📊 Pesi appresi dal meta-learner:")
    print(f"   RF weight: {meta_learner.coef_[0]:.4f}")
    print(f"   XGB weight: {meta_learner.coef_[1]:.4f}")
    print(f"   Intercept: {meta_learner.intercept_:.4f}")
    
    # Save tutto
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    stacking_model = {
        'rf_models': rf_models,
        'xgb_models': xgb_models,
        'meta_learner': meta_learner,
        'feature_names': X.columns.tolist()
    }
    
    model_path = exp_dir / "stacking_model_exp013.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(stacking_model, f)
    
    print(f"\n✅ Stacking model salvato: {model_path}")
    print("="*70)
    
    return stacking_model


if __name__ == "__main__":
    train_data_path = Path(__file__).parent
    exp_dir = Path(__file__).parent / "experiments" / "exp_013_stacking"
    
    create_stacking_model(str(train_data_path), exp_dir)
