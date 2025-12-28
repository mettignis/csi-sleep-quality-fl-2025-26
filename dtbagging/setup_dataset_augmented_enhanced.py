"""
Script per preparare i dati per Federated Learning con Data Augmentation Avanzata.

Questo script applica multiple tecniche di augmentation:
1. Interpolazione lineare tra elementi consecutivi (2 nuove righe)
2. Gaussian noise injection
3. Magnitude scaling
4. Mixup tra campioni vicini

Questo crea un dataset più robusto e diversificato per il training.
"""

import os
import glob
import ast
import pandas as pd
import numpy as np
from pathlib import Path


def add_gaussian_noise(row, noise_level=0.02):
    """
    Aggiunge rumore gaussiano a una riga.
    
    Args:
        row: pandas Series
        noise_level: Deviazione standard del rumore (relativa ai valori)
        
    Returns:
        Nuova riga con rumore aggiunto
    """
    new_row = row.copy()
    
    for col in row.index:
        val = row[col]
        
        # Solo per valori numerici
        if pd.isna(val) or isinstance(val, str):
            continue
        
        if col in ['day']:
            # Non aggiungere rumore alla colonna day
            continue
        
        try:
            # Aggiungi rumore gaussiano proporzionale al valore
            noise = np.random.normal(0, abs(val) * noise_level + 1e-6)
            new_row[col] = val + noise
        except:
            pass
    
    return new_row


def scale_magnitude(row, scale_factor=None):
    """
    Scala i valori di una riga per un fattore casuale.
    
    Args:
        row: pandas Series
        scale_factor: Fattore di scala (se None, usa random tra 0.95-1.05)
        
    Returns:
        Nuova riga scalata
    """
    if scale_factor is None:
        scale_factor = np.random.uniform(0.97, 1.03)
    
    new_row = row.copy()
    
    for col in row.index:
        val = row[col]
        
        if pd.isna(val) or isinstance(val, str) or col in ['day']:
            continue
        
        try:
            new_row[col] = val * scale_factor
        except:
            pass
    
    return new_row


def mixup_rows(row1, row2, alpha=0.3):
    """
    Crea una combinazione convessa di due righe (mixup augmentation).
    
    Args:
        row1: Prima riga
        row2: Seconda riga
        alpha: Peso per mixup (0.5 = media, < 0.5 più simile a row1)
        
    Returns:
        Nuova riga mixup
    """
    new_row = {}
    
    for col in row1.index:
        val1 = row1[col]
        val2 = row2[col]
        
        if pd.isna(val1) or pd.isna(val2):
            new_row[col] = val1 if not pd.isna(val1) else val2
            continue
        
        if isinstance(val1, str) or isinstance(val2, str):
            new_row[col] = val1
            continue
        
        if col in ['day']:
            new_row[col] = int(np.round(val1 * (1 - alpha) + val2 * alpha))
            continue
        
        try:
            new_row[col] = val1 * (1 - alpha) + val2 * alpha
        except:
            new_row[col] = val1
    
    return pd.Series(new_row)


def linear_interpolate_rows(row1, row2, num_steps=2):
    """
    Crea righe interpolate linearmente tra due righe.
    
    Args:
        row1: Prima riga (pandas Series)
        row2: Seconda riga (pandas Series)
        num_steps: Numero di righe da interpolare
        
    Returns:
        Lista di Series con le righe interpolate
    """
    interpolated_rows = []
    
    for step in range(1, num_steps + 1):
        alpha = step / (num_steps + 1)
        
        new_row = {}
        
        for col in row1.index:
            val1 = row1[col]
            val2 = row2[col]
            
            if pd.isna(val1) or pd.isna(val2):
                new_row[col] = np.nan
                continue
            
            if isinstance(val1, str) or isinstance(val2, str):
                new_row[col] = val1
            elif col in ['day']:
                new_row[col] = int(np.round(val1 * (1 - alpha) + val2 * alpha))
            elif col.endswith('_time_series'):
                try:
                    if isinstance(val1, str):
                        arr1 = ast.literal_eval(val1)
                    else:
                        arr1 = val1 if isinstance(val1, list) else [val1]
                    
                    if isinstance(val2, str):
                        arr2 = ast.literal_eval(val2)
                    else:
                        arr2 = val2 if isinstance(val2, list) else [val2]
                    
                    min_len = min(len(arr1), len(arr2))
                    if min_len > 0:
                        interpolated = [
                            arr1[i] * (1 - alpha) + arr2[i] * alpha
                            for i in range(min_len)
                        ]
                        new_row[col] = str(interpolated)
                    else:
                        new_row[col] = val1
                except:
                    new_row[col] = val1
            else:
                try:
                    new_row[col] = val1 * (1 - alpha) + val2 * alpha
                except:
                    new_row[col] = val1
        
        interpolated_rows.append(pd.Series(new_row))
    
    return interpolated_rows


def augment_dataframe(df, augmentation_factor=3):
    """
    Applica multiple tecniche di augmentation a un DataFrame.
    
    Args:
        df: DataFrame originale
        augmentation_factor: Quante volte moltiplicare i dati
        
    Returns:
        DataFrame augmented
    """
    all_rows = []
    
    # Aggiungi tutti i dati originali
    for _, row in df.iterrows():
        all_rows.append(row)
    
    n_original = len(df)
    print(f"      Applicando augmentation (factor={augmentation_factor})...")
    
    # Per ogni riga originale, crea varianti augmented
    for idx, row in df.iterrows():
        # Tecnica 1: Gaussian noise (2 varianti)
        for _ in range(min(2, augmentation_factor)):
            noisy_row = add_gaussian_noise(row, noise_level=0.03)
            all_rows.append(noisy_row)
        
        # Tecnica 2: Magnitude scaling (1 variante)
        if augmentation_factor >= 2:
            scaled_row = scale_magnitude(row)
            all_rows.append(scaled_row)
        
        # Tecnica 3: Interpolazione con riga successiva
        if idx < len(df) - 1 and augmentation_factor >= 3:
            next_row = df.iloc[idx + 1]
            interp_rows = linear_interpolate_rows(row, next_row, num_steps=2)
            all_rows.extend(interp_rows)
        
        # Tecnica 4: Mixup con riga vicina
        if idx < len(df) - 1 and augmentation_factor >= 4:
            next_row = df.iloc[idx + 1]
            mixed_row = mixup_rows(row, next_row, alpha=0.3)
            all_rows.append(mixed_row)
    
    augmented_df = pd.DataFrame(all_rows)
    augmented_df = augmented_df.reset_index(drop=True)
    
    print(f"      → Da {n_original} a {len(augmented_df)} righe (x{len(augmented_df)/n_original:.1f})")
    
    return augmented_df


def merge_users_with_augmentation(csv_files, augmentation_factor=3):
    """
    Unisce i CSV di diversi utenti applicando augmentation.
    
    Args:
        csv_files: Lista di percorsi ai file CSV
        augmentation_factor: Fattore di moltiplicazione per augmentation
        
    Returns:
        DataFrame unito con data augmentation
    """
    all_user_dfs = []
    
    for idx, csv_file in enumerate(sorted(csv_files)):
        try:
            # Leggi il CSV dell'utente corrente
            df = None
            
            try:
                df = pd.read_csv(csv_file, sep=';', engine='python', on_bad_lines='skip')
                if df.shape[1] == 1:
                    df = pd.read_csv(csv_file, sep=',', engine='python', on_bad_lines='skip')
            except:
                pass
            
            if df is None or len(df) == 0:
                try:
                    df = pd.read_csv(csv_file, sep=';', on_bad_lines='warn', encoding='utf-8')
                except:
                    pass
            
            if df is None or len(df) == 0:
                try:
                    df = pd.read_csv(csv_file, sep=';', quoting=3, engine='python')
                except:
                    pass
            
            if df is None or len(df) == 0:
                print(f"   ⚠️  Impossibile caricare: {os.path.basename(csv_file)}")
                continue
            
            print(f"   ✓ Caricato: {os.path.basename(csv_file)} ({len(df)} righe)")
            
            # Applica augmentation a questo utente
            df_augmented = augment_dataframe(df, augmentation_factor)
            all_user_dfs.append(df_augmented)
                
        except Exception as e:
            print(f"   ✗ Errore nel caricamento di {csv_file}: {e}")
            continue
    
    if not all_user_dfs:
        return None
    
    # Concatena tutti gli utenti
    merged_df = pd.concat(all_user_dfs, ignore_index=True)
    
    return merged_df


def preprocess_timeseries_columns(df):
    """
    Preprocessa le colonne time-series convertendole da stringhe a statistiche numeriche.
    """
    timeseries_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
    
    for col in timeseries_cols:
        if col not in df.columns:
            continue
            
        print(f"   Preprocessing colonna: {col}")
        
        def safe_parse(x):
            if pd.isna(x) or x is None:
                return []
            if isinstance(x, (list, np.ndarray)):
                return list(x)
            if isinstance(x, str):
                try:
                    parsed = ast.literal_eval(x.strip())
                    if isinstance(parsed, (list, tuple)):
                        return list(parsed)
                    return []
                except:
                    return []
            return []
        
        series_data = df[col].apply(safe_parse)
        
        base_name = col.replace('_time_series', '')
        
        def safe_mean(x):
            if isinstance(x, list) and len(x) > 0:
                try:
                    valid_vals = [v for v in x if v is not None and not pd.isna(v)]
                    return np.mean(valid_vals) if len(valid_vals) > 0 else np.nan
                except:
                    return np.nan
            return np.nan
        
        def safe_min(x):
            if isinstance(x, list) and len(x) > 0:
                try:
                    valid_vals = [v for v in x if v is not None and not pd.isna(v)]
                    return np.min(valid_vals) if len(valid_vals) > 0 else np.nan
                except:
                    return np.nan
            return np.nan
        
        def safe_max(x):
            if isinstance(x, list) and len(x) > 0:
                try:
                    valid_vals = [v for v in x if v is not None and not pd.isna(v)]
                    return np.max(valid_vals) if len(valid_vals) > 0 else np.nan
                except:
                    return np.nan
            return np.nan
        
        def safe_std(x):
            if isinstance(x, list) and len(x) > 0:
                try:
                    valid_vals = [v for v in x if v is not None and not pd.isna(v)]
                    return np.std(valid_vals) if len(valid_vals) > 0 else np.nan
                except:
                    return np.nan
            return np.nan
        
        df[f'{base_name}_mean'] = series_data.apply(safe_mean)
        df[f'{base_name}_min'] = series_data.apply(safe_min)
        df[f'{base_name}_max'] = series_data.apply(safe_max)
        df[f'{base_name}_std'] = series_data.apply(safe_std)
        
        df = df.drop(columns=[col])
    
    return df


def merge_and_preprocess_group(group_path, group_name, augmentation_factor=3):
    """
    Unisce tutti i CSV di un gruppo con data augmentation e li preprocessa.
    """
    print(f"\n📂 Processando {group_name}...")
    
    csv_files = glob.glob(os.path.join(group_path, "*.csv"))
    
    if not csv_files:
        print(f"   ⚠️  Nessun file CSV trovato in {group_path}")
        return None
    
    print(f"   Trovati {len(csv_files)} file CSV")
    
    # Unisci i CSV con data augmentation
    merged_df = merge_users_with_augmentation(csv_files, augmentation_factor)
    
    if merged_df is None or len(merged_df) == 0:
        print(f"   ⚠️  Nessun DataFrame valido per {group_name}")
        return None
    
    print(f"   ✓ DataFrame unito: {len(merged_df)} righe totali")
    
    # Preprocessa le colonne time-series
    print(f"   🔄 Preprocessing time-series...")
    processed_df = preprocess_timeseries_columns(merged_df)
    
    print(f"   ✓ Preprocessing completato: {processed_df.shape[1]} colonne finali")
    
    return processed_df


def setup_dataset():
    """
    Funzione principale per preparare il dataset con data augmentation avanzata.
    """
    print("🚀 Inizio preparazione dataset con Data Augmentation Avanzata\n")
    print("=" * 70)
    print("Tecniche applicate:")
    print("  1. Gaussian noise injection")
    print("  2. Magnitude scaling")
    print("  3. Linear interpolation")
    print("  4. Mixup augmentation")
    print("=" * 70)
    
    # Percorsi
    script_dir = Path(__file__).parent
    data_dir = script_dir / "clients_data"
    output_dir = script_dir / "ready_for_flwr_augmented"
    
    if not data_dir.exists():
        print(f"❌ Errore: La cartella {data_dir} non esiste!")
        return
    
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 Cartella di output: {output_dir}\n")
    
    group_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    if not group_folders:
        print(f"❌ Errore: Nessuna sottocartella trovata in {data_dir}")
        return
    
    print(f"📊 Trovati {len(group_folders)} gruppi\n")
    
    # Augmentation factor: quanto moltiplicare i dati (3-4x è un buon compromesso)
    augmentation_factor = 3
    
    for idx, group_folder in enumerate(group_folders):
        group_name = group_folder.name
        
        processed_df = merge_and_preprocess_group(group_folder, group_name, augmentation_factor)
        
        if processed_df is not None:
            output_file = output_dir / f"client_{idx}.csv"
            processed_df.to_csv(output_file, index=False)
            print(f"   💾 Salvato: {output_file.name} ({len(processed_df)} righe, {processed_df.shape[1]} colonne)")
    
    print("\n" + "=" * 70)
    print("✅ Preparazione dataset completata!")
    print(f"📁 I file pronti si trovano in: {output_dir}")
    
    ready_files = list(output_dir.glob("client_*.csv"))
    print(f"\n📋 Riepilogo: {len(ready_files)} file client pronti")
    print(f"   Augmentation factor: ~{augmentation_factor}x")
    print(f"   Tecniche: Noise + Scaling + Interpolation + Mixup")


if __name__ == "__main__":
    setup_dataset()
