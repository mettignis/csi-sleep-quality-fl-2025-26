"""
Script per preparare i dati per Federated Learning con Data Augmentation.

Questo script:
1. Scansiona le sottocartelle in clients_data
2. Unisce i 5 CSV di ogni gruppo in un unico DataFrame
3. Applica data augmentation: inserisce 5 righe interpolate linearmente tra l'ultima riga
   di un dataset user e la prima riga del dataset user successivo
4. Preprocessa le colonne time-series (calcola statistiche: media, min, max, std)
5. Salva i file processati in ready_for_flwr come client_0.csv, client_1.csv, etc.
"""

import os
import glob
import ast
import pandas as pd
import numpy as np
from pathlib import Path


def linear_interpolate_rows(row1, row2, num_steps=5):
    """
    Crea righe interpolate linearmente tra due righe.
    
    Args:
        row1: Prima riga (pandas Series)
        row2: Seconda riga (pandas Series)
        num_steps: Numero di righe da interpolare
        
    Returns:
        Lista di DataFrame con le righe interpolate
    """
    interpolated_rows = []
    
    for step in range(1, num_steps + 1):
        # Calcola il peso per l'interpolazione (0 = row1, 1 = row2)
        alpha = step / (num_steps + 1)
        
        # Crea nuova riga interpolata
        new_row = {}
        
        for col in row1.index:
            val1 = row1[col]
            val2 = row2[col]
            
            # Salta colonne non numeriche o con valori mancanti
            if pd.isna(val1) or pd.isna(val2):
                new_row[col] = np.nan
                continue
            
            # Gestisci colonne categoriali o non numeriche
            if isinstance(val1, str) or isinstance(val2, str):
                # Per stringhe, usa il valore del primo user
                new_row[col] = val1
            elif col in ['day']:
                # Per la colonna day, interpola linearmente e arrotonda
                new_row[col] = int(np.round(val1 * (1 - alpha) + val2 * alpha))
            elif col.endswith('_time_series'):
                # Per time series, interpola gli array
                try:
                    if isinstance(val1, str):
                        arr1 = ast.literal_eval(val1)
                    else:
                        arr1 = val1 if isinstance(val1, list) else [val1]
                    
                    if isinstance(val2, str):
                        arr2 = ast.literal_eval(val2)
                    else:
                        arr2 = val2 if isinstance(val2, list) else [val2]
                    
                    # Assicurati che abbiano la stessa lunghezza
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
                # Per valori numerici, interpola linearmente
                try:
                    new_row[col] = val1 * (1 - alpha) + val2 * alpha
                except:
                    new_row[col] = val1
        
        interpolated_rows.append(new_row)
    
    return interpolated_rows


def merge_users_with_augmentation(csv_files):
    """
    Unisce i CSV di diversi utenti aggiungendo righe interpolate tra loro.
    
    Args:
        csv_files: Lista di percorsi ai file CSV
        
    Returns:
        DataFrame unito con data augmentation
    """
    all_data = []
    
    for idx, csv_file in enumerate(sorted(csv_files)):
        try:
            # Leggi il CSV dell'utente corrente
            df = None
            
            # Prova diversi metodi di parsing
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
            
            # Se non è il primo utente, aggiungi righe interpolate
            if all_data and len(df) > 0:
                # Prendi l'ultima riga dell'utente precedente
                last_row = all_data[-1]
                # Prendi la prima riga dell'utente corrente
                first_row = df.iloc[0]
                
                # Genera 5 righe interpolate
                print(f"      → Aggiungendo 5 righe interpolate...")
                interpolated = linear_interpolate_rows(last_row, first_row, num_steps=5)
                
                # Aggiungi le righe interpolate
                for interp_row in interpolated:
                    all_data.append(pd.Series(interp_row))
            
            # Aggiungi tutte le righe dell'utente corrente
            for _, row in df.iterrows():
                all_data.append(row)
                
        except Exception as e:
            print(f"   ✗ Errore nel caricamento di {csv_file}: {e}")
            continue
    
    if not all_data:
        return None
    
    # Converti in DataFrame
    merged_df = pd.DataFrame(all_data)
    merged_df = merged_df.reset_index(drop=True)
    
    return merged_df


def preprocess_timeseries_columns(df):
    """
    Preprocessa le colonne time-series convertendole da stringhe a statistiche numeriche.
    
    Args:
        df: DataFrame con colonne time-series come stringhe
        
    Returns:
        DataFrame con colonne time-series elaborate (media, min, max, std)
    """
    timeseries_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
    
    for col in timeseries_cols:
        if col not in df.columns:
            continue
            
        print(f"   Preprocessing colonna: {col}")
        
        # Converti stringa in lista
        try:
            def safe_parse(x):
                """Parse sicuro che gestisce vari casi."""
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
            
        except Exception as e:
            print(f"   ⚠️  Errore nel parsing di {col}: {e}")
            # Prova un altro approccio: rimuovi spazi e converti
            try:
                series_data = df[col].apply(lambda x: ast.literal_eval(str(x).strip()) if pd.notna(x) else [])
            except:
                print(f"   ✗ Impossibile processare {col}, colonna saltata")
                continue
        
        # Calcola statistiche (filtra solo liste valide non vuote)
        base_name = col.replace('_time_series', '')
        
        def safe_mean(x):
            if isinstance(x, list) and len(x) > 0:
                try:
                    return np.mean([v for v in x if v is not None])
                except:
                    return np.nan
            return np.nan
        
        def safe_min(x):
            if isinstance(x, list) and len(x) > 0:
                try:
                    return np.min([v for v in x if v is not None])
                except:
                    return np.nan
            return np.nan
        
        def safe_max(x):
            if isinstance(x, list) and len(x) > 0:
                try:
                    return np.max([v for v in x if v is not None])
                except:
                    return np.nan
            return np.nan
        
        def safe_std(x):
            if isinstance(x, list) and len(x) > 0:
                try:
                    return np.std([v for v in x if v is not None])
                except:
                    return np.nan
            return np.nan
        
        df[f'{base_name}_mean'] = series_data.apply(safe_mean)
        df[f'{base_name}_min'] = series_data.apply(safe_min)
        df[f'{base_name}_max'] = series_data.apply(safe_max)
        df[f'{base_name}_std'] = series_data.apply(safe_std)
        
        # Rimuovi la colonna originale
        df = df.drop(columns=[col])
    
    return df


def merge_and_preprocess_group(group_path, group_name):
    """
    Unisce tutti i CSV di un gruppo con data augmentation e li preprocessa.
    
    Args:
        group_path: Path alla cartella del gruppo
        group_name: Nome del gruppo (es. 'group0')
        
    Returns:
        DataFrame unito e preprocessato con data augmentation
    """
    print(f"\n📂 Processando {group_name}...")
    
    # Trova tutti i CSV nella cartella
    csv_files = glob.glob(os.path.join(group_path, "*.csv"))
    
    if not csv_files:
        print(f"   ⚠️  Nessun file CSV trovato in {group_path}")
        return None
    
    print(f"   Trovati {len(csv_files)} file CSV")
    
    # Unisci i CSV con data augmentation
    merged_df = merge_users_with_augmentation(csv_files)
    
    if merged_df is None or len(merged_df) == 0:
        print(f"   ⚠️  Nessun DataFrame valido per {group_name}")
        return None
    
    original_rows = len(csv_files) * (len(merged_df) // (len(csv_files) + (len(csv_files) - 1) * 5))
    augmented_rows = len(merged_df) - original_rows
    print(f"   ✓ DataFrame unito: {len(merged_df)} righe totali")
    print(f"      ({original_rows} originali + {augmented_rows} augmented)")
    
    # Preprocessa le colonne time-series
    print(f"   🔄 Preprocessing time-series...")
    processed_df = preprocess_timeseries_columns(merged_df)
    
    print(f"   ✓ Preprocessing completato: {processed_df.shape[1]} colonne finali")
    
    return processed_df


def setup_dataset():
    """
    Funzione principale per preparare il dataset con data augmentation.
    """
    print("🚀 Inizio preparazione dataset per Federated Learning (con Data Augmentation)\n")
    print("=" * 70)
    
    # Percorsi
    script_dir = Path(__file__).parent
    data_dir = script_dir / "clients_data"
    output_dir = script_dir / "ready_for_flwr_augmented"
    
    # Verifica che la cartella dei dati esista
    if not data_dir.exists():
        print(f"❌ Errore: La cartella {data_dir} non esiste!")
        return
    
    # Crea la cartella di output
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Cartella di output: {output_dir}\n")
    
    # Trova tutte le sottocartelle (gruppi)
    group_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    if not group_folders:
        print(f"❌ Errore: Nessuna sottocartella trovata in {data_dir}")
        return
    
    print(f"📊 Trovati {len(group_folders)} gruppi\n")
    
    # Processa ogni gruppo
    for idx, group_folder in enumerate(group_folders):
        group_name = group_folder.name
        
        # Processa e unisci i dati del gruppo con augmentation
        processed_df = merge_and_preprocess_group(group_folder, group_name)
        
        if processed_df is not None:
            # Salva il risultato
            output_file = output_dir / f"client_{idx}.csv"
            processed_df.to_csv(output_file, index=False)
            print(f"   💾 Salvato: {output_file.name} ({len(processed_df)} righe, {processed_df.shape[1]} colonne)")
    
    print("\n" + "=" * 70)
    print("✅ Preparazione dataset completata!")
    print(f"📁 I file pronti si trovano in: {output_dir}")
    
    # Mostra un riepilogo
    ready_files = list(output_dir.glob("client_*.csv"))
    print(f"\n📋 Riepilogo: {len(ready_files)} file client pronti per Flower")
    print(f"   (con data augmentation: 5 righe interpolate tra ogni coppia di utenti)")


if __name__ == "__main__":
    setup_dataset()
