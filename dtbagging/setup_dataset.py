"""
Script per preparare i dati per Federated Learning.

Questo script:
1. Scansiona le sottocartelle in clients_data
2. Unisce i 5 CSV di ogni gruppo in un unico DataFrame
3. Preprocessa le colonne time-series (calcola statistiche: media, min, max, std)
4. Salva i file processati in ready_for_flwr come client_0.csv, client_1.csv, etc.
"""

import os
import glob
import ast
import pandas as pd
import numpy as np
from pathlib import Path


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
    Unisce tutti i CSV di un gruppo e li preprocessa.
    
    Args:
        group_path: Path alla cartella del gruppo
        group_name: Nome del gruppo (es. 'group0')
        
    Returns:
        DataFrame unito e preprocessato
    """
    print(f"\n📂 Processando {group_name}...")
    
    # Trova tutti i CSV nella cartella
    csv_files = glob.glob(os.path.join(group_path, "*.csv"))
    
    if not csv_files:
        print(f"   ⚠️  Nessun file CSV trovato in {group_path}")
        return None
    
    print(f"   Trovati {len(csv_files)} file CSV")
    
    # Leggi e unisci tutti i CSV
    dataframes = []
    for csv_file in csv_files:
        try:
            # Prova diversi metodi di parsing
            df = None
            
            # Metodo 1: Con engine python (più permissivo)
            try:
                df = pd.read_csv(csv_file, sep=';', engine='python', on_bad_lines='skip')
                if df.shape[1] == 1:
                    df = pd.read_csv(csv_file, sep=',', engine='python', on_bad_lines='skip')
            except:
                pass
            
            # Metodo 2: Con error_bad_lines=False (per pandas vecchi)
            if df is None or len(df) == 0:
                try:
                    df = pd.read_csv(csv_file, sep=';', on_bad_lines='warn', encoding='utf-8')
                except:
                    pass
            
            # Metodo 3: Prova con quoting
            if df is None or len(df) == 0:
                try:
                    df = pd.read_csv(csv_file, sep=';', quoting=3, engine='python')  # quoting=3 = QUOTE_NONE
                except:
                    pass
            
            if df is not None and len(df) > 0:
                dataframes.append(df)
                print(f"   ✓ Caricato: {os.path.basename(csv_file)} ({len(df)} righe, {df.shape[1]} colonne)")
            else:
                print(f"   ✗ Impossibile caricare: {os.path.basename(csv_file)}")
                
        except Exception as e:
            print(f"   ✗ Errore nel caricamento di {csv_file}: {e}")
    
    if not dataframes:
        print(f"   ⚠️  Nessun DataFrame valido per {group_name}")
        return None
    
    # Unisci tutti i DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"   ✓ DataFrame unito: {len(merged_df)} righe totali")
    
    # Preprocessa le colonne time-series
    print(f"   🔄 Preprocessing time-series...")
    processed_df = preprocess_timeseries_columns(merged_df)
    
    print(f"   ✓ Preprocessing completato: {processed_df.shape[1]} colonne finali")
    
    return processed_df


def setup_dataset():
    """
    Funzione principale per preparare il dataset.
    """
    print("🚀 Inizio preparazione dataset per Federated Learning\n")
    print("=" * 60)
    
    # Percorsi
    script_dir = Path(__file__).parent
    data_dir = script_dir / "clients_data"
    output_dir = script_dir / "ready_for_flwr"
    
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
        
        # Processa e unisci i dati del gruppo
        processed_df = merge_and_preprocess_group(group_folder, group_name)
        
        if processed_df is not None:
            # Salva il risultato
            output_file = output_dir / f"client_{idx}.csv"
            processed_df.to_csv(output_file, index=False)
            print(f"   💾 Salvato: {output_file.name} ({len(processed_df)} righe, {processed_df.shape[1]} colonne)")
    
    print("\n" + "=" * 60)
    print("✅ Preparazione dataset completata!")
    print(f"📁 I file pronti si trovano in: {output_dir}")
    
    # Mostra un riepilogo
    ready_files = list(output_dir.glob("client_*.csv"))
    print(f"\n📋 Riepilogo: {len(ready_files)} file client pronti per Flower")


if __name__ == "__main__":
    setup_dataset()
