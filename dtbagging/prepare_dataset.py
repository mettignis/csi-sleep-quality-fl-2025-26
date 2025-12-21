import pandas as pd
import numpy as np
import ast
import os
import glob

# CONFIGURAZIONE
RAW_DATA_DIR = "raw_data"  # La cartella con le sottocartelle user_0, user_1...
OUTPUT_DIR = "clients_data" # Dove salveremo i file pronti per Flower

def clean_and_parse(df):
    """
    1. Unisce i pezzi.
    2. Converte le stringhe time-series in feature numeriche (Media, Std, Min, Max).
    """
    # Colonne Time Series da processare
    ts_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
    
    for col in ts_cols:
        if col in df.columns:
            # Funzione interna per parsare una singola riga
            def parse_row(val):
                try:
                    # Se è stringa "[1, 2]" -> Lista [1, 2]
                    if isinstance(val, str):
                        x = ast.literal_eval(val)
                        return np.array([i for i in x if i is not None], dtype=float)
                    return np.array([], dtype=float)
                except:
                    return np.array([], dtype=float)

            # Convertiamo la colonna in array numpy
            parsed_series = df[col].apply(parse_row)
            
            # Estraiamo Features Statistiche (MOLTO IMPORTANTE PER XGBOOST)
            df[f'{col}_mean'] = parsed_series.apply(lambda x: np.mean(x) if len(x) > 0 else 0)
            df[f'{col}_std'] = parsed_series.apply(lambda x: np.std(x) if len(x) > 0 else 0)
            df[f'{col}_min'] = parsed_series.apply(lambda x: np.min(x) if len(x) > 0 else 0)
            df[f'{col}_max'] = parsed_series.apply(lambda x: np.max(x) if len(x) > 0 else 0)
            
            # Rimuoviamo la colonna originale pesante
            df.drop(columns=[col], inplace=True)

    # Gestione NaN residui e colonne inutili
    df.fillna(0, inplace=True)
    if 'day' in df.columns: df.drop(columns=['day'], inplace=True)
    if 'date' in df.columns: df.drop(columns=['date'], inplace=True)
    
    return df

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Trova tutte le cartelle user_*
    user_folders = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "user_*")))
    
    if not user_folders:
        print(f"⚠️ Nessuna cartella trovata in {RAW_DATA_DIR}. Controlla i percorsi!")
        return

    print(f"Trovati {len(user_folders)} utenti/client.")

    for i, folder in enumerate(user_folders):
        print(f"Processando Client {i} (Folder: {folder})...")
        
        # Carica tutti i CSV dentro la cartella dell'utente
        all_files = glob.glob(os.path.join(folder, "*.csv"))
        if not all_files:
            print(f"  -> Nessun csv trovato in {folder}")
            continue
            
        # Merge (Unione)
        df_list = [pd.read_csv(f, sep=';') for f in all_files]
        full_df = pd.concat(df_list, ignore_index=True)
        
        # Pulizia e Feature Extraction
        clean_df = clean_and_parse(full_df)
        
        # Salvataggio
        output_path = os.path.join(OUTPUT_DIR, f"client_{i}.csv")
        clean_df.to_csv(output_path, index=False)
        print(f"  -> Salvato: {output_path} ({len(clean_df)} righe)")

    # Facciamo lo stesso per x_test.csv se esiste
    if os.path.exists("x_test.csv"):
        print("Processando x_test.csv...")
        test_df = pd.read_csv("x_test.csv", sep=";")
        clean_test = clean_and_parse(test_df)
        clean_test.to_csv(os.path.join(OUTPUT_DIR, "x_test_processed.csv"), index=False)
        print("Done.")

if __name__ == "__main__":
    main()