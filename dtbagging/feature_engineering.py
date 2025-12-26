"""
Feature Engineering per Esperimento 006

Questo script estende il preprocessing del baseline aggiungendo:
1. Interazioni tra variabili correlate
2. Statistiche avanzate (percentili, range, coefficient of variation)
3. Ratios e feature derivate
"""

import pandas as pd
import numpy as np
from pathlib import Path


def add_engineered_features(df):
    """
    Aggiunge feature engineered al DataFrame.
    
    Args:
        df: DataFrame con feature base già preprocessate
        
    Returns:
        DataFrame con feature aggiuntive
    """
    print("   🔧 Feature Engineering in corso...")
    
    # =================================================================
    # 1. INTERAZIONI TRA VARIABILI CORRELATE
    # =================================================================
    
    # Heart rate × Stress (correlazione cuore-stress)
    if 'hr_mean' in df.columns and 'stress_mean' in df.columns:
        df['hr_stress_interaction'] = df['hr_mean'] * df['stress_mean']
        print("      ✓ hr_stress_interaction")
    
    # Sleep efficiency × Heart rate (qualità sonno-frequenza cardiaca)
    if 'sleep_sleepTimeSeconds' in df.columns and 'hr_mean' in df.columns:
        # Approssimazione sleep efficiency: sleepTime / (sleepTime + awakeTime)
        if 'sleep_awakeSleepSeconds' in df.columns:
            total_time = df['sleep_sleepTimeSeconds'] + df['sleep_awakeSleepSeconds']
            sleep_efficiency = df['sleep_sleepTimeSeconds'] / (total_time + 1)  # +1 per evitare divisione per 0
            df['sleep_hr_interaction'] = sleep_efficiency * df['hr_mean']
            print("      ✓ sleep_hr_interaction")
    
    # Activity intensity (calories per distance)
    if 'act_totalCalories' in df.columns and 'act_distance' in df.columns:
        df['activity_intensity'] = df['act_totalCalories'] / (df['act_distance'] + 0.001)  # +0.001 per evitare div/0
        print("      ✓ activity_intensity")
    
    # Respiratory × Stress
    if 'resp_mean' in df.columns and 'stress_mean' in df.columns:
        df['resp_stress_interaction'] = df['resp_mean'] * df['stress_mean']
        print("      ✓ resp_stress_interaction")
    
    # =================================================================
    # 2. STATISTICHE AVANZATE SU TIME-SERIES
    # =================================================================
    
    for prefix in ['hr', 'resp', 'stress']:
        mean_col = f'{prefix}_mean'
        std_col = f'{prefix}_std'
        min_col = f'{prefix}_min'
        max_col = f'{prefix}_max'
        
        # Range (max - min)
        if min_col in df.columns and max_col in df.columns:
            df[f'{prefix}_range'] = df[max_col] - df[min_col]
            print(f"      ✓ {prefix}_range")
        
        # Coefficient of Variation (std / mean) - misura variabilità relativa
        if mean_col in df.columns and std_col in df.columns:
            df[f'{prefix}_cv'] = df[std_col] / (df[mean_col] + 0.001)  # +0.001 per evitare div/0
            print(f"      ✓ {prefix}_cv")
    
    # =================================================================
    # 3. RATIOS E FEATURE DERIVATE
    # =================================================================
    
    # Deep sleep ratio (deepSleep / totalSleep)
    if 'sleep_deepSleepSeconds' in df.columns and 'sleep_sleepTimeSeconds' in df.columns:
        df['deep_sleep_ratio'] = df['sleep_deepSleepSeconds'] / (df['sleep_sleepTimeSeconds'] + 1)
        print("      ✓ deep_sleep_ratio")
    
    # REM sleep ratio
    if 'sleep_remSleepSeconds' in df.columns and 'sleep_sleepTimeSeconds' in df.columns:
        df['rem_sleep_ratio'] = df['sleep_remSleepSeconds'] / (df['sleep_sleepTimeSeconds'] + 1)
        print("      ✓ rem_sleep_ratio")
    
    # Light sleep ratio
    if 'sleep_lightSleepSeconds' in df.columns and 'sleep_sleepTimeSeconds' in df.columns:
        df['light_sleep_ratio'] = df['sleep_lightSleepSeconds'] / (df['sleep_sleepTimeSeconds'] + 1)
        print("      ✓ light_sleep_ratio")
    
    # Active time efficiency (activeCalories / activeTime)
    if 'act_activeKilocalories' in df.columns and 'act_activeTime' in df.columns:
        df['active_efficiency'] = df['act_activeKilocalories'] / (df['act_activeTime'] + 1)
        print("      ✓ active_efficiency")
    
    # HR variability indicator (std / range)
    if 'hr_std' in df.columns and 'hr_range' in df.columns:
        df['hr_variability_index'] = df['hr_std'] / (df['hr_range'] + 0.001)
        print("      ✓ hr_variability_index")
    
    # Sleep fragmentation (awakeCount / sleepTime in hours)
    if 'sleep_awakeCount' in df.columns and 'sleep_sleepTimeSeconds' in df.columns:
        sleep_hours = df['sleep_sleepTimeSeconds'] / 3600
        df['sleep_fragmentation'] = df['sleep_awakeCount'] / (sleep_hours + 0.1)
        print("      ✓ sleep_fragmentation")
    
    # =================================================================
    # 4. NORMALIZZAZIONI E TRASFORMAZIONI
    # =================================================================
    
    # Log transform per variabili con alta skewness (es: calories, distance)
    for col in ['act_totalCalories', 'act_distance']:
        if col in df.columns:
            log_col = f'{col}_log'
            df[log_col] = np.log1p(df[col])  # log1p = log(1+x) per gestire zeri
            print(f"      ✓ {log_col}")
    
    print(f"   ✅ Aggiunte {len(df.columns) - len(df.columns)} nuove feature")
    
    return df


def get_feature_count_change(original_count, new_count):
    """Calcola quante feature sono state aggiunte."""
    return new_count - original_count


if __name__ == "__main__":
    print("Feature Engineering Module")
    print("Questo modulo viene importato da altri script.")
