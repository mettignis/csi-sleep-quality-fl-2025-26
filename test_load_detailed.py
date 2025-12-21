#!/usr/bin/env python3
"""Test loading data with more details"""
import pandas as pd
from pathlib import Path

# Test reading one file
data_file = Path("/Users/annamettifogo/Desktop/polimi/csi/proj4/prova1/dtbagging/ready_for_flwr/client_0.csv")
print(f"File exists: {data_file.exists()}")
print(f"File path: {data_file}")

# Load
data = pd.read_csv(data_file)
print(f"\nShape: {data.shape}")
print(f"\nColumns: {list(data.columns)}")
print(f"\nFirst rows:\n{data.head()}")

# Check label
print(f"\nLabel column exists: {'label' in data.columns}")
print(f"Label values: {data['label'].unique()[:10]}")

# Drop NA
original_len = len(data)
data_clean = data.dropna()
print(f"\nRows before dropna: {original_len}")
print(f"Rows after dropna: {len(data_clean)}")

# Feature selection
feature_cols = [col for col in data_clean.columns 
                if col not in ['day', 'label'] 
                and not col.endswith('_time_series')
                and data_clean[col].dtype in ['int64', 'float64']]

print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")

X = data_clean[feature_cols]
y = data_clean['label']

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")

# Test split
split_idx = int(len(X) * 0.8)
print(f"\nSplit index: {split_idx}")
print(f"Train samples: {split_idx}")
print(f"Test samples: {len(X) - split_idx}")
