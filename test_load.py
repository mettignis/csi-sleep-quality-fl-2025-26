#!/usr/bin/env python3
"""Test loading data"""
from dtbagging.task import load_data

print("Testing data loading...")
for i in range(3):  # Test first 3 clients
    try:
        train_data, valid_data, num_train, num_test = load_data(i, 0.2)
        print(f"Client {i}: Train={num_train}, Test={num_test}")
        if num_train > 0:
            print(f"  Features: {train_data.num_col()}")
        else:
            print(f"  ERROR: No training samples!")
    except Exception as e:
        print(f"Client {i}: ERROR - {e}")
