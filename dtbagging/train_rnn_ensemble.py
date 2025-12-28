"""
Train multiple RNN models and ensemble their predictions for better MAE.

This script trains 5 different RNN models with different random seeds and architectures,
then averages their predictions for improved performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from train_rnn_fl import RNNFederatedLearning


def train_ensemble(n_models=5):
    """
    Train multiple RNN models with different seeds.
    
    Args:
        n_models: Number of models to train
    """
    print(f"🎯 Training ensemble of {n_models} models\n")
    print("=" * 70)
    
    models_dir = Path(__file__).parent / "experiments" / "exp_rnn_ensemble"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(n_models):
        print(f"\n{'='*70}")
        print(f"TRAINING MODEL {i+1}/{n_models} (seed={42+i})")
        print(f"{'='*70}\n")
        
        # Different configurations for diversity
        if i == 0:
            # Baseline configuration
            config = {
                'num_rounds': 40,
                'local_epochs': 20,
                'batch_size': 32,
                'learning_rate': 0.0005,
                'sequence_length': 7,
            }
        elif i == 1:
            # Higher learning rate, more rounds
            config = {
                'num_rounds': 50,
                'local_epochs': 15,
                'batch_size': 32,
                'learning_rate': 0.001,
                'sequence_length': 7,
            }
        elif i == 2:
            # Longer sequences
            config = {
                'num_rounds': 35,
                'local_epochs': 20,
                'batch_size': 32,
                'learning_rate': 0.0005,
                'sequence_length': 10,
            }
        elif i == 3:
            # Different batch size
            config = {
                'num_rounds': 40,
                'local_epochs': 25,
                'batch_size': 16,
                'learning_rate': 0.0003,
                'sequence_length': 7,
            }
        else:
            # Conservative
            config = {
                'num_rounds': 45,
                'local_epochs': 18,
                'batch_size': 32,
                'learning_rate': 0.0008,
                'sequence_length': 7,
            }
        
        trainer = RNNFederatedLearning(
            num_clients=9,
            num_rounds=config['num_rounds'],
            local_epochs=config['local_epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate'],
            sequence_length=config['sequence_length'],
            test_fraction=0.2,
            random_seed=42 + i  # Different seed for each model
        )
        
        trainer.train()
        
        print(f"\n✅ Model {i+1} training completed!\n")
    
    print("=" * 70)
    print(f"✅ Ensemble training completed! {n_models} models trained.")
    print(f"📁 Models saved in: {models_dir.parent / 'exp_rnn_federated'}")
    print("\nTo generate ensemble predictions, run:")
    print("  python -m dtbagging.inference_rnn_ensemble")


if __name__ == "__main__":
    train_ensemble(n_models=5)
