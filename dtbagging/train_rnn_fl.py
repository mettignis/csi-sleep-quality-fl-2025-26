"""
Federated Learning Training with RNN (LSTM) for Sleep Quality Prediction.

This script implements federated learning using LSTM networks to predict sleep quality
from time series data. Each client trains locally on their data, and models are
aggregated using FedAvg (Federated Averaging).
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from pathlib import Path
from typing import Tuple, List, Dict
import json
from datetime import datetime


class RNNFederatedLearning:
    """Federated Learning trainer using LSTM for sleep quality regression."""
    
    def __init__(
        self,
        num_clients: int = 9,
        num_rounds: int = 15,
        local_epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        sequence_length: int = 7,
        test_fraction: float = 0.2,
        random_seed: int = 42
    ):
        """
        Initialize the Federated Learning trainer.
        
        Args:
            num_clients: Number of federated clients
            num_rounds: Number of FL rounds
            local_epochs: Epochs per client per round
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            sequence_length: Number of days in sequence for LSTM
            test_fraction: Fraction of data for validation
            random_seed: Random seed for reproducibility
        """
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        self.test_fraction = test_fraction
        self.random_seed = random_seed
        self.use_sequences = True  # Enable sequences for LSTM
        
        # Set random seeds
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        # Paths
        self.data_dir = Path(__file__).parent / "ready_for_flwr_augmented"  # Directory with client data preprocessed
        self.output_dir = Path(__file__).parent / "experiments" / "exp_rnn_federated"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model and training history
        self.global_model = None
        self.feature_cols = None
        self.num_features = None
        self.global_mean = None
        self.global_std = None
        self.history = {
            'rounds': [],
            'train_loss': [],
            'train_mae': [],
            'val_loss': [],
            'val_mae': []
        }
        
    def _compute_global_statistics(self):
        """Compute global mean and std across all clients for consistent normalization."""
        all_data = []
        
        for client_id in range(self.num_clients):
            data_file = self.data_dir / f"client_{client_id}.csv"
            if not data_file.exists():
                continue
            
            df = pd.read_csv(data_file)
            df = df.dropna(axis=1, how='all')
            
            if self.feature_cols is None:
                self.feature_cols = [
                    col for col in df.columns 
                    if col not in ['Unnamed: 0', 'day', 'label'] 
                    and not col.endswith('_time_series')
                    and df[col].dtype in ['int64', 'float64']
                ]
                self.num_features = len(self.feature_cols)
            
            X = df[self.feature_cols].values
            valid_idx = ~np.isnan(X).any(axis=1)
            X = X[valid_idx]
            all_data.append(X)
        
        if all_data:
            all_data = np.vstack(all_data)
            self.global_mean = np.mean(all_data, axis=0)
            self.global_std = np.std(all_data, axis=0) + 1e-8
            print(f"      Global statistics computed from {len(all_data)} samples")
    
    def load_client_data(self, client_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess data for a specific client.
        
        Args:
            client_id: Client identifier (0-8)
            
        Returns:
            X_train, y_train, X_val, y_val as numpy arrays
        """
        data_file = self.data_dir / f"client_{client_id}.csv"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Load data
        df = pd.read_csv(data_file)
        
        # Drop completely empty columns
        df = df.dropna(axis=1, how='all')
        
        # Select feature columns (exclude metadata and label)
        if self.feature_cols is None:
            self.feature_cols = [
                col for col in df.columns 
                if col not in ['Unnamed: 0', 'day', 'label'] 
                and not col.endswith('_time_series')
                and df[col].dtype in ['int64', 'float64']
            ]
            self.num_features = len(self.feature_cols)
        
        # Extract features and labels
        X = df[self.feature_cols].values
        y = df['label'].values
        
        # Handle missing values
        valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Normalize features using global statistics for consistency
        if self.global_mean is not None and self.global_std is not None:
            X = (X - self.global_mean) / self.global_std
        else:
            # Fallback to per-client normalization
            X_mean = np.nanmean(X, axis=0)
            X_std = np.nanstd(X, axis=0) + 1e-8
            X = (X - X_mean) / X_std
        
        # Create sequences for LSTM if enabled
        if self.use_sequences:
            X_seq, y_seq = self._create_sequences(X, y)
        else:
            X_seq = X
            y_seq = y
        
        # Train-validation split
        n_samples = len(X_seq)
        n_val = int(n_samples * self.test_fraction)
        n_train = n_samples - n_val
        
        # Random shuffle
        indices = np.random.permutation(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]
        
        X_train = X_seq[train_idx]
        y_train = y_seq[train_idx]
        X_val = X_seq[val_idx]
        y_val = y_seq[val_idx]
        
        return X_train, y_train, X_val, y_val
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM from sequential data.
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Label array (n_samples,)
            
        Returns:
            X_seq: (n_sequences, sequence_length, n_features)
            y_seq: (n_sequences,) - labels for last element in sequence
        """
        n_samples = len(X)
        
        if n_samples < self.sequence_length:
            # If not enough data, pad with zeros
            pad_length = self.sequence_length - n_samples
            X_padded = np.vstack([np.zeros((pad_length, X.shape[1])), X])
            y_padded = np.hstack([np.zeros(pad_length), y])
            X = X_padded
            y = y_padded
            n_samples = len(X)
        
        # Create sliding windows
        X_seq = []
        y_seq = []
        
        for i in range(n_samples - self.sequence_length + 1):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length - 1])  # Predict based on last day
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_model(self) -> keras.Model:
        """
        Build LSTM model for sleep quality prediction.
        
        Returns:
            Compiled Keras model
        """
        # Enhanced LSTM architecture with more capacity
        model = models.Sequential([
            layers.Input(shape=(self.sequence_length, self.num_features)),
            layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.2)),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2)),
            layers.LSTM(32, dropout=0.2),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0),
            loss=keras.losses.MeanSquaredError(),
            metrics=[keras.metrics.MeanAbsoluteError(), keras.metrics.RootMeanSquaredError()]
        )
        
        return model
    
    def get_model_weights(self, model: keras.Model) -> List[np.ndarray]:
        """Get model weights as list of numpy arrays."""
        return model.get_weights()
    
    def set_model_weights(self, model: keras.Model, weights: List[np.ndarray]):
        """Set model weights from list of numpy arrays."""
        model.set_weights(weights)
    
    def federated_averaging(
        self, 
        client_weights: List[List[np.ndarray]], 
        client_sizes: List[int]
    ) -> List[np.ndarray]:
        """
        Perform Federated Averaging (FedAvg) to aggregate client models.
        
        Args:
            client_weights: List of weight lists from each client
            client_sizes: Number of training samples per client
            
        Returns:
            Averaged weights
        """
        total_size = sum(client_sizes)
        
        # Initialize with zeros
        avg_weights = [np.zeros_like(w) for w in client_weights[0]]
        
        # Weighted average
        for client_w, size in zip(client_weights, client_sizes):
            weight = size / total_size
            for i, w in enumerate(client_w):
                avg_weights[i] += w * weight
        
        return avg_weights
    
    def train_client(
        self, 
        client_id: int, 
        global_weights: List[np.ndarray] = None
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model on a single client's data.
        
        Args:
            client_id: Client identifier
            global_weights: Initial weights from global model
            
        Returns:
            Tuple of (updated_weights, n_train_samples, metrics)
        """
        # Load client data
        X_train, y_train, X_val, y_val = self.load_client_data(client_id)
        n_train = len(X_train)
        
        if n_train == 0:
            print(f"  ⚠️  Client {client_id}: No training data available")
            return None, 0, {}
        
        # Build model
        model = self.build_model()
        
        # Initialize with global weights if available
        if global_weights is not None:
            self.set_model_weights(model, global_weights)
        
        # Callbacks for better training
        callbacks_list = []
        
        if len(X_val) > 0:
            # Early stopping
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=0
            )
            callbacks_list.append(early_stop)
            
            # Reduce learning rate on plateau
            reduce_lr = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-6,
                verbose=0
            )
            callbacks_list.append(reduce_lr)
        
        # Train with callbacks
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if len(X_val) > 0 else None,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            callbacks=callbacks_list,
            verbose=0
        )
        
        # Get updated weights
        updated_weights = self.get_model_weights(model)
        
        # Collect metrics - handle different metric key names
        # Keras may use 'mean_absolute_error' or 'mae' depending on version
        mae_key = 'mean_absolute_error' if 'mean_absolute_error' in history.history else 'mae'
        val_mae_key = 'val_mean_absolute_error' if 'val_mean_absolute_error' in history.history else 'val_mae'
        
        metrics = {
            'train_loss': history.history['loss'][-1],
            'train_mae': history.history[mae_key][-1],
            'val_loss': history.history['val_loss'][-1] if X_val.shape[0] > 0 else None,
            'val_mae': history.history[val_mae_key][-1] if X_val.shape[0] > 0 and val_mae_key in history.history else None,
            'n_train': n_train,
            'n_val': len(X_val)
        }
        
        return updated_weights, n_train, metrics
    
    def evaluate_global_model(self) -> Dict:
        """
        Evaluate global model on all clients' validation data.
        
        Returns:
            Dictionary with aggregated metrics
        """
        all_losses = []
        all_maes = []
        total_samples = 0
        
        for client_id in range(self.num_clients):
            try:
                # Load client data
                _, _, X_val, y_val = self.load_client_data(client_id)
                
                if len(X_val) == 0:
                    continue
                
                # Evaluate - returns [loss, mae, rmse]
                results = self.global_model.evaluate(X_val, y_val, verbose=0)
                loss = results[0]
                mae = results[1]
                
                all_losses.append(loss * len(X_val))
                all_maes.append(mae * len(X_val))
                total_samples += len(X_val)
                
            except Exception as e:
                print(f"  ⚠️  Error evaluating client {client_id}: {e}")
                continue
        
        if total_samples == 0:
            return {'loss': None, 'mae': None}
        
        avg_loss = sum(all_losses) / total_samples
        avg_mae = sum(all_maes) / total_samples
        
        return {'loss': avg_loss, 'mae': avg_mae}
    
    def train(self):
        """Execute federated learning training."""
        print("=" * 70)
        print("🚀 Starting RNN Federated Learning Training")
        print("=" * 70)
        print(f"📊 Configuration:")
        print(f"   Clients: {self.num_clients}")
        print(f"   Rounds: {self.num_rounds}")
        print(f"   Local epochs: {self.local_epochs}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Sequence length: {self.sequence_length}")
        print(f"   Learning rate: {self.learning_rate}")
        print("=" * 70)
        
        # Initialize global model
        print("\n🔧 Initializing global model...")
        
        # Compute global normalization statistics
        print("   Computing global normalization statistics...")
        self._compute_global_statistics()
        
        # Load first client to determine feature dimensions
        X_temp, _, _, _ = self.load_client_data(0)
        print(f"   Input shape: {X_temp.shape}")
        print(f"   Features: {self.num_features}")
        
        self.global_model = self.build_model()
        print(f"   Model parameters: {self.global_model.count_params():,}")
        
        # Training loop
        for round_num in range(1, self.num_rounds + 1):
            print(f"\n{'=' * 70}")
            print(f"ROUND {round_num}/{self.num_rounds}")
            print(f"{'=' * 70}")
            
            # Get current global weights
            global_weights = self.get_model_weights(self.global_model)
            
            # Train on all clients
            client_weights = []
            client_sizes = []
            round_metrics = []
            
            for client_id in range(self.num_clients):
                print(f"\n📱 Client {client_id}:")
                
                try:
                    weights, n_train, metrics = self.train_client(client_id, global_weights)
                    
                    if weights is None:
                        continue
                    
                    client_weights.append(weights)
                    client_sizes.append(n_train)
                    round_metrics.append(metrics)
                    
                    print(f"   Train samples: {metrics['n_train']}, Val samples: {metrics['n_val']}")
                    print(f"   Train Loss: {metrics['train_loss']:.4f}, Train MAE: {metrics['train_mae']:.4f}")
                    if metrics['val_loss'] is not None:
                        print(f"   Val Loss: {metrics['val_loss']:.4f}, Val MAE: {metrics['val_mae']:.4f}")
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
                    continue
            
            if not client_weights:
                print("\n❌ No clients completed training in this round!")
                break
            
            # Aggregate models using FedAvg
            print(f"\n🔄 Aggregating {len(client_weights)} client models...")
            aggregated_weights = self.federated_averaging(client_weights, client_sizes)
            self.set_model_weights(self.global_model, aggregated_weights)
            
            # Evaluate global model
            print("\n📊 Evaluating global model...")
            global_metrics = self.evaluate_global_model()
            
            if global_metrics['loss'] is not None:
                print(f"   Global Val Loss: {global_metrics['loss']:.4f}")
                print(f"   Global Val MAE: {global_metrics['mae']:.4f}")
                
                # Store history
                self.history['rounds'].append(round_num)
                self.history['val_loss'].append(global_metrics['loss'])
                self.history['val_mae'].append(global_metrics['mae'])
            
            # Calculate average training metrics
            avg_train_loss = np.mean([m['train_loss'] for m in round_metrics])
            avg_train_mae = np.mean([m['train_mae'] for m in round_metrics])
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_mae'].append(avg_train_mae)
            
            print(f"\n✓ Round {round_num} completed!")
        
        # Save final model and results
        self._save_results()
        
        print("\n" + "=" * 70)
        print("✅ Federated Learning Training Completed!")
        print("=" * 70)
    
    def _save_results(self):
        """Save the trained model and training history."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model (use .keras extension for Keras 3)
        model_path = self.output_dir / f"rnn_federated_model_{timestamp}.keras"
        self.global_model.save(model_path)
        print(f"\n💾 Model saved: {model_path}")
        
        # Save weights as pickle (handles different shapes better)
        weights_path = self.output_dir / f"rnn_federated_weights_{timestamp}.pkl"
        import pickle
        with open(weights_path, 'wb') as f:
            pickle.dump(self.get_model_weights(self.global_model), f)
        print(f"💾 Weights saved: {weights_path}")
        
        # Save history
        history_path = self.output_dir / f"training_history_{timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"💾 History saved: {history_path}")
        
        # Save configuration
        config = {
            'num_clients': self.num_clients,
            'num_rounds': self.num_rounds,
            'local_epochs': self.local_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'sequence_length': self.sequence_length,
            'test_fraction': self.test_fraction,
            'num_features': self.num_features,
            'feature_cols': self.feature_cols,
            'global_mean': self.global_mean.tolist() if self.global_mean is not None else None,
            'global_std': self.global_std.tolist() if self.global_std is not None else None,
            'timestamp': timestamp
        }
        config_path = self.output_dir / f"config_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Config saved: {config_path}")
        
        # Create README
        readme_path = self.output_dir / "README.md"
        readme_content = f"""# RNN Federated Learning Experiment

## Configuration
- **Model**: LSTM Sequential (64→32 units with dropout 0.2, Dense 16 units with dropout 0.3)
- **Clients**: {self.num_clients}
- **Rounds**: {self.num_rounds}
- **Local Epochs**: {self.local_epochs}
- **Batch Size**: {self.batch_size}
- **Learning Rate**: {self.learning_rate}
- **Sequence Length**: {self.sequence_length} days
- **Features**: {self.num_features}

## Results
- **Final Val Loss**: {self.history['val_loss'][-1] if self.history['val_loss'] else 'N/A'}
- **Final Val MAE**: {self.history['val_mae'][-1] if self.history['val_mae'] else 'N/A'}

## Files
- `{model_path.name}`: Trained model (Keras format)
- `{weights_path.name}`: Model weights
- `{history_path.name}`: Training history
- `{config_path.name}`: Configuration

## Training Date
{timestamp}
"""
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"📝 README saved: {readme_path}")


def main():
    """Main entry point for RNN federated learning training."""
    # Configure training with improved hyperparameters for better MAE
    trainer = RNNFederatedLearning(
        num_clients=9,
        num_rounds=15,  # More rounds for better convergence
        local_epochs=10,  # More epochs per client
        batch_size=32,  # Larger batch for more stable gradients
        learning_rate=0.0005,  # Lower LR for fine-tuning
        sequence_length=5,
        test_fraction=0.2,
        random_seed=42
    )
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main()
