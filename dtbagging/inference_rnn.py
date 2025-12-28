"""
Inference script for RNN Federated Learning model.

This script loads a trained LSTM model and generates predictions for test data.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import argparse
from typing import List


class RNNPredictor:
    """Predictor class for trained RNN federated learning model."""
    
    def __init__(self, model_path: str, config_path: str = None):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to trained .h5 model file
            config_path: Optional path to config.json file
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        print(f"📂 Loading model from: {self.model_path}")
        # Handle .keras, .h5, and SavedModel formats
        try:
            self.model = keras.models.load_model(self.model_path, compile=False)
            # Recompile with proper metrics
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss=keras.losses.MeanSquaredError(),
                metrics=[keras.metrics.MeanAbsoluteError()]
            )
            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Load configuration
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Try to find config in same directory
            model_dir = self.model_path.parent
            config_files = list(model_dir.glob("config_*.json"))
            if config_files:
                self.config_path = config_files[-1]  # Use most recent
            else:
                self.config_path = None
        
        if self.config_path and self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            print(f"✓ Config loaded from: {self.config_path}")
        else:
            print("⚠️  No config file found, using defaults")
            self.config = {
                'sequence_length': 7,
                'feature_cols': None
            }
        
        self.sequence_length = self.config.get('sequence_length', 7)
        self.feature_cols = self.config.get('feature_cols')
        self.global_mean = np.array(self.config['global_mean']) if self.config.get('global_mean') else None
        self.global_std = np.array(self.config['global_std']) if self.config.get('global_std') else None
    
    def preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess test data for prediction.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Preprocessed numpy array (n_sequences, sequence_length, n_features)
        """
        import ast
        
        # Check if data has time_series columns that need preprocessing
        timeseries_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
        has_timeseries = any(col in df.columns for col in timeseries_cols)
        
        if has_timeseries:
            print("   Preprocessing time series columns...")
            df = self._preprocess_timeseries(df)
            print(f"   ✓ After preprocessing: {df.shape[1]} columns")
        
        # Select feature columns
        if self.feature_cols:
            # Use same features as training
            missing_cols = [col for col in self.feature_cols if col not in df.columns]
            if missing_cols:
                print(f"   ⚠️  Missing {len(missing_cols)} columns, filling with zeros")
                # Add missing columns with zeros
                for col in missing_cols:
                    df[col] = 0.0
            
            # Ensure correct column order
            X = df[self.feature_cols].values
        else:
            # Infer features (exclude metadata)
            feature_cols = [
                col for col in df.columns 
                if col not in ['Unnamed: 0', 'day', 'label', 'id']
                and not col.endswith('_time_series')
                and df[col].dtype in ['int64', 'float64']
            ]
            X = df[feature_cols].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Normalize using training statistics for consistency
        if self.global_mean is not None and self.global_std is not None:
            print(f"   Using global normalization statistics from training")
            X = (X - self.global_mean) / self.global_std
        else:
            print(f"   ⚠️  Using local normalization (may cause inconsistent predictions)")
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0) + 1e-8
            X = (X - X_mean) / X_std
        
        # Create sequences for LSTM
        print(f"   Creating sequences (length={self.sequence_length})...")
        X_seq = self._create_sequences(X)
        print(f"   ✓ Sequences created: {X_seq.shape}")
        return X_seq
    
    def _preprocess_timeseries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess time-series columns by extracting statistics.
        Similar to setup_dataset.py preprocessing.
        """
        import ast
        
        timeseries_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series']
        
        for col in timeseries_cols:
            if col not in df.columns:
                continue
            
            print(f"      Processing {col}...")
            base_name = col.replace('_time_series', '')
            
            def safe_parse(x):
                if pd.isna(x) or x is None:
                    return []
                if isinstance(x, (list, np.ndarray)):
                    # Filter out None values
                    return [v for v in x if v is not None]
                if isinstance(x, str):
                    try:
                        parsed = ast.literal_eval(x.strip())
                        if isinstance(parsed, (list, tuple)):
                            # Filter out None values
                            return [v for v in parsed if v is not None]
                        return []
                    except:
                        return []
                return []
            
            series_data = df[col].apply(safe_parse)
            
            # Calculate statistics - safe_mean handles empty lists
            def safe_mean(x):
                return np.mean(x) if len(x) > 0 else 0.0
            def safe_min(x):
                return np.min(x) if len(x) > 0 else 0.0
            def safe_max(x):
                return np.max(x) if len(x) > 0 else 0.0
            def safe_std(x):
                return np.std(x) if len(x) > 0 else 0.0
            
            df[f'{base_name}_mean'] = series_data.apply(safe_mean)
            df[f'{base_name}_min'] = series_data.apply(safe_min)
            df[f'{base_name}_max'] = series_data.apply(safe_max)
            df[f'{base_name}_std'] = series_data.apply(safe_std)
            
            # Drop original time series column
            df = df.drop(columns=[col])
        
        return df
    
    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Create sequences for LSTM from data.
        
        Args:
            X: Feature array (n_samples, n_features)
            
        Returns:
            X_seq: (n_sequences, sequence_length, n_features)
        """
        n_samples = len(X)
        
        if n_samples < self.sequence_length:
            # Pad with zeros if not enough data
            pad_length = self.sequence_length - n_samples
            X = np.vstack([np.zeros((pad_length, X.shape[1])), X])
            n_samples = len(X)
        
        # Create sliding windows
        X_seq = []
        for i in range(n_samples - self.sequence_length + 1):
            X_seq.append(X[i:i + self.sequence_length])
        
        return np.array(X_seq)
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for input data.
        
        Args:
            data: DataFrame with features
            
        Returns:
            Array of predictions
        """
        # Preprocess
        X = self.preprocess_data(data)
        
        print(f"📊 Input shape: {X.shape}")
        
        # Predict
        predictions = self.model.predict(X, verbose=0)
        predictions = predictions.flatten()
        
        print(f"✓ Generated {len(predictions)} predictions")
        
        return predictions
    
    def predict_from_csv(self, csv_path: str, output_path: str = None):
        """
        Generate predictions from CSV file and save results.
        
        Args:
            csv_path: Path to input CSV file
            output_path: Optional path to save predictions (defaults to submission.csv)
        """
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        print(f"\n📂 Loading data from: {csv_path}")
        # Try both comma and semicolon delimiters
        df = pd.read_csv(csv_path)
        if len(df.columns) == 1:
            # Likely semicolon-delimited
            print("   Retrying with semicolon delimiter...")
            df = pd.read_csv(csv_path, sep=';')
        print(f"✓ Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Generate predictions
        print("\n🔮 Generating predictions...")
        predictions = self.predict(df)
        
        # Create submission DataFrame
        # Note: Number of predictions might differ from original rows due to sequencing
        if len(predictions) < len(df):
            # Pad beginning with mean prediction
            pad_length = len(df) - len(predictions)
            mean_pred = np.mean(predictions)
            predictions = np.concatenate([np.full(pad_length, mean_pred), predictions])
        elif len(predictions) > len(df):
            # Truncate to match
            predictions = predictions[:len(df)]
        
        submission = pd.DataFrame({
            'id': range(len(predictions)),
            'label': predictions
        })
        
        # Save submission
        if output_path is None:
            output_path = Path(__file__).parent / "submission_rnn.csv"
        else:
            output_path = Path(output_path)
        
        submission.to_csv(output_path, index=False)
        print(f"\n💾 Predictions saved to: {output_path}")
        print(f"   Rows: {len(submission)}")
        print(f"   Mean prediction: {predictions.mean():.2f}")
        print(f"   Std prediction: {predictions.std():.2f}")
        print(f"   Min: {predictions.min():.2f}, Max: {predictions.max():.2f}")
        
        return submission


def main():
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(
        description="Generate predictions using trained RNN FL model"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.keras, .h5, or SavedModel directory)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (optional)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='dtbagging/test_data/x_test.csv',
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='dtbagging/submission_rnn.csv',
        help='Path to output submission file'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔮 RNN Federated Learning Inference")
    print("=" * 70)
    
    # Initialize predictor
    predictor = RNNPredictor(args.model, args.config)
    
    # Generate predictions
    predictor.predict_from_csv(args.input, args.output)
    
    print("\n" + "=" * 70)
    print("✅ Inference completed!")
    print("=" * 70)


if __name__ == "__main__":
    # If no arguments provided, use default paths
    import sys
    
    if len(sys.argv) == 1:
        print("=" * 70)
        print("🔮 RNN Federated Learning Inference")
        print("=" * 70)
        print("\nNo arguments provided. Usage example:")
        print("  python inference_rnn.py --model experiments/exp_rnn_federated/model.h5 --input test_data/x_test.csv")
        print("\nSearching for latest model...")
        
        # Try to find latest model
        exp_dir = Path(__file__).parent / "experiments" / "exp_rnn_federated"
        if exp_dir.exists():
            # Look for .keras files, then SavedModel directories, then .h5 files
            keras_files = list(exp_dir.glob("rnn_federated_model_*.keras"))
            model_dirs = [d for d in exp_dir.glob("rnn_federated_model_*") if d.is_dir()]
            h5_files = list(exp_dir.glob("rnn_federated_model_*.h5"))
            all_models = keras_files + model_dirs + h5_files
            
            if all_models:
                model_files = sorted(all_models)
                latest_model = model_files[-1]
                print(f"✓ Found model: {latest_model}")
                
                # Try to find config
                config_files = sorted(exp_dir.glob("config_*.json"))
                config_path = config_files[-1] if config_files else None
                
                # Use default test data
                test_data = Path(__file__).parent / "test_data" / "x_test.csv"
                
                if test_data.exists():
                    predictor = RNNPredictor(str(latest_model), str(config_path) if config_path else None)
                    predictor.predict_from_csv(str(test_data))
                    print("\n" + "=" * 70)
                    print("✅ Inference completed!")
                    print("=" * 70)
                else:
                    print(f"❌ Test data not found: {test_data}")
            else:
                print("❌ No model files found in experiments/exp_rnn_federated/")
                print("\n💡 To train a model, run:")
                print("   python -m dtbagging.train_rnn_fl")
        else:
            print("❌ Experiment directory not found. Please train a model first.")
            print("\n💡 To train a model, run:")
            print("   python -m dtbagging.train_rnn_fl")
    else:
        main()
