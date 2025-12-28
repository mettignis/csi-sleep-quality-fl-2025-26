"""
Ensemble inference: Average predictions from multiple trained models for better accuracy.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from inference_rnn import RNNPredictor
import argparse


def ensemble_predict(model_paths, input_csv, output_csv):
    """
    Generate ensemble predictions by averaging multiple models.
    
    Args:
        model_paths: List of paths to trained models
        input_csv: Path to test data
        output_csv: Path to save ensemble predictions
    """
    print("=" * 70)
    print("🎯 RNN ENSEMBLE INFERENCE")
    print("=" * 70)
    
    print(f"\n📂 Loading {len(model_paths)} models...\n")
    
    # Load all models
    predictors = []
    for i, model_path in enumerate(model_paths):
        print(f"  [{i+1}/{len(model_paths)}] Loading: {Path(model_path).name}")
        try:
            predictor = RNNPredictor(model_path)
            predictors.append(predictor)
        except Exception as e:
            print(f"    ⚠️  Failed to load: {e}")
    
    if not predictors:
        print("\n❌ No models loaded successfully!")
        return
    
    print(f"\n✓ Successfully loaded {len(predictors)} models")
    
    # Load test data
    print(f"\n📂 Loading test data: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Try semicolon delimiter if only 1 column
    if len(df.columns) == 1:
        df = pd.read_csv(input_csv, sep=';')
    
    print(f"✓ Loaded {len(df)} test samples")
    
    # Generate predictions from each model
    print(f"\n🔮 Generating predictions from {len(predictors)} models...")
    all_predictions = []
    
    for i, predictor in enumerate(predictors):
        print(f"  [{i+1}/{len(predictors)}] Predicting...")
        predictions = predictor.predict(df.copy())
        all_predictions.append(predictions)
        print(f"    Mean: {predictions.mean():.2f}, Std: {predictions.std():.2f}")
    
    # Ensemble: Average predictions
    print(f"\n📊 Computing ensemble (averaging {len(all_predictions)} predictions)...")
    all_predictions = np.array(all_predictions)
    ensemble_predictions = np.mean(all_predictions, axis=0)
    
    # Also compute median as alternative
    median_predictions = np.median(all_predictions, axis=0)
    
    print(f"\n📈 Ensemble Statistics:")
    print(f"  Mean prediction: {ensemble_predictions.mean():.2f}")
    print(f"  Std prediction: {ensemble_predictions.std():.2f}")
    print(f"  Min: {ensemble_predictions.min():.2f}, Max: {ensemble_predictions.max():.2f}")
    print(f"\n  Individual model variance (avg): {all_predictions.std(axis=0).mean():.2f}")
    
    # Save ensemble predictions
    submission_df = pd.DataFrame({
        'id': range(len(ensemble_predictions)),
        'label': ensemble_predictions
    })
    
    output_path = Path(output_csv)
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n💾 Ensemble predictions saved to: {output_path}")
    print(f"   Rows: {len(submission_df)}")
    
    # Also save median version
    median_output = output_path.parent / (output_path.stem + "_median.csv")
    median_df = pd.DataFrame({
        'id': range(len(median_predictions)),
        'label': median_predictions
    })
    median_df.to_csv(median_output, index=False)
    print(f"💾 Median predictions saved to: {median_output}")
    
    print("\n" + "=" * 70)
    print("✅ Ensemble inference completed!")
    print("=" * 70)


def find_latest_models(exp_dir, n_models=5):
    """Find the N most recent trained models."""
    model_files = sorted(exp_dir.glob("rnn_federated_model_*.keras"), 
                        key=lambda x: x.stat().st_mtime, 
                        reverse=True)
    return model_files[:n_models]


def main():
    parser = argparse.ArgumentParser(description="Ensemble RNN inference")
    parser.add_argument('--models', nargs='+', help='Paths to model files')
    parser.add_argument('--n-models', type=int, default=5, help='Number of latest models to use')
    parser.add_argument('--input', default='test_data/x_test.csv', help='Input CSV file')
    parser.add_argument('--output', default='submission_rnn_ensemble.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    exp_dir = script_dir / "experiments" / "exp_rnn_federated"
    
    if args.models:
        model_paths = [Path(m) for m in args.models]
    else:
        print(f"🔍 Searching for {args.n_models} latest models in: {exp_dir}")
        model_paths = find_latest_models(exp_dir, args.n_models)
        
        if not model_paths:
            print(f"❌ No models found in {exp_dir}")
            return
        
        print(f"✓ Found {len(model_paths)} models")
    
    input_path = script_dir / args.input
    output_path = script_dir / args.output
    
    ensemble_predict(model_paths, input_path, output_path)


if __name__ == "__main__":
    main()
