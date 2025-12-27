#!/bin/bash
# Installation script for RNN Federated Learning dependencies

echo "🔧 Installing TensorFlow and dependencies for RNN FL..."
echo "=================================================="

# Activate virtual environment if it exists
if [ -d "flvenv" ]; then
    source flvenv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv flvenv
    source flvenv/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install TensorFlow
echo ""
echo "📦 Installing TensorFlow..."
pip install tensorflow

# Install additional dependencies
echo ""
echo "📦 Installing additional dependencies..."
pip install numpy pandas scikit-learn matplotlib

echo ""
echo "=================================================="
echo "✅ Installation complete!"
echo ""
echo "To verify installation, run:"
echo "  python -c 'import tensorflow as tf; print(tf.__version__)'"
echo ""
echo "To start training:"
echo "  python -m dtbagging.train_rnn_fl"
echo "=================================================="
