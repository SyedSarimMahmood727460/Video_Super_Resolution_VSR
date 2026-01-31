#!/bin/bash
set -euo pipefail

echo "========================================="
echo "=== LiteVSR Streamlit UI ==="
echo "========================================="
echo ""

# Optional: load Mahti modules (adjust to your project setup)
if [[ "$HOSTNAME" == *"mahti"* ]]; then
    echo "Loading Mahti modules..."
    module load gcc/11.2.0 || true
    module load pytorch/2.7 || module load pytorch || true
    echo "✓ Modules loaded"
fi

echo ""

# Activate venv (assumes project root contains .venv)
echo "Activating virtual environment..."
if [ -d .venv ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ Virtual environment not found, creating..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Installing dependencies..."
    pip install -U pip
    pip install -e .
    echo "✓ Environment ready"
fi

echo ""

# Check GPU availability
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU available:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ GPU not detected, will use CPU"
fi

echo ""
echo "========================================="

# Launch Streamlit UI bound to all interfaces for port forwarding
PORT=${PORT:-8501}
HOSTNAME_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1")

echo "🚀 Launching Streamlit UI..."
echo "📍 Local:    http://localhost:$PORT"
echo "📍 Network:  http://$HOSTNAME_IP:$PORT"
echo ""
echo "Press Ctrl+C to stop"
echo "========================================="
echo ""

streamlit run vsr/ui.py --server.port $PORT --server.address 0.0.0.0
