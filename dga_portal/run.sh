#!/bin/bash
# DGA Portal Launch Script
# ========================

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║          🔬 DGA Analysis Portal                      ║"
echo "║          Physics-Informed Fault Detection            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Check if models exist
if [ ! -f "../dga_model_v2.joblib" ]; then
    echo "⚠️  Warning: RF v2 model not found"
fi

if [ ! -f "../pinn_dga_model.pt" ]; then
    echo "⚠️  Warning: PINN model not found"
fi

echo "Starting server..."
echo "Open http://localhost:5000 in your browser"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python app.py
