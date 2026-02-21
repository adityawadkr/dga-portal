#!/usr/bin/env bash
set -e

# Install PyTorch CPU-only (saves ~1.5GB vs full CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
