#!/bin/bash
# Train and Inference Pipeline
# Runs training followed by inference sequentially

set -e  # Exit on error

echo "=========================================="
echo "Surface Anomaly Detection - Train & Infer"
echo "=========================================="

# Check if config exists
CONFIG_FILE="${1:-config/config.yaml}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if data exists
if [ ! -d "data/surface" ]; then
    echo "Error: Data directory not found: data/surface"
    exit 1
fi

echo ""
echo "Step 1: Starting Training..."
echo "=========================================="
python scripts/train.py "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
    echo ""
    echo "Step 2: Starting Inference..."
    echo "=========================================="
    python scripts/inference.py \
        --model results/models/patchcore_surface.pkl \
        --folder data/surface/test \
        --output ./results/inference_latest \
        --visualize

    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "Pipeline completed successfully!"
        echo "=========================================="
        echo ""
        echo "View results in MLflow:"
        echo "  mlflow ui --backend-store-uri ./mlruns"
        echo ""
    else
        echo "Error: Inference failed"
        exit 1
    fi
else
    echo "Error: Training failed"
    exit 1
fi
