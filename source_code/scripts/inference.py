#!/usr/bin/env python3
"""
PatchCore Anomaly Detection - Inference Entrypoint

Run (single image):
    python scripts/inference.py --model <path/to/model.pkl> --image <path/to/image.png> --visualize

Run (folder of images, as used by run_pipeline.py):
    python scripts/inference.py --model <path/to/model.pkl> --config config/config.yaml \\
        --folder data/surface/test --output ./results/inference_latest --visualize
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.inference import main

if __name__ == "__main__":
    main()
