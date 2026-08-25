#!/usr/bin/env python3
"""
PatchCore Anomaly Detection - Training Entrypoint

Run: python scripts/train.py [config/config.yaml]

Trains a PatchCore model per config/config.yaml and writes it to
results/models/anomaly_localization_surface_<feature_extractor>.pkl.
Config path defaults to config/config.yaml if omitted.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.main import main

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    results = main(config_file)
