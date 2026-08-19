#!/usr/bin/env python3
"""
PatchCore Anomaly Detection - Main Entrypoint

Run: python run.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.main import main

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    results = main(config_file)
