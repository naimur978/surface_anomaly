#!/usr/bin/env python3
"""
PatchCore Anomaly Detection - Inference Entrypoint

Run: python infer.py --model <path> --image <path> --visualize
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import main

if __name__ == "__main__":
    main()
