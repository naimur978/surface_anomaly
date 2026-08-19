#!/usr/bin/env python3
"""
PatchCore Anomaly Detection - Data Validation Entrypoint

Run: python validate.py ./data surface
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils_data import print_dataset_stats, verify_dataset_integrity

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python validate.py <root_dir> <category>")
        print("Example: python validate.py ./data surface")
        sys.exit(1)

    root_dir = sys.argv[1]
    category = sys.argv[2]

    print_dataset_stats(root_dir, category)
    is_valid = verify_dataset_integrity(root_dir, category)

    if is_valid:
        print("✓ Dataset validation passed!")
    else:
        print("✗ Dataset validation failed!")
        sys.exit(1)
