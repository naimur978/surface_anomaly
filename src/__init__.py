"""
PatchCore Anomaly Detection - Production Implementation

Modules:
    - models: FeatureExtractor and PatchCore model classes
    - metrics: Metric computation and evaluation utilities
    - visualization: Plotting functions for training/evaluation
    - main: Training pipeline
    - inference: Inference wrapper
    - utils_data: Data validation utilities
    - check_device: Device detection and selection
"""

from .models import FeatureExtractor, PatchCore
from .metrics import compute_metrics, find_threshold_for_recall
from .dataset import MVTecDataset
from .main import setup_logging

__version__ = "1.0.0"
__all__ = [
    "FeatureExtractor",
    "PatchCore",
    "MVTecDataset",
    "compute_metrics",
    "find_threshold_for_recall",
    "setup_logging",
]
