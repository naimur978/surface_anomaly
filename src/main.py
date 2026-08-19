"""
PatchCore Anomaly Detection Training Orchestration

Main entry point that coordinates data loading, model training, and evaluation.
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
import json
import warnings

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
import random

from .dataset import MVTecDataset
from .training import train_patchcore, evaluate_model, save_results
from .check_device import get_device

warnings.filterwarnings('ignore')


# ============================================================================
# SETUP UTILITIES
# ============================================================================
def setup_logging(log_file):
    """Setup logging configuration."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_directories(config):
    """Create necessary output directories."""
    output_config = config['output']
    for dir_key in ['results_dir', 'figures_dir', 'models_dir', 'predictions_dir', 'logs_dir']:
        Path(output_config[dir_key]).mkdir(parents=True, exist_ok=True)
    return output_config


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_gpu_info(logger, device):
    """Print GPU information if available."""
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif device.type == 'mps':
        logger.info("Using Metal Performance Shaders (MacBook GPU)")
    else:
        logger.info("Using CPU")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================
def main(config_path="config.yaml"):
    """Main training pipeline."""
    # Setup
    config = load_config(config_path)
    set_seed(config['training'].get('seed', 42))
    output_config = setup_directories(config)

    log_file = Path(output_config['logs_dir']) / "training.log"
    logger = setup_logging(log_file)

    logger.info("=" * 60)
    logger.info("PATCHCORE ANOMALY DETECTION - TRAINING")
    logger.info("=" * 60)

    device = get_device(logger=logger)
    print_gpu_info(logger, device)

    # Load data (suppress dataset prints)
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()

    train_dataset = MVTecDataset(
        config['data']['root_dir'],
        config['data']['category'],
        split='train',
        crop_size=config['image']['crop_size']
    )
    val_dataset = MVTecDataset(
        config['data']['root_dir'],
        config['data']['category'],
        split='test',
        crop_size=config['image']['crop_size']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=config['training'].get('pin_memory', False)
    )
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    sys.stdout = old_stdout

    # Train
    model, extractor = train_patchcore(train_loader, val_loader, device, config, output_config, logger)

    # Evaluate
    metrics, val_scores, val_labels, _, _ = evaluate_model(
        val_loader, extractor, model, train_loader, config, output_config, logger
    )

    # Save
    save_results(model, metrics, val_scores, val_labels, output_config, config, logger)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    main(config_file)
