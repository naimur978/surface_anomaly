"""
Configuration and setup utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import torch
import random
import numpy as np


def setup_logging(log_file: Path) -> logging.Logger:
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


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_directories(config: Dict[str, Any]) -> Dict[str, str]:
    """Create necessary output directories."""
    output_config = config['output']
    for dir_key in ['results_dir', 'figures_dir', 'models_dir', 'predictions_dir', 'logs_dir']:
        Path(output_config[dir_key]).mkdir(parents=True, exist_ok=True)
    return output_config


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_gpu_info(logger: logging.Logger, device: torch.device) -> None:
    """Print GPU information if available."""
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif device.type == 'mps':
        logger.info("Using Metal Performance Shaders (MacBook GPU)")
    else:
        logger.info("Using CPU")
