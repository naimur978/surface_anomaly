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
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

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
# DATASET
# ============================================================================
class MVTecDataset(Dataset):
    """MVTec dataset loader with preprocessing."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, root, category, split="train", crop_size=224):
        self.root = Path(root) / category / split
        self.mask_root = Path(root) / category / "ground_truth"
        self.split = split
        self.transform = self._build_transform(crop_size, normalize=True)
        self.mask_transform = self._build_transform(crop_size, normalize=False)
        self.samples = self._build_samples()
        self._print_summary(category)

    def _pad_to_square(self, img, pad_color=0):
        """Pad image to square."""
        w, h = img.size
        size = max(w, h)
        new_img = Image.new(img.mode, (size, size), pad_color)
        new_img.paste(img, ((size - w) // 2, (size - h) // 2))
        return new_img

    def _build_transform(self, crop_size, normalize):
        """Build image transformation pipeline."""
        ops = [
            transforms.Lambda(lambda img: self._pad_to_square(img, 0)),
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
        ]
        if normalize:
            ops.append(transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD))
        return transforms.Compose(ops)

    def _resolve_mask_path(self, class_name, img_path):
        """Find mask file for defective image."""
        mask_path = self.mask_root / class_name / (img_path.stem + "_mask.png")
        return mask_path if mask_path.exists() else None

    def _build_samples(self):
        """Build list of (image_path, mask_path, label) tuples."""
        samples = []
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = 0 if class_dir.name == "good" else 1
            for img_path in sorted(class_dir.glob("*.png")):
                mask_path = self._resolve_mask_path(class_dir.name, img_path) if label == 1 else None
                samples.append((img_path, mask_path, label))
        return samples

    def _print_summary(self, category):
        """Print dataset summary (suppressed via StringIO in training)."""
        n_normal = sum(1 for _, _, l in self.samples if l == 0)
        n_defect = sum(1 for _, _, l in self.samples if l == 1)
        print(f"Dataset '{self.split}' [{category}]: {n_normal} normal | {n_defect} defective")

    def __len__(self):
        return len(self.samples)

    def _load_mask(self, mask_path, img_t):
        """Load ground truth mask."""
        if mask_path is None:
            return torch.zeros(1, img_t.shape[1], img_t.shape[2])
        mask = Image.open(mask_path).convert("L")
        mask_t = self.mask_transform(mask)
        return (mask_t > 0.5).float()

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        img_t = self.transform(Image.open(img_path).convert("RGB"))
        mask_t = self._load_mask(mask_path, img_t)
        return img_t, mask_t, label, str(img_path)


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
