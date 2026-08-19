"""
PatchCore Anomaly Detection Implementation
Complete anomaly detection pipeline using DINOv2 features and PatchCore method

This module provides a complete implementation of the PatchCore anomaly detection
method, including feature extraction, model training, and evaluation.

Usage:
    python patchcore_anomaly_detection.py
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import cv2
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, confusion_matrix,
    ConfusionMatrixDisplay, fbeta_score
)
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import time
import random

warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(log_file):
    """Setup logging configuration"""
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


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================
def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config):
    """Create necessary output directories"""
    output_config = config['output']
    directories = [
        output_config['results_dir'],
        output_config['figures_dir'],
        output_config['models_dir'],
        output_config['predictions_dir'],
        output_config['logs_dir'],
    ]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    return output_config


# ============================================================================
# UTILITIES
# ============================================================================
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get computation device with priority: CUDA > Metal (MacBook) > CPU"""
    # Priority 1: CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger = logging.getLogger(__name__)
        logger.info("Using CUDA (NVIDIA GPU)")
        return device

    # Priority 2: Metal Performance Shaders (MacBook GPU)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger = logging.getLogger(__name__)
        logger.info("Using Metal Performance Shaders (MacBook GPU)")
        return device

    # Fallback: CPU
    device = torch.device('cpu')
    logger = logging.getLogger(__name__)
    logger.info("Using CPU")
    return device


def print_gpu_info(logger, device):
    """Print GPU information"""
    if device.type == 'cuda':
        logger.info(f"Number of CUDA GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"GPU {i}: {name} ({memory:.2f} GB)")
    elif device.type == 'mps':
        logger.info("Metal Performance Shaders (MacBook GPU) available")
        logger.info("Note: Set MPS fallback for better compatibility")
        # Enable MPS fallback for operations not supported on GPU
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    else:
        logger.info("Using CPU only")


# ============================================================================
# DATASET
# ============================================================================
class MVTecDataset(Dataset):
    """
    Dataset for MVTec-style anomaly detection data.

    Expected directory structure:
        root/category/train/good/*.png
        root/category/test/good/*.png
        root/category/test/broken_*/*.png
        root/category/ground_truth/broken_*/*_mask.png
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, root, category, split="train", crop_size=224,
                 img_pad_color=0, mask_pad_color=0):
        """
        Initialize dataset.

        Args:
            root: Root directory path
            category: Product category
            split: 'train' or 'test'
            crop_size: Size to crop images to
            img_pad_color: Padding color for images
            mask_pad_color: Padding color for masks
        """
        self.root = Path(root) / category / split
        self.mask_root = Path(root) / category / "ground_truth"
        self.split = split
        self.crop_size = crop_size

        self.transform = self._build_transform(crop_size, img_pad_color, normalize=True)
        self.mask_transform = self._build_transform(crop_size, mask_pad_color, normalize=False)
        self.samples = self._build_samples()
        self._print_summary(category)

    def _pad_to_square(self, img, pad_color):
        """Pad image to square"""
        w, h = img.size
        size = max(w, h)
        new_img = Image.new(img.mode, (size, size), pad_color)
        new_img.paste(img, ((size - w) // 2, (size - h) // 2))
        return new_img

    def _build_transform(self, crop_size, pad_color, normalize):
        """Build image transform pipeline"""
        ops = [
            transforms.Lambda(lambda img: self._pad_to_square(img, pad_color)),
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
        ]
        if normalize:
            ops.append(transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD))
        return transforms.Compose(ops)

    def _resolve_mask_path(self, class_name, img_path):
        """Resolve mask path for defective images"""
        mask_path = self.mask_root / class_name / (img_path.stem + "_mask.png")
        if not mask_path.exists():
            mask_path = self.mask_root / class_name / img_path.name
        return mask_path if mask_path.exists() else None

    def _build_samples(self):
        """Build sample list"""
        samples = []
        if not self.root.exists():
            return samples

        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = 0 if class_dir.name == "good" else 1
            for img_path in sorted(class_dir.glob("*.png")):
                mask_path = self._resolve_mask_path(class_dir.name, img_path) if label == 1 else None
                samples.append((img_path, mask_path, label))
        return samples

    def _print_summary(self, category):
        """Print dataset summary"""
        n_normal = sum(1 for _, _, l in self.samples if l == 0)
        n_defect = sum(1 for _, _, l in self.samples if l == 1)
        print(f"Dataset '{self.split}' [{category}]: {n_normal} normal | {n_defect} defective")

    def __len__(self):
        return len(self.samples)

    def _load_mask(self, mask_path, img_t):
        """Load mask for defective images"""
        if mask_path is None:
            return torch.zeros(1, img_t.shape[1], img_t.shape[2])
        mask = Image.open(mask_path).convert("L")
        mask_t = self.mask_transform(mask)
        return (mask_t > 0.5).float()

    def __getitem__(self, idx):
        """Get dataset item"""
        img_path, mask_path, label = self.samples[idx]
        img_t = self.transform(Image.open(img_path).convert("RGB"))
        mask_t = self._load_mask(mask_path, img_t)
        return img_t, mask_t, label, str(img_path)


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
class FeatureExtractor(nn.Module):
    """Extract features using DINOv2"""

    def __init__(self, device, model_name='dinov2_vitb14', logger=None):
        """
        Initialize feature extractor.

        Args:
            device: Torch device
            model_name: Name of DINOv2 model
            logger: Logger instance
        """
        super().__init__()
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        self.logger.info(f"Loading {model_name}...")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(device).eval()

        for p in self.model.parameters():
            p.requires_grad = False

        self.feature_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size
        self.logger.info(f"{model_name} loaded successfully")

    @torch.no_grad()
    def extract(self, imgs):
        """
        Extract patch features.

        Args:
            imgs: Input images (batch)

        Returns:
            Patch features and spatial grid dimensions
        """
        imgs = imgs.to(self.device)
        B, C, H, W = imgs.shape
        gh, gw = H // self.patch_size, W // self.patch_size
        tokens = self.model.forward_features(imgs)['x_norm_patchtokens']
        return tokens.cpu(), (gh, gw)


# ============================================================================
# PATCHCORE MODEL
# ============================================================================
class PatchCore:
    """
    PatchCore anomaly detection model.

    Uses patch-level features and KNN-based anomaly scoring with
    coreset-based memory bank selection for efficiency.
    """

    def __init__(self, coreset_ratio=0.10, n_neighbors=9, device='cuda', logger=None):
        """
        Initialize PatchCore model.

        Args:
            coreset_ratio: Ratio of patches to keep in memory bank
            n_neighbors: Number of nearest neighbors for scoring
            device: Torch device
            logger: Logger instance
        """
        self.coreset_ratio = coreset_ratio
        self.n_neighbors = n_neighbors
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.memory_bank = None
        self.hw_shape = None

    def fit(self, all_patches, hw_shape, seed=42):
        """
        Fit model with patch features.

        Args:
            all_patches: All patch features
            hw_shape: Spatial dimensions
            seed: Random seed for reproducibility
        """
        self.hw_shape = hw_shape
        feats = all_patches if isinstance(all_patches, torch.Tensor) else torch.from_numpy(all_patches)

        self.logger.info(f"Initial memory bank: {len(feats):,} vectors, dim {feats.shape[1]}")
        n_keep = max(1, int(len(feats) * self.coreset_ratio))
        self.logger.info(f"Applying coreset (ratio={self.coreset_ratio*100:.0f}%), keeping {n_keep:,} vectors")

        selected_idx = self._greedy_coreset(feats, n_keep, seed)
        self.memory_bank = feats[selected_idx].to(self.device)

        self.logger.info(f"Final memory bank: {len(self.memory_bank):,} vectors")
        self.logger.info(f"Reduction: {len(feats)/len(self.memory_bank):.0f}x fewer vectors")

    def _greedy_coreset(self, feats, n_keep, seed=42):
        """Select coreset using random sampling"""
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(feats), size=n_keep, replace=False)
        self.logger.info(f"  Coreset: Random sampling {n_keep}/{len(feats)} vectors")
        return selected

    def _to_tensor(self, patch_features):
        """Convert to tensor"""
        if isinstance(patch_features, np.ndarray):
            return torch.from_numpy(patch_features)
        return patch_features

    def _compute_patch_scores(self, patch_features):
        """Compute anomaly scores for patches"""
        x = self._to_tensor(patch_features).to(self.device)
        dists = torch.cdist(x, self.memory_bank)
        knn_dists, _ = torch.topk(dists, self.n_neighbors, dim=1, largest=False)
        return knn_dists[:, 0].cpu().numpy()

    def score_image(self, patch_features):
        """Score entire image"""
        scores = self._compute_patch_scores(patch_features)
        return float(scores.max())

    def score_map(self, patch_features, heatmap_size=224, smoothing_sigma=4):
        """Generate anomaly score heatmap"""
        H, W = self.hw_shape
        scores = self._compute_patch_scores(patch_features).reshape(H, W)
        scores_up = cv2.resize(scores, (heatmap_size, heatmap_size),
                              interpolation=cv2.INTER_LINEAR)
        return gaussian_filter(scores_up, sigma=smoothing_sigma)


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
def extract_all_patches(loader, extractor, logger=None):
    """
    Extract features from all images.

    Args:
        loader: Data loader
        extractor: Feature extractor
        logger: Logger instance

    Returns:
        All patch features and spatial grid dimensions
    """
    logger = logger or logging.getLogger(__name__)
    all_patches = []
    hw_shape = None

    for batch_idx, (imgs, _, _, _) in enumerate(loader):
        patches, hw = extractor.extract(imgs)
        hw_shape = hw
        B, HW, C = patches.shape
        all_patches.append(patches.reshape(-1, C))

        if batch_idx % 5 == 0:
            logger.info(f"Batch {batch_idx+1}/{len(loader)} extracted")

    return torch.cat(all_patches, dim=0), hw_shape


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================
def compute_image_scores(loader, extractor, model, logger=None):
    """Compute anomaly scores for all images"""
    logger = logger or logging.getLogger(__name__)
    scores = []
    for imgs, _, _, _ in tqdm(loader, desc="Computing scores"):
        patches, hw = extractor.extract(imgs)
        B, HW, C = patches.shape
        for i in range(B):
            scores.append(model.score_image(patches[i].numpy()))
    return scores


def collect_predictions(loader, extractor, model, logger=None):
    """Collect predictions and score maps"""
    logger = logger or logging.getLogger(__name__)
    scores, labels, paths = [], [], []
    score_maps, masks_list = [], []

    for imgs, masks, lbls, pths in tqdm(loader, desc="Collecting predictions"):
        patches, hw = extractor.extract(imgs)
        B, HW, C = patches.shape
        for i in range(B):
            patch_i = patches[i].numpy()
            scores.append(model.score_image(patch_i))
            score_maps.append(model.score_map(patch_i))
            labels.append(lbls[i].item())
            paths.append(pths[i])
            masks_list.append(masks[i, 0].numpy())

    return np.array(scores), np.array(labels), paths, score_maps, masks_list


def compute_metrics(scores, labels, score_maps, masks_list, threshold):
    """Compute evaluation metrics"""
    auroc_image = roc_auc_score(labels, scores)
    gt_flat = np.concatenate([m.flatten() for m in masks_list])
    pred_flat = np.concatenate([s.flatten() for s in score_maps])
    auroc_pixel = roc_auc_score((gt_flat > 0.5).astype(int), pred_flat)
    preds_binary = (scores > threshold).astype(int)
    f1 = f1_score(labels, preds_binary, zero_division=0)
    return auroc_image, auroc_pixel, f1


def find_threshold_for_recall(scores, labels, target_recall=1.0):
    """Find threshold for target recall"""
    thresholds = np.sort(scores)
    best_t = thresholds[0]
    for t in thresholds:
        preds = (scores > t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if recall >= target_recall:
            best_t = t
    return best_t


def measure_inference_time(dataset, extractor, model, device, n_warmup=5, n_samples=10):
    """Measure inference time"""
    indices = random.sample(range(len(dataset)), n_samples)
    samples = [dataset[i][0].unsqueeze(0) for i in indices]
    extractor.model = extractor.model.to(device)
    extractor.device = device

    for i in range(n_warmup):
        img = samples[i % len(samples)]
        patches, hw = extractor.extract(img)
        _ = model.score_image(patches[0])

    times = []
    for img in samples:
        start = time.perf_counter()
        patches, hw = extractor.extract(img)
        _ = model.score_image(patches[0])
        end = time.perf_counter()
        times.append(end - start)

    return np.array(times)


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_roc_curve(ax, fpr, tpr, auroc_image):
    """Plot ROC curve"""
    ax.plot(fpr, tpr, color="#2E86AB", lw=2.5, label=f"AUROC = {auroc_image:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2E86AB")
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=11)
    ax.set_title("ROC Curve - Image Level", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])


def plot_score_distribution(ax, test_scores, test_labels, threshold):
    """Plot score distribution"""
    normal_scores = test_scores[test_labels == 0]
    defect_scores = test_scores[test_labels == 1]
    ax.hist(normal_scores, bins=30, alpha=0.7, color="#27AE60", label="Normal", density=True)
    ax.hist(defect_scores, bins=30, alpha=0.7, color="#E74C3C", label="Defective", density=True)
    ax.axvline(threshold, color="black", lw=2, linestyle="--", label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Anomaly score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Anomaly Score Distribution", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)


def plot_evaluation(test_scores, test_labels, auroc_image, threshold, category,
                   output_dir="./results/figures", dpi=150):
    """Plot evaluation metrics"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(test_labels, test_scores)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=dpi)

    plot_roc_curve(axes[0], fpr, tpr, auroc_image)
    plot_score_distribution(axes[1], test_scores, test_labels, threshold)

    plt.tight_layout()
    save_path = Path(output_dir) / "evaluation_roc.png"
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return save_path


def plot_confusion_matrix(labels, preds, output_dir="./results/figures", dpi=150):
    """Plot confusion matrix"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    disp = ConfusionMatrixDisplay(cm, display_labels=["normal", "defective"])
    disp.plot(cmap="Blues", values_format="d", ax=ax)

    save_path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    return save_path


# ============================================================================
# RESULTS SAVING
# ============================================================================
def save_metrics(metrics, output_dir="./results"):
    """Save evaluation metrics to JSON"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metrics_clean = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                     for k, v in metrics.items()}

    save_path = Path(output_dir) / "metrics.json"
    with open(save_path, 'w') as f:
        json.dump(metrics_clean, f, indent=4)

    return save_path


def save_predictions(scores, labels, paths, output_dir="./results/predictions"):
    """Save predictions to CSV"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        'path': paths,
        'score': scores,
        'label': labels,
    })

    save_path = Path(output_dir) / "predictions.csv"
    df.to_csv(save_path, index=False)

    return save_path


def save_model(model, output_dir="./results/models", category="surface"):
    """Save trained model"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    save_path = Path(output_dir) / f"patchcore_{category}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)

    return save_path


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main(config_path="config.yaml"):
    """
    Main training and evaluation pipeline.

    Args:
        config_path: Path to configuration file
    """
    start_time = datetime.now()

    # Load configuration
    config = load_config(config_path)
    output_config = setup_directories(config)

    # Setup logging
    log_file = Path(output_config['logs_dir']) / "patchcore.log"
    logger = setup_logging(log_file)

    # Setup
    set_seed(config['training']['seed'])
    device = get_device()

    logger.info("=" * 70)
    logger.info("PATCHCORE ANOMALY DETECTION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Category: {config['data']['category']}")
    logger.info(f"Device: {device}")
    logger.info(f"Data root: {config['data']['root_dir']}")

    print_gpu_info(logger, device)

    # Load datasets
    logger.info("\n[1/6] Loading datasets...")
    train_dataset = MVTecDataset(
        config['data']['root_dir'],
        config['data']['category'],
        split=config['data']['train_split'],
        crop_size=config['image']['crop_size']
    )
    test_dataset = MVTecDataset(
        config['data']['root_dir'],
        config['data']['category'],
        split=config['data']['test_split'],
        crop_size=config['image']['crop_size']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")

    # Feature extraction
    logger.info("\n[2/6] Initializing feature extractor...")
    extractor = FeatureExtractor(
        device=device,
        model_name=config['model']['feature_extractor'],
        logger=logger
    )

    logger.info("\n[3/6] Extracting features from training set...")
    all_train_patches, hw_shape = extract_all_patches(train_loader, extractor, logger)
    logger.info(f"Total patches: {len(all_train_patches):,}")
    logger.info(f"Patch dimension: {all_train_patches.shape[1]}")

    # Model training
    logger.info("\n[4/6] Training PatchCore model...")
    model = PatchCore(
        coreset_ratio=config['model']['coreset_ratio'],
        n_neighbors=config['model']['n_neighbors'],
        device=device,
        logger=logger
    )
    model.fit(all_train_patches, hw_shape, seed=config['training']['seed'])

    # Save model
    model_path = save_model(model, output_config['models_dir'], config['data']['category'])
    logger.info(f"Model saved to {model_path}")

    # Compute threshold
    logger.info("\nComputing threshold...")
    train_scores = compute_image_scores(train_loader, extractor, model, logger)
    threshold = np.percentile(train_scores, config['evaluation']['threshold_percentile'])
    logger.info(f"Train scores - min: {min(train_scores):.4f} | max: {max(train_scores):.4f}")
    logger.info(f"Threshold (percentile {config['evaluation']['threshold_percentile']}): {threshold:.4f}")

    # Evaluation
    logger.info("\n[5/6] Evaluating on test set...")
    test_scores, test_labels, test_paths, all_score_maps, all_masks = collect_predictions(
        test_loader, extractor, model, logger
    )
    auroc_image, auroc_pixel, f1 = compute_metrics(
        test_scores, test_labels, all_score_maps, all_masks, threshold
    )

    # Log results
    logger.info("\n" + "=" * 70)
    logger.info(f"RESULTS - {config['data']['category'].upper()}")
    logger.info("=" * 70)
    logger.info(f"AUROC Image-level: {auroc_image:.4f} ({auroc_image*100:.1f}%)")
    logger.info(f"AUROC Pixel-level: {auroc_pixel:.4f} ({auroc_pixel*100:.1f}%)")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Threshold: {threshold:.4f}")
    logger.info("=" * 70)

    # Recall threshold
    recall_threshold = find_threshold_for_recall(
        test_scores, test_labels,
        target_recall=config['evaluation']['target_recall']
    )
    preds_binary = (test_scores > recall_threshold).astype(int)
    f2_recall = fbeta_score(test_labels, preds_binary, beta=2, zero_division=0)

    logger.info(f"\nRecall threshold: {recall_threshold:.4f}")
    logger.info(f"F2 score at recall threshold: {f2_recall:.4f}")

    # Inference timing - only benchmark available devices
    logger.info("\n" + "=" * 70)
    logger.info("INFERENCE TIME BENCHMARKS")
    logger.info("=" * 70)

    benchmark_results = []

    # Only test available devices
    if torch.cuda.is_available():
        try:
            logger.info("Measuring inference time on CUDA...")
            cuda_times = measure_inference_time(
                test_dataset, extractor, model,
                device=torch.device("cuda"),
                n_warmup=config['inference']['warmup_samples'],
                n_samples=config['inference']['benchmark_samples']
            )
            benchmark_results.append({
                "device": "cuda",
                "mean_ms": cuda_times.mean() * 1000,
                "std_ms": cuda_times.std() * 1000,
                "min_ms": cuda_times.min() * 1000,
                "max_ms": cuda_times.max() * 1000,
            })
        except Exception as e:
            logger.warning(f"CUDA benchmark failed: {str(e)}")

    # Try CPU benchmark
    try:
        logger.info("Measuring inference time on CPU...")
        cpu_times = measure_inference_time(
            test_dataset, extractor, model,
            device=torch.device("cpu"),
            n_warmup=config['inference']['warmup_samples'],
            n_samples=config['inference']['benchmark_samples']
        )
        benchmark_results.append({
            "device": "cpu",
            "mean_ms": cpu_times.mean() * 1000,
            "std_ms": cpu_times.std() * 1000,
            "min_ms": cpu_times.min() * 1000,
            "max_ms": cpu_times.max() * 1000,
        })
    except Exception as e:
        logger.warning(f"CPU benchmark failed: {str(e)}")

    if benchmark_results:
        summary_df = pd.DataFrame(benchmark_results)
        logger.info("\n" + summary_df.to_string(index=False))

    # Save results
    logger.info("\n[6/6] Saving results...")
    metrics = {
        'auroc_image': float(auroc_image),
        'auroc_pixel': float(auroc_pixel),
        'f1': float(f1),
        'threshold': float(threshold),
        'recall_threshold': float(recall_threshold),
        'f2_recall': float(f2_recall),
    }
    save_metrics(metrics, output_config['results_dir'])

    if config['inference']['save_predictions']:
        save_predictions(test_scores, test_labels, test_paths, output_config['predictions_dir'])

    # Visualizations
    logger.info("\nGenerating visualizations...")

    fig_config = config['figures']

    plot_evaluation(
        test_scores, test_labels, auroc_image, recall_threshold,
        config['data']['category'],
        output_dir=output_config['figures_dir'],
        dpi=fig_config['dpi']
    )
    logger.info(f"Saved ROC curve to {output_config['figures_dir']}/evaluation_roc.png")

    preds_recall = (test_scores > recall_threshold).astype(int)
    plot_confusion_matrix(
        test_labels, preds_recall,
        output_dir=output_config['figures_dir'],
        dpi=fig_config['dpi']
    )
    logger.info(f"Saved confusion matrix to {output_config['figures_dir']}/confusion_matrix.png")

    # Duration
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60

    logger.info("\n" + "=" * 70)
    logger.info(f"PIPELINE COMPLETED")
    logger.info(f"Total execution time: {duration:.2f} minutes")
    logger.info("=" * 70)

    return {
        'model': model,
        'extractor': extractor,
        'metrics': metrics,
        'predictions': {
            'scores': test_scores,
            'labels': test_labels,
            'paths': test_paths,
            'score_maps': all_score_maps,
        },
        'config': config,
    }


if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    results = main(config_file)
