"""
Training pipeline for PatchCore model.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pickle
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from .models import FeatureExtractor, PatchCore
from .metrics import (
    compute_metrics, find_threshold_for_recall,
    save_metrics, plot_confusion_matrix
)
from .visualization import (
    plot_evaluation, visualize_random_heatmaps,
    visualize_preprocessing_pipeline
)
from .tracing import ExecutionTracer


def extract_all_patches(loader: DataLoader, extractor: Any, logger: Optional[logging.Logger] = None) -> Tuple[torch.Tensor, Any]:
    """Extract all patches from training images."""
    all_patches: List[torch.Tensor] = []
    hw_shape: Optional[Any] = None
    for batch_idx, (imgs, _, _, _) in enumerate(loader):
        patches, hw = extractor.extract(imgs)
        hw_shape = hw
        B, HW, C = patches.shape
        all_patches.append(patches.reshape(-1, C))

        if logger and batch_idx % 5 == 0:
            logger.info(f"  Processed {batch_idx + 1}/{len(loader)} batches")

    return torch.cat(all_patches, dim=0), hw_shape


def compute_image_scores(loader: DataLoader, extractor: Any, model: Any, logger: Optional[logging.Logger] = None) -> List[float]:
    """Compute anomaly scores for all images."""
    scores: List[float] = []
    for imgs, _, _, _ in tqdm(loader, desc="Scoring", disable=not logger):
        patches, _ = extractor.extract(imgs)
        B, HW, C = patches.shape
        for i in range(B):
            scores.append(model.score_image(patches[i].numpy()))
    return scores


def collect_predictions(loader: DataLoader, extractor: Any, model: Any, logger: Optional[logging.Logger] = None) -> Tuple[np.ndarray, np.ndarray, List[str], List[np.ndarray], List[np.ndarray]]:
    """Collect predictions and score maps for evaluation."""
    scores: List[float] = []
    labels: List[int] = []
    paths: List[str] = []
    score_maps: List[np.ndarray] = []
    masks_list: List[np.ndarray] = []

    for imgs, masks, lbls, pths in tqdm(loader, desc="Evaluating", disable=not logger):
        patches, _ = extractor.extract(imgs)
        B, HW, C = patches.shape
        for i in range(B):
            patch_i = patches[i].numpy()
            scores.append(model.score_image(patch_i))
            score_maps.append(model.score_map(patch_i))
            labels.append(lbls[i].item())
            paths.append(pths[i])
            masks_list.append(masks[i, 0].numpy())

    return np.array(scores), np.array(labels), paths, score_maps, masks_list


def train_patchcore(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    output_config: Dict[str, str],
    logger: logging.Logger
) -> Tuple[Any, Any]:
    """Train PatchCore model."""
    with ExecutionTracer("Feature Extraction"):
        logger.info("\nExtracting features...")
        model_name = config['model'].get('feature_extractor', 'dinov2_vitb14')
        extractor = FeatureExtractor(device=device, model_name=model_name, logger=logger)

        all_train_patches, hw_shape = extract_all_patches(train_loader, extractor, logger)
        logger.info(f"Training patches: {len(all_train_patches):,} vectors")

    # Train model
    with ExecutionTracer("Model Training"):
        logger.info("\nTraining PatchCore...")
        model = PatchCore(
            coreset_ratio=config['model']['coreset_ratio'],
            n_neighbors=config['model'].get('n_neighbors', 9),
            device=device,
            logger=logger,
            heatmap_smoothing=config['evaluation'].get('heatmap_smoothing', 4.0)
        )
        model.fit(all_train_patches, hw_shape)

    return model, extractor


def evaluate_model(
    val_loader: DataLoader,
    extractor: Any,
    model: Any,
    train_loader: DataLoader,
    config: Dict[str, Any],
    output_config: Dict[str, str],
    logger: logging.Logger
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Evaluate model and compute metrics."""
    with ExecutionTracer("Evaluation"):
        logger.info("\nEvaluating on validation set...")
        val_scores, val_labels, val_paths, val_score_maps, val_masks = collect_predictions(
            val_loader, extractor, model, logger
        )

        # Compute thresholds
        threshold_99_percentile = np.percentile(
            np.array(compute_image_scores(train_loader, extractor, model)), 99
        )
        threshold_100_recall = find_threshold_for_recall(val_scores, val_labels, target_recall=1.0)

        logger.info(f"\n99th Percentile Threshold: {threshold_99_percentile:.4f}")
        logger.info(f"100% Recall Threshold: {threshold_100_recall:.4f}")

        # Compute metrics
        auroc_image, auroc_pixel, f1 = compute_metrics(
            val_scores, val_labels, val_score_maps, val_masks, threshold_100_recall
        )

        # F2 score (recall-focused)
        from sklearn.metrics import fbeta_score
        preds_binary = (val_scores > threshold_100_recall).astype(int)
        f2 = fbeta_score(val_labels, preds_binary, beta=2, zero_division=0)

        metrics = {
            'auroc_image': float(auroc_image),
            'auroc_pixel': float(auroc_pixel),
            'f1': float(f1),
            'threshold': float(threshold_99_percentile),
            'recall_threshold': float(threshold_100_recall),
            'f2_recall': float(f2)
        }

        logger.info(f"\nResults:")
        logger.info(f"  AUROC (image): {auroc_image:.4f}")
        logger.info(f"  AUROC (pixel): {auroc_pixel:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  F2 Score (recall focus): {f2:.4f}")

    return metrics, val_scores, val_labels, val_score_maps, val_masks


def save_results(
    model: Any,
    metrics: Dict[str, float],
    val_scores: np.ndarray,
    val_labels: np.ndarray,
    val_score_maps: List[np.ndarray],
    val_dataset: Any,
    output_config: Dict[str, str],
    config: Dict[str, Any],
    logger: logging.Logger
) -> None:
    """Save model, metrics, and visualizations."""
    logger.info(f"\nSaving outputs...")

    # Save model with backbone name to keep separate trained models
    feature_extractor = config['model'].get('feature_extractor', 'dinov2_vitb14')
    model_name = f"anomaly_localization_{config['data']['category']}_{feature_extractor}.pkl"
    model_path = Path(output_config['models_dir']) / model_name
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"  Model: {model_path}")

    # Save metrics
    save_metrics(metrics, output_config['results_dir'])
    logger.info(f"  Metrics: {Path(output_config['results_dir']) / 'metrics.json'}")

    # Visualizations
    preds_cm = (val_scores > metrics['recall_threshold']).astype(int)
    plot_confusion_matrix(val_labels, preds_cm, output_config['figures_dir'])
    logger.info(f"  Confusion Matrix: {Path(output_config['figures_dir']) / 'confusion_matrix.png'}")

    plot_evaluation(
        val_scores, val_labels, metrics['auroc_image'], metrics['recall_threshold'],
        f"{output_config['figures_dir']}/evaluation.png"
    )
    logger.info(f"  Evaluation Plot: {Path(output_config['figures_dir']) / 'evaluation.png'}")

    # Random heatmap samples
    if val_score_maps:
        visualize_random_heatmaps(
            val_dataset, val_scores, val_labels, val_score_maps,
            metrics['recall_threshold'],
            n_samples=4,
            output_path=f"{output_config['figures_dir']}/random_heatmaps.png"
        )
        logger.info(f"  Random Heatmaps: {Path(output_config['figures_dir']) / 'random_heatmaps.png'}")

    # Preprocessing pipeline
    visualize_preprocessing_pipeline(
        val_dataset, n_samples=2,
        output_path=f"{output_config['figures_dir']}/preprocessing_pipeline.png"
    )
    logger.info(f"  Preprocessing Pipeline: {Path(output_config['figures_dir']) / 'preprocessing_pipeline.png'}")
