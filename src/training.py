"""
Training pipeline for PatchCore model.
"""

import logging
from pathlib import Path
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
from .visualization import plot_evaluation
from .tracing import ExecutionTracer


def extract_all_patches(loader, extractor, logger=None):
    """Extract all patches from training images."""
    all_patches = []
    hw_shape = None
    for batch_idx, (imgs, _, _, _) in enumerate(loader):
        patches, hw = extractor.extract(imgs)
        hw_shape = hw
        B, HW, C = patches.shape
        all_patches.append(patches.reshape(-1, C))

        if logger and batch_idx % 5 == 0:
            logger.info(f"  Processed {batch_idx + 1}/{len(loader)} batches")

    return torch.cat(all_patches, dim=0), hw_shape


def compute_image_scores(loader, extractor, model, logger=None):
    """Compute anomaly scores for all images."""
    scores = []
    for imgs, _, _, _ in tqdm(loader, desc="Scoring", disable=not logger):
        patches, _ = extractor.extract(imgs)
        B, HW, C = patches.shape
        for i in range(B):
            scores.append(model.score_image(patches[i].numpy()))
    return scores


def collect_predictions(loader, extractor, model, logger=None):
    """Collect predictions and score maps for evaluation."""
    scores, labels, paths = [], [], []
    score_maps, masks_list = [], []

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
    train_loader, val_loader, device, config,
    output_config, logger
):
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
            logger=logger
        )
        model.fit(all_train_patches, hw_shape)

    return model, extractor


def evaluate_model(val_loader, extractor, model, train_loader, config, output_config, logger):
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


def save_results(model, metrics, val_scores, val_labels, output_config, config, logger):
    """Save model, metrics, and visualizations."""
    logger.info(f"\nSaving outputs...")

    # Save model
    model_path = Path(output_config['models_dir']) / f"patchcore_{config['data']['category']}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"  Model: {model_path}")

    # Save metrics
    save_metrics(metrics, output_config['results_dir'])
    logger.info(f"  Metrics: {Path(output_config['results_dir']) / 'metrics.json'}")

    # Visualizations
    preds_cm = (val_scores > metrics['recall_threshold']).astype(int)
    plot_confusion_matrix(val_labels, preds_cm, output_config['figures_dir'])
    plot_evaluation(
        val_scores, val_labels, metrics['auroc_image'], metrics['recall_threshold'],
        f"{output_config['figures_dir']}/evaluation.png"
    )
