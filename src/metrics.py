"""
Metrics computation and evaluation utilities.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, fbeta_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
from pathlib import Path


def compute_metrics(scores: np.ndarray, labels: np.ndarray, score_maps: List[np.ndarray], masks_list: List[np.ndarray], threshold: float) -> Tuple[float, float, float]:
    """Compute AUROC and F1 metrics."""
    auroc_image = roc_auc_score(labels, scores)

    gt_flat = np.concatenate([m.flatten() for m in masks_list])
    pred_flat = np.concatenate([s.flatten() for s in score_maps])
    auroc_pixel = roc_auc_score((gt_flat > 0.5).astype(int), pred_flat)

    preds_binary = (scores > threshold).astype(int)
    f1 = f1_score(labels, preds_binary, zero_division=0)

    return auroc_image, auroc_pixel, f1


def find_threshold_for_recall(scores: np.ndarray, labels: np.ndarray, target_recall: float = 1.0) -> float:
    """Find threshold that achieves target recall."""
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


def compute_confusion_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Any]:
    """Compute confusion matrix and derived metrics."""
    preds = (scores > threshold).astype(int)
    cm = confusion_matrix(labels, preds)

    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = f1_score(labels, preds, zero_division=0)
    f2 = fbeta_score(labels, preds, beta=2, zero_division=0)

    return {
        'cm': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'f2': f2
    }


def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, output_dir: str = "./results/figures", dpi: int = 150) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

    disp = ConfusionMatrixDisplay(cm, display_labels=["normal", "defective"])
    disp.plot(cmap="Blues", values_format="d", ax=ax)

    plt.savefig(Path(output_dir) / "confusion_matrix.png", dpi=dpi, bbox_inches="tight")
    plt.close()


def save_metrics(metrics_dict: Dict[str, Any], output_dir: str = "./results") -> None:
    """Save metrics to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics_dict, f, indent=4)
