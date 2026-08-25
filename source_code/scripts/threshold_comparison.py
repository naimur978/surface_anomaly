#!/usr/bin/env python3
"""
Generate a side-by-side comparison of 99th percentile vs 100% recall thresholds.

Run: python scripts/threshold_comparison.py [--config config/config.yaml] [--output <path>]

Requires a trained model (run scripts/train.py first). Writes a comparison
figure to --output and per-threshold metrics to <output_dir>/threshold_metrics.json.
"""

import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.dataset import MVTecDataset
from src.check_device import get_device
from src.models import FeatureExtractor
from src.metrics import compute_confusion_metrics
from src.training import collect_predictions, compute_image_scores


def find_threshold_for_recall(scores, labels, target_recall=1.0):
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


def plot_threshold_comparison(test_scores, test_labels, percentile_threshold, recall_threshold, output_path="results/figures/threshold_comparison.png"):
    """Create side-by-side confusion matrix comparison of the two thresholds."""
    cm_percentile = compute_confusion_metrics(test_labels, test_scores, percentile_threshold)["cm"]
    cm_recall = compute_confusion_metrics(test_labels, test_scores, recall_threshold)["cm"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: 99th percentile threshold
    disp_percentile = ConfusionMatrixDisplay(cm_percentile, display_labels=["Normal", "Defective"])
    disp_percentile.plot(cmap="Blues", values_format="d", ax=axes[0], colorbar=True)
    axes[0].set_title(f"99th Percentile\n(threshold={percentile_threshold:.4f})", fontsize=12, fontweight="bold")

    # Right: 100% recall threshold
    disp_recall = ConfusionMatrixDisplay(cm_recall, display_labels=["Normal", "Defective"])
    disp_recall.plot(cmap="Greens", values_format="d", ax=axes[1], colorbar=True)
    axes[1].set_title(f"100% Recall\n(threshold={recall_threshold:.4f})", fontsize=12, fontweight="bold")

    plt.tight_layout()

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[OK] Saved threshold comparison to {output_path}")
    plt.close()


def compute_metrics_for_threshold(test_scores, test_labels, threshold):
    """Compute confusion-matrix metrics for a given threshold.

    Thin wrapper around src.metrics.compute_confusion_metrics that adds the
    threshold value and strips the raw confusion matrix array, which isn't
    JSON-serializable and isn't needed by this script's output.
    """
    metrics = compute_confusion_metrics(test_labels, test_scores, threshold)
    metrics.pop("cm")
    metrics["threshold"] = float(threshold)
    return {key: (float(val) if isinstance(val, (float, np.floating)) else int(val))
            if key != "threshold" else val
            for key, val in metrics.items()}


def print_metrics(metrics, label_width=12):
    """Print threshold metrics, skipping the threshold value itself (printed separately)."""
    for key, val in metrics.items():
        if key == "threshold":
            continue
        formatted = f"{val:.4f}" if isinstance(val, float) else str(val)
        print(f"  {key:<{label_width}s}: {formatted}")


def load_trained_model(config, device):
    """Load the trained PatchCore model matching the current config."""
    feature_extractor = config['model']['feature_extractor']
    models_dir = Path(config['output']['models_dir'])
    model_path = models_dir / f"anomaly_localization_{config['data']['category']}_{feature_extractor}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run scripts/train.py first."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    extractor = FeatureExtractor(device=device, model_name=feature_extractor)
    return model, extractor


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate threshold comparison visualization")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--output", default="results/figures/threshold_comparison.png", help="Output image path")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    device = get_device()

    # Load trained model + feature extractor
    print("Loading trained model...")
    model, extractor = load_trained_model(config, device)

    # Build data loaders
    train_dataset = MVTecDataset(
        config['data']['root_dir'], config['data']['category'], split='train',
        crop_size=config['image']['crop_size'],
        apply_roi_mask=config['image'].get('apply_roi_mask', False)
    )
    test_dataset = MVTecDataset(
        config['data']['root_dir'], config['data']['category'], split='test',
        crop_size=config['image']['crop_size'],
        apply_roi_mask=config['image'].get('apply_roi_mask', False)
    )
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # Run inference to get scores
    print("Running inference on test set...")
    test_scores, test_labels, test_paths, score_maps, masks = collect_predictions(test_loader, extractor, model)

    # Compute thresholds (99th percentile is computed on normal training scores,
    # matching the convention used in src/training.py's evaluate_model)
    train_scores = np.array(compute_image_scores(train_loader, extractor, model))
    percentile_99_threshold = np.percentile(train_scores, config['evaluation'].get('threshold_percentile', 99))
    recall_100_threshold = find_threshold_for_recall(
        test_scores, test_labels, target_recall=config['evaluation'].get('target_recall', 1.0)
    )

    print(f"\n{'='*60}")
    print("THRESHOLD COMPARISON")
    print(f"{'='*60}")
    print(f"99th Percentile Threshold:  {percentile_99_threshold:.4f}")
    print(f"100% Recall Threshold:      {recall_100_threshold:.4f}")
    print(f"Difference:                 {abs(recall_100_threshold - percentile_99_threshold):.4f}")
    print(f"{'='*60}\n")

    # Compute metrics for each threshold
    print("METRICS AT 99TH PERCENTILE:")
    metrics_99 = compute_metrics_for_threshold(test_scores, test_labels, percentile_99_threshold)
    print_metrics(metrics_99)

    print("\nMETRICS AT 100% RECALL:")
    metrics_recall = compute_metrics_for_threshold(test_scores, test_labels, recall_100_threshold)
    print_metrics(metrics_recall)

    # Generate visualization
    print("\nGenerating visualization...")
    plot_threshold_comparison(test_scores, test_labels, percentile_99_threshold, recall_100_threshold, args.output)

    # Save metrics to JSON
    metrics_path = Path(args.output).parent / "threshold_metrics.json"
    metrics_data = {
        "percentile_99": metrics_99,
        "recall_100": metrics_recall,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[OK] Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
