#!/usr/bin/env python3
"""
Generate a side-by-side comparison of 99th percentile vs 100% recall thresholds.

Run: python scripts/threshold_comparison.py [--config config/config.yaml] [--output <path>]

Requires a trained model (run scripts/train.py first). Writes a comparison
figure to --output and per-threshold metrics to <output_dir>/threshold_metrics.json.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.inference import run_inference
from config.config import load_config
from src.metrics import compute_confusion_metrics


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
    """Create side-by-side comparison of thresholds."""
    normal_scores = test_scores[test_labels == 0]
    defect_scores = test_scores[test_labels == 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: 99th percentile threshold
    axes[0].hist(normal_scores, bins=30, alpha=0.7, color="#27AE60", label="Normal", density=True)
    axes[0].hist(defect_scores, bins=30, alpha=0.7, color="#E74C3C", label="Defective", density=True)
    axes[0].axvline(percentile_threshold, color="black", lw=2.5, linestyle="--", label=f"99th Percentile = {percentile_threshold:.4f}")
    axes[0].set_xlabel("Anomaly score", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].set_title("99th Percentile Threshold", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # Right: 100% recall threshold
    axes[1].hist(normal_scores, bins=30, alpha=0.7, color="#27AE60", label="Normal", density=True)
    axes[1].hist(defect_scores, bins=30, alpha=0.7, color="#E74C3C", label="Defective", density=True)
    axes[1].axvline(recall_threshold, color="black", lw=2.5, linestyle="--", label=f"100% Recall = {recall_threshold:.4f}")
    axes[1].set_xlabel("Anomaly score", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].set_title("100% Recall Threshold", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

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


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate threshold comparison visualization")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--output", default="results/figures/threshold_comparison.png", help="Output image path")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Run inference to get scores
    print("Running inference on test set...")
    test_scores, test_labels, test_paths, score_maps, masks = run_inference(config)

    # Compute thresholds
    percentile_99_threshold = np.percentile(test_scores, 99)
    recall_100_threshold = find_threshold_for_recall(test_scores, test_labels, target_recall=1.0)

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
