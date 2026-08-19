#!/usr/bin/env python3
"""
Compare confusion matrices for both thresholds:
- 99th Percentile Threshold
- 100% Recall Threshold
"""

import sys
import os
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def compare_thresholds(results_file, metrics_file, output_dir='./results/figures'):
    """Compare confusion matrices for both thresholds."""

    # Load results
    with open(results_file) as f:
        results = json.load(f)

    # Load metrics
    with open(metrics_file) as f:
        metrics = json.load(f)

    # Extract scores and labels
    scores = np.array([r['anomaly_score'] for r in results])
    true_labels = np.array([r['expected_label'] for r in results])

    # Both thresholds
    threshold_99_percentile = metrics['threshold']
    threshold_100_recall = metrics['recall_threshold']

    print("="*70)
    print("THRESHOLD COMPARISON")
    print("="*70)
    print(f"\n99th Percentile Threshold: {threshold_99_percentile:.4f}")
    print(f"100% Recall Threshold: {threshold_100_recall:.4f}\n")

    # Predictions for both thresholds
    pred_99 = (scores > threshold_99_percentile).astype(int)
    pred_100 = (scores > threshold_100_recall).astype(int)

    # Confusion matrices
    cm_99 = confusion_matrix(true_labels, pred_99)
    cm_100 = confusion_matrix(true_labels, pred_100)

    # Print results
    print("99th Percentile Confusion Matrix:")
    print(cm_99)
    tn_99, fp_99, fn_99, tp_99 = cm_99[0,0], cm_99[0,1], cm_99[1,0], cm_99[1,1]
    recall_99 = tp_99 / (tp_99 + fn_99) if (tp_99 + fn_99) > 0 else 0
    precision_99 = tp_99 / (tp_99 + fp_99) if (tp_99 + fp_99) > 0 else 0
    print(f"  TN: {tn_99} | FP: {fp_99}")
    print(f"  FN: {fn_99} | TP: {tp_99}")
    print(f"  Recall: {recall_99:.4f} | Precision: {precision_99:.4f}\n")

    print("100% Recall Confusion Matrix:")
    print(cm_100)
    tn_100, fp_100, fn_100, tp_100 = cm_100[0,0], cm_100[0,1], cm_100[1,0], cm_100[1,1]
    recall_100 = tp_100 / (tp_100 + fn_100) if (tp_100 + fn_100) > 0 else 0
    precision_100 = tp_100 / (tp_100 + fp_100) if (tp_100 + fp_100) > 0 else 0
    print(f"  TN: {tn_100} | FP: {fp_100}")
    print(f"  FN: {fn_100} | TP: {tp_100}")
    print(f"  Recall: {recall_100:.4f} | Precision: {precision_100:.4f}\n")

    # Save visualization
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    disp_99 = ConfusionMatrixDisplay(cm_99, display_labels=["Normal", "Defective"])
    disp_99.plot(cmap="Blues", values_format="d", ax=axes[0])
    axes[0].set_title(f"99th Percentile\n(threshold={threshold_99_percentile:.4f})",
                      fontsize=12, weight='bold')

    disp_100 = ConfusionMatrixDisplay(cm_100, display_labels=["Normal", "Defective"])
    disp_100.plot(cmap="Greens", values_format="d", ax=axes[1])
    axes[1].set_title(f"100% Recall\n(threshold={threshold_100_recall:.4f})",
                      fontsize=12, weight='bold')

    plt.tight_layout()

    output_file = output_dir / "confusion_matrices_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print("="*70)
    print(f"✓ Saved: {output_file}")
    print("="*70)


if __name__ == "__main__":
    # Default paths
    results_file = "results/inference_all/results.json"
    metrics_file = "results/metrics.json"
    output_dir = "results/figures"

    # Check if files exist
    if not Path(results_file).exists():
        print(f"Error: {results_file} not found")
        print("Run inference first: python scripts/inference.py ...")
        sys.exit(1)

    if not Path(metrics_file).exists():
        print(f"Error: {metrics_file} not found")
        print("Run training first: python scripts/train.py config/config.yaml")
        sys.exit(1)

    compare_thresholds(results_file, metrics_file, output_dir)
