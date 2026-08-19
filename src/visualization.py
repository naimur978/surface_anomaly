"""
Visualization utilities for training and evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def plot_roc_curve(ax, fpr, tpr, auroc_image):
    """Plot ROC curve."""
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
    """Plot score distribution with threshold line."""
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


def plot_evaluation(test_scores, test_labels, auroc_image, threshold, output_path="./results/figures/evaluation.png"):
    """Create ROC and score distribution plots."""
    fpr, tpr, _ = roc_curve(test_labels, test_scores)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    plot_roc_curve(axes[0], fpr, tpr, auroc_image)
    plot_score_distribution(axes[1], test_scores, test_labels, threshold)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
