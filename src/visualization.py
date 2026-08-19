"""
Visualization utilities for training and evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from pathlib import Path
from PIL import Image
import random


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


def visualize_random_heatmaps(test_dataset, test_scores, test_labels, score_maps, threshold,
                             n_samples=4, output_path="./results/figures/random_heatmaps.png", dpi=150):
    """
    Visualize random predictions with heatmaps.
    Shows: original image | heatmap | overlay with prediction
    """
    indices = random.sample(range(len(test_dataset)), min(n_samples, len(test_dataset)))

    # Get vmin/vmax from sample maps
    sample_maps = [score_maps[i] for i in indices]
    vmin = min(m.min() for m in sample_maps)
    vmax = max(m.max() for m in sample_maps)

    fig, axes = plt.subplots(n_samples, 3, figsize=(9, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, 3)

    for row, idx in enumerate(indices):
        # Load original image
        img_path = test_dataset.samples[idx][0]
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_np = np.array(img) / 255.0

        score = test_scores[idx]
        label = test_labels[idx]
        heatmap = score_maps[idx]

        # Determine prediction
        pred = 1 if score > threshold else 0
        actual = "Defective" if label == 1 else "Normal"
        predicted = "Defective" if pred == 1 else "Normal"
        correct = "✓" if pred == label else "✗"

        # Original image
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f"Actual: {actual}")
        axes[row, 0].axis("off")

        # Heatmap
        axes[row, 1].imshow(heatmap, cmap="jet", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title("Anomaly Heatmap")
        axes[row, 1].axis("off")

        # Overlay
        axes[row, 2].imshow(img_np)
        axes[row, 2].imshow(heatmap, cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax)
        axes[row, 2].set_title(f"Pred: {predicted} {correct}\nScore: {score:.3f}")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def visualize_roi_masking_effect(dataset_with_roi, dataset_without_roi, n_samples=4,
                                output_path="./results/figures/roi_masking_effect.png", dpi=150):
    """
    Visualize effect of ROI masking.
    Shows: original image | without masking | with masking
    """
    indices = random.sample(range(len(dataset_with_roi)), min(n_samples, len(dataset_with_roi)))

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, 3)

    for row, idx in enumerate(indices):
        img_path = dataset_with_roi.samples[idx][0]

        # Original image (from disk)
        original = Image.open(img_path).convert("RGB")
        original_np = np.array(original) / 255.0

        # Without ROI masking
        without_roi = dataset_without_roi[idx][0].permute(1, 2, 0).numpy()
        without_roi = without_roi * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        without_roi = np.clip(without_roi, 0, 1)

        # With ROI masking
        with_roi = dataset_with_roi[idx][0].permute(1, 2, 0).numpy()
        with_roi = with_roi * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        with_roi = np.clip(with_roi, 0, 1)

        # Original
        axes[row, 0].imshow(original_np)
        axes[row, 0].set_title("Original Image")
        axes[row, 0].axis("off")

        # Without ROI mask
        axes[row, 1].imshow(without_roi)
        axes[row, 1].set_title("Without ROI Masking")
        axes[row, 1].axis("off")

        # With ROI mask
        axes[row, 2].imshow(with_roi)
        axes[row, 2].set_title("With ROI Masking")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def visualize_preprocessing_pipeline(test_dataset, n_samples=2,
                                    output_path="./results/figures/preprocessing_pipeline.png", dpi=150):
    """
    Visualize preprocessing pipeline.
    Shows: original → padded → resized for both image and mask
    """
    indices = random.sample(range(len(test_dataset)), min(n_samples, len(test_dataset)))

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, 3)

    for row, idx in enumerate(indices):
        img_path, mask_path, label = test_dataset.samples[idx]

        # Original image
        original = Image.open(img_path).convert("RGB")
        original_np = np.array(original)

        # Padded (to square)
        padded = test_dataset._pad_to_square(original, pad_color=0)
        padded_np = np.array(padded)

        # Resized
        resized = padded.resize((224, 224))
        resized_np = np.array(resized)

        # Original
        axes[row, 0].imshow(original_np)
        axes[row, 0].set_title(f"Original\n{original.size[0]}×{original.size[1]}")
        axes[row, 0].axis("off")

        # Padded
        axes[row, 1].imshow(padded_np)
        axes[row, 1].set_title(f"Padded to Square\n{padded.size[0]}×{padded.size[1]}")
        axes[row, 1].axis("off")

        # Resized
        axes[row, 2].imshow(resized_np)
        axes[row, 2].set_title(f"Resized\n224×224")
        axes[row, 2].axis("off")

        # Add label
        label_text = "Normal" if label == 0 else "Defective"
        fig.text(0.02, 0.98 - row*0.5, label_text, fontsize=10, weight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
