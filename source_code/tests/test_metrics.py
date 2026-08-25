"""Tests for metrics computation."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.metrics import compute_metrics, find_threshold_for_recall, compute_confusion_metrics


def test_compute_metrics():
    """Test metric computation."""
    # Create fake data
    n_samples = 100
    scores = np.random.rand(n_samples) * 100
    labels = np.random.randint(0, 2, n_samples)
    score_maps = [np.random.rand(224, 224) * 100 for _ in range(n_samples)]
    masks = [np.random.randint(0, 2, (224, 224)) for _ in range(n_samples)]
    threshold = 50.0

    auroc_image, auroc_pixel, f1 = compute_metrics(
        scores, labels, score_maps, masks, threshold
    )

    assert 0 <= auroc_image <= 1, f"AUROC image should be in [0, 1], got {auroc_image}"
    assert 0 <= auroc_pixel <= 1, f"AUROC pixel should be in [0, 1], got {auroc_pixel}"
    assert 0 <= f1 <= 1, f"F1 should be in [0, 1], got {f1}"


def test_find_threshold_for_recall():
    """Test threshold finding for target recall."""
    # Create fake data where some samples are anomalies
    n_samples = 100
    scores = np.random.rand(n_samples) * 100
    labels = np.zeros(n_samples)
    labels[-20:] = 1  # Last 20 are anomalies

    threshold = find_threshold_for_recall(scores, labels, target_recall=0.9)

    assert isinstance(threshold, (float, np.floating))
    assert 0 <= threshold <= 100


def test_compute_confusion_metrics():
    """Test confusion matrix computation."""
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    scores = np.array([10, 20, 80, 75, 15, 85, 25, 30, 90, 70])
    threshold = 50.0

    metrics = compute_confusion_metrics(labels, scores, threshold)

    assert 'cm' in metrics
    assert 'tn' in metrics
    assert 'fp' in metrics
    assert 'fn' in metrics
    assert 'tp' in metrics
    assert 'recall' in metrics
    assert 'precision' in metrics
    assert 'f1' in metrics

    # Check ranges
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['f1'] <= 1
