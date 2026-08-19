"""Tests for model components."""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import PatchCore


def test_patchcore_initialization():
    """Test PatchCore model initialization."""
    model = PatchCore(coreset_ratio=0.15, n_neighbors=9, device='cpu')

    assert model.coreset_ratio == 0.15
    assert model.n_neighbors == 9
    assert model.memory_bank is None


def test_patchcore_fit():
    """Test PatchCore model fitting."""
    # Create fake training data
    n_samples = 100
    feature_dim = 768
    features = torch.randn(n_samples, feature_dim)

    model = PatchCore(coreset_ratio=0.15, n_neighbors=9, device='cpu')
    model.fit(features, hw_shape=(14, 14))

    # Check memory bank is created
    assert model.memory_bank is not None
    expected_size = int(n_samples * 0.15)
    assert model.memory_bank.shape[0] >= expected_size * 0.9  # Allow some variance
    assert model.memory_bank.shape[1] == feature_dim


def test_patchcore_scoring():
    """Test PatchCore scoring on test data."""
    # Create fake data
    n_train = 100
    n_test = 10
    feature_dim = 768

    train_features = torch.randn(n_train, feature_dim)
    test_features = torch.randn(n_test, feature_dim).numpy()

    model = PatchCore(coreset_ratio=0.15, n_neighbors=9, device='cpu')
    model.fit(train_features, hw_shape=(14, 14))

    # Score test samples
    scores = []
    for i in range(n_test):
        score = model.score_image(test_features[i:i+1])
        scores.append(score)

    assert len(scores) == n_test
    assert all(isinstance(s, (float, np.floating)) for s in scores)
    assert all(s >= 0 for s in scores)  # Distances should be non-negative


def test_patchcore_score_map():
    """Test PatchCore heatmap generation."""
    # Create fake data
    n_train = 100
    feature_dim = 768
    patch_h, patch_w = 14, 14

    train_features = torch.randn(n_train, feature_dim)
    test_features = torch.randn(196, feature_dim).numpy()  # 14x14 patches

    model = PatchCore(coreset_ratio=0.15, n_neighbors=9, device='cpu')
    model.fit(train_features, hw_shape=(patch_h, patch_w))

    # Generate heatmap
    heatmap = model.score_map(test_features)

    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0
    assert heatmap.max() > 0
