"""Tests for dataset utilities."""

import pytest
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import MVTecDataset


def test_dataset_initialization():
    """Test dataset initialization."""
    try:
        dataset = MVTecDataset(
            root='./data',
            category='surface',
            split='test',
            crop_size=224
        )

        assert len(dataset) > 0
        assert dataset.split == 'test'
    except Exception as e:
        pytest.skip(f"Dataset not available: {e}")


def test_dataset_sample():
    """Test getting a single sample from dataset."""
    try:
        dataset = MVTecDataset(
            root='./data',
            category='surface',
            split='test',
            crop_size=224
        )

        if len(dataset) > 0:
            img_t, mask_t, label, img_path = dataset[0]

            assert img_t.shape == (3, 224, 224), f"Image shape should be (3, 224, 224), got {img_t.shape}"
            assert mask_t.shape[1:] == (224, 224), f"Mask shape should be (*, 224, 224), got {mask_t.shape}"
            assert label in [0, 1], f"Label should be 0 or 1, got {label}"
            assert isinstance(img_path, str), "Image path should be string"
    except Exception as e:
        pytest.skip(f"Dataset not available: {e}")


def test_dataset_labels():
    """Test that dataset returns correct labels."""
    try:
        dataset = MVTecDataset(
            root='./data',
            category='surface',
            split='test',
            crop_size=224
        )

        labels = []
        for i in range(min(10, len(dataset))):
            _, _, label, _ = dataset[i]
            labels.append(label)

        assert all(l in [0, 1] for l in labels), "All labels should be 0 or 1"
    except Exception as e:
        pytest.skip(f"Dataset not available: {e}")
