"""Tests for configuration module."""

import pytest
import tempfile
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import load_config, setup_directories, set_seed


def test_load_config_exists():
    """Test loading existing config file."""
    config = load_config('config/config.yaml')
    assert config is not None
    assert 'data' in config
    assert 'model' in config
    assert 'training' in config


def test_load_config_keys():
    """Test config has all required keys."""
    config = load_config('config/config.yaml')

    # Data config
    assert 'root_dir' in config['data']
    assert 'category' in config['data']

    # Model config
    assert 'feature_extractor' in config['model']
    assert 'coreset_ratio' in config['model']

    # Training config
    assert 'batch_size' in config['training']
    assert 'seed' in config['training']


def test_setup_directories():
    """Test directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_config = {
            'output': {
                'results_dir': f'{tmpdir}/results',
                'figures_dir': f'{tmpdir}/results/figures',
                'models_dir': f'{tmpdir}/results/models',
                'predictions_dir': f'{tmpdir}/results/predictions',
                'logs_dir': f'{tmpdir}/results/logs',
            }
        }

        output_config = setup_directories(test_config)

        # Verify all directories exist
        assert Path(output_config['results_dir']).exists()
        assert Path(output_config['figures_dir']).exists()
        assert Path(output_config['models_dir']).exists()
        assert Path(output_config['logs_dir']).exists()


def test_set_seed():
    """Test seed setting for reproducibility."""
    import numpy as np
    import torch

    set_seed(42)

    # Generate random numbers
    np1 = np.random.randn(5)
    t1 = torch.randn(5)

    # Reset seed
    set_seed(42)

    # Generate again
    np2 = np.random.randn(5)
    t2 = torch.randn(5)

    # Should be identical (reproducible)
    assert np.allclose(np1, np2), "NumPy random not reproducible"
    assert torch.allclose(t1, t2), "PyTorch random not reproducible"
