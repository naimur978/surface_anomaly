#!/usr/bin/env python3
"""
Benchmark inference performance on different devices.
Measures per-image inference time for feature extraction + anomaly scoring.
"""

import sys
import os
import time
import random
import pickle
from pathlib import Path
from io import StringIO

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataset import MVTecDataset
from src.models import FeatureExtractor
from src.config import load_config, setup_logging
from src.check_device import get_device


def measure_inference_time(dataset, model, extractor, device, n_warmup=5, n_samples=10):
    """
    Measure per-image inference time.

    Args:
        dataset: Test dataset
        model: Trained PatchCore model
        extractor: Feature extractor
        device: Torch device
        n_warmup: Number of warmup samples
        n_samples: Number of samples to benchmark

    Returns:
        Array of inference times in seconds
    """
    # Select random samples
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    samples = [dataset[i][0].unsqueeze(0) for i in indices]

    # Move model to device
    extractor.model = extractor.model.to(device)
    extractor.device = device

    # Warmup
    for i in range(n_warmup):
        img = samples[i % len(samples)]
        patches, _ = extractor.extract(img)
        _ = model.score_image(patches[0].numpy())

    # Benchmark
    times = []
    for img in samples:
        start = time.perf_counter()
        patches, _ = extractor.extract(img)
        _ = model.score_image(patches[0].numpy())
        end = time.perf_counter()
        times.append(end - start)

    return np.array(times)


def benchmark_devices(test_dataset, model, extractor, model_name, n_warmup=5, n_samples=10):
    """
    Benchmark inference on available devices.

    Args:
        test_dataset: Test dataset
        model: Trained PatchCore model
        extractor: Feature extractor
        model_name: Feature extractor model name (for logging)
        n_warmup: Number of warmup samples
        n_samples: Number of samples to benchmark

    Returns:
        Dictionary with timing results
    """
    results = {}

    # Benchmark CUDA if available
    if torch.cuda.is_available():
        print("\n🚀 Benchmarking CUDA...")
        cuda_times = measure_inference_time(
            test_dataset, model, extractor,
            device=torch.device("cuda"),
            n_warmup=n_warmup, n_samples=n_samples
        )
        results['cuda'] = cuda_times
        print(f"  ✓ {len(cuda_times)} samples completed")

    # Benchmark MPS (MacBook GPU) if available
    if torch.backends.mps.is_available():
        print("\n🚀 Benchmarking MPS (MacBook GPU)...")
        mps_times = measure_inference_time(
            test_dataset, model, extractor,
            device=torch.device("mps"),
            n_warmup=n_warmup, n_samples=n_samples
        )
        results['mps'] = mps_times
        print(f"  ✓ {len(mps_times)} samples completed")

    # Benchmark CPU
    print("\n🚀 Benchmarking CPU...")
    cpu_times = measure_inference_time(
        test_dataset, model, extractor,
        device=torch.device("cpu"),
        n_warmup=n_warmup, n_samples=n_samples
    )
    results['cpu'] = cpu_times
    print(f"  ✓ {len(cpu_times)} samples completed")

    return results


def print_results(results, extractor_name):
    """Print benchmark results in table format."""
    print("\n" + "="*80)
    print(f"INFERENCE BENCHMARK RESULTS - {extractor_name}")
    print("="*80)
    print(f"{'Device':<15} {'Mean (ms)':<15} {'Std (ms)':<15} {'Min (ms)':<15} {'Max (ms)':<15}")
    print("-"*80)

    for device, times in sorted(results.items()):
        mean_ms = times.mean() * 1000
        std_ms = times.std() * 1000
        min_ms = times.min() * 1000
        max_ms = times.max() * 1000
        print(f"{device:<15} {mean_ms:<15.2f} {std_ms:<15.2f} {min_ms:<15.2f} {max_ms:<15.2f}")

    # Calculate speedup
    if 'cuda' in results and 'cpu' in results:
        speedup = results['cpu'].mean() / results['cuda'].mean()
        print("-"*80)
        print(f"{'CUDA Speedup':<15} {speedup:.1f}x faster than CPU")

    if 'mps' in results and 'cpu' in results:
        speedup = results['cpu'].mean() / results['mps'].mean()
        print(f"{'MPS Speedup':<15} {speedup:.1f}x faster than CPU")

    print("="*80 + "\n")


def main():
    """Main benchmark script."""
    # Load config
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    config = load_config(config_file)

    # Setup logging
    log_file = Path(config['output']['logs_dir']) / "benchmark.log"
    logger = setup_logging(log_file)

    logger.info("="*60)
    logger.info("INFERENCE BENCHMARK")
    logger.info("="*60)

    # Get feature extractor config
    model_name = config['model'].get('feature_extractor', 'dinov2_vitb14')
    model_path = (
        f"results/models/anomaly_localization_surface_{model_name}.pkl"
    )

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print(f"Train model first: python scripts/run_pipeline.py {config_file}")
        sys.exit(1)

    print(f"\n📦 Loading model: {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load dataset (suppress prints)
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    test_dataset = MVTecDataset(
        config['data']['root_dir'],
        config['data']['category'],
        split='test',
        crop_size=config['image']['crop_size'],
        apply_roi_mask=config['image'].get('apply_roi_mask', False)
    )

    sys.stdout = old_stdout

    # Create feature extractor
    device = get_device(logger=logger)
    extractor = FeatureExtractor(
        device=device,
        model_name=model_name,
        logger=logger
    )

    # Run benchmarks
    print(f"\n📊 Benchmarking {model_name} ({len(test_dataset)} test images)")
    print(f"Config: {config['image'].get('apply_roi_mask', False) and '✓ ROI masking' or '✗ No ROI masking'}")

    results = benchmark_devices(
        test_dataset, model, extractor, model_name,
        n_warmup=config['inference'].get('warmup_samples', 5),
        n_samples=config['inference'].get('benchmark_samples', 10)
    )

    # Print results
    print_results(results, model_name)

    logger.info(f"Benchmark results saved to {log_file}")


if __name__ == "__main__":
    main()
