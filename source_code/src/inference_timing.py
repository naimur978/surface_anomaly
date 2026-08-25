"""
Inference timing utilities for GPU/CPU performance comparison.
Logs timing metrics to MLflow.
"""

import time
from pathlib import Path
from typing import Optional, Dict, List, Any
import numpy as np
import torch
import mlflow


class InferenceTimer:
    """Context manager for timing inference operations."""

    def __init__(self, name: str = "inference") -> None:
        self.name = name
        self.start_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self) -> 'InferenceTimer':
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self.start_time

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000 if self.elapsed else 0


def measure_inference_time(detector: Any, image_paths: List[str | Path], device: str, n_warmup: int = 5, logger: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """
    Measure per-image inference time.

    Args:
        detector: AnomalyDetector instance
        image_paths: List of image paths to test
        device: Device name ('cuda', 'mps', 'cpu')
        n_warmup: Number of warmup iterations
        logger: Logger instance

    Returns:
        Dictionary with timing statistics
    """
    if not image_paths:
        return None

    # Move model to device
    detector.model.device = torch.device(device)
    detector.extractor.device = torch.device(device)
    detector.extractor.model = detector.extractor.model.to(device)

    # Warmup
    for i in range(min(n_warmup, len(image_paths))):
        try:
            _ = detector.detect(image_paths[i], return_heatmap=False)
        except Exception as e:
            if logger:
                logger.warning(f"Warmup iteration {i} failed: {e}")

    # Benchmark
    times = []
    for img_path in image_paths:
        try:
            with InferenceTimer() as timer:
                _ = detector.detect(img_path, return_heatmap=False)
            times.append(timer.elapsed)
        except Exception as e:
            if logger:
                logger.warning(f"Inference on {img_path} failed: {e}")

    if not times:
        return None

    times_ms = np.array(times) * 1000

    return {
        'device': device,
        'mean_ms': float(times_ms.mean()),
        'std_ms': float(times_ms.std()),
        'min_ms': float(times_ms.min()),
        'max_ms': float(times_ms.max()),
        'median_ms': float(np.median(times_ms)),
        'n_samples': len(times)
    }


def benchmark_inference(detector: Any, image_paths: List[str | Path], config: Dict[str, Any], logger: Optional[Any] = None) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark inference on available devices.

    Args:
        detector: AnomalyDetector instance
        image_paths: List of image paths to test
        config: Config dictionary with inference settings
        logger: Logger instance

    Returns:
        Dictionary with results for each device
    """
    results = {}
    n_warmup = config['inference'].get('warmup_samples', 5)
    n_samples = min(config['inference'].get('benchmark_samples', 10), len(image_paths))
    test_paths = image_paths[:n_samples]

    # Benchmark CUDA if available
    if torch.cuda.is_available():
        if logger:
            logger.info("Benchmarking CUDA...")
        cuda_result = measure_inference_time(
            detector, test_paths, 'cuda', n_warmup=n_warmup, logger=logger
        )
        if cuda_result:
            results['cuda'] = cuda_result
            if logger:
                logger.info(f"  CUDA: {cuda_result['mean_ms']:.2f}ms per image")

    # Benchmark MPS if available
    if torch.backends.mps.is_available():
        if logger:
            logger.info("Benchmarking MPS...")
        mps_result = measure_inference_time(
            detector, test_paths, 'mps', n_warmup=n_warmup, logger=logger
        )
        if mps_result:
            results['mps'] = mps_result
            if logger:
                logger.info(f"  MPS: {mps_result['mean_ms']:.2f}ms per image")

    # Benchmark CPU
    if logger:
        logger.info("Benchmarking CPU...")
    cpu_result = measure_inference_time(
        detector, test_paths, 'cpu', n_warmup=n_warmup, logger=logger
    )
    if cpu_result:
        results['cpu'] = cpu_result
        if logger:
            logger.info(f"  CPU: {cpu_result['mean_ms']:.2f}ms per image")

    return results


def log_inference_timing_to_mlflow(results: Dict[str, Dict[str, Any]]) -> None:
    """
    Log inference timing results to MLflow.

    Args:
        results: Dictionary with timing results for each device
    """
    if not results:
        return

    for device, metrics in results.items():
        # Log as metrics (numeric values)
        mlflow.log_metric(f"inference_mean_ms_{device}", metrics['mean_ms'])
        mlflow.log_metric(f"inference_std_ms_{device}", metrics['std_ms'])
        mlflow.log_metric(f"inference_min_ms_{device}", metrics['min_ms'])
        mlflow.log_metric(f"inference_max_ms_{device}", metrics['max_ms'])
        mlflow.log_metric(f"inference_median_ms_{device}", metrics['median_ms'])

        # Log as parameters (categorical info)
        mlflow.log_param(f"inference_device_{device}_samples", metrics['n_samples'])

    # Calculate and log speedup ratios
    if 'cuda' in results and 'cpu' in results:
        speedup = results['cpu']['mean_ms'] / results['cuda']['mean_ms']
        mlflow.log_metric("inference_speedup_cuda_vs_cpu", speedup)

    if 'mps' in results and 'cpu' in results:
        speedup = results['cpu']['mean_ms'] / results['mps']['mean_ms']
        mlflow.log_metric("inference_speedup_mps_vs_cpu", speedup)

    # Log summary as artifact (JSON)
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = Path(tmpdir) / "inference_timing.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        mlflow.log_artifact(str(tmpdir), artifact_path="inference")
