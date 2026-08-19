#!/usr/bin/env python3
"""
Device detection utility - Check available compute devices
"""

import torch
import os


def get_device(logger=None):
    """
    Auto-select best available device with priority:
    1. CUDA (NVIDIA GPU)
    2. MPS (MacBook Metal GPU)
    3. CPU (fallback)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if logger:
            logger.info("Using CUDA (NVIDIA GPU)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        if logger:
            logger.info("Using Metal Performance Shaders (MacBook GPU)")
    else:
        device = torch.device("cpu")
        if logger:
            logger.info("Using CPU")

    return device


def check_devices():
    """Check and display available compute devices"""
    print("\n" + "="*60)
    print("PYTORCH DEVICE AVAILABILITY CHECK")
    print("="*60 + "\n")

    # Priority 1: CUDA
    print("1. CUDA (NVIDIA GPUs)")
    print("-" * 40)
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {name} ({memory:.2f} GB)")
        selected = "CUDA"
    else:
        print("   ✗ CUDA not available")

    # Priority 2: Metal Performance Shaders (MacBook)
    print("\n2. Metal Performance Shaders (MacBook GPU)")
    print("-" * 40)
    if hasattr(torch.backends, 'mps'):
        if torch.backends.mps.is_available():
            print(f"   ✓ Metal Performance Shaders available")
            if not torch.cuda.is_available():
                selected = "MPS"
        else:
            print(f"   ✗ Metal not available or macOS version too old")
    else:
        print(f"   ✗ Not on macOS or PyTorch version doesn't support MPS")

    # Fallback: CPU
    print("\n3. CPU (Fallback)")
    print("-" * 40)
    print(f"   ✓ CPU always available")
    if 'selected' not in locals():
        selected = "CPU"

    # Summary
    print("\n" + "="*60)
    print(f"SELECTED DEVICE: {selected}")
    print("="*60)

    # Device configuration
    print("\nPyTorch Configuration:")
    print("-" * 40)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")

    print(f"Cudnn enabled: {torch.backends.cudnn.enabled}")

    # Testing device
    print("\n" + "="*60)
    print("DEVICE TEST")
    print("="*60)

    device = torch.device(selected.lower())
    print(f"\nTesting on {device}...")

    try:
        # Create test tensor
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)

        # Perform operations
        z = torch.matmul(x, y)

        print(f"✓ Tensor operations work correctly")
        print(f"  Result shape: {z.shape}")
        print(f"  Result device: {z.device}")

    except Exception as e:
        print(f"✗ Error during test: {str(e)}")

    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    if selected == "CUDA":
        print("""
✓ CUDA is available - optimal for training
  - Edit config.yaml: device: "cuda"
  - Expect: ~28-30ms per image inference
        """)
    elif selected == "MPS":
        print("""
✓ Metal Performance Shaders (MacBook GPU) available
  - Edit config.yaml: device: "mps"
  - Expect: ~50-100ms per image inference (M1/M2) or faster (M3+)
  - Note: May have compatibility issues with some operations
  - Fallback: Set PYTORCH_ENABLE_MPS_FALLBACK=1
        """)
    else:
        print("""
⚠ CPU only mode
  - Slower inference: ~300ms per image
  - Consider using CUDA GPU for faster processing
  - For development/testing, CPU is acceptable
        """)

    print("="*60 + "\n")


def benchmark_device():
    """Quick performance benchmark"""
    print("\nQUICK PERFORMANCE BENCHMARK")
    print("="*60)

    devices = []

    # CUDA
    if torch.cuda.is_available():
        devices.append(('cuda', torch.device('cuda')))

    # MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append(('mps', torch.device('mps')))

    # CPU
    devices.append(('cpu', torch.device('cpu')))

    for name, device in devices:
        try:
            import time

            # Warmup
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            _ = torch.matmul(x, y)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            # Benchmark
            start = time.perf_counter()
            for _ in range(10):
                x = torch.randn(1000, 1000).to(device)
                y = torch.randn(1000, 1000).to(device)
                _ = torch.matmul(x, y)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()

            time_ms = (end - start) / 10 * 1000
            print(f"{name.upper():10} : {time_ms:7.2f} ms/op")

        except Exception as e:
            print(f"{name.upper():10} : Error - {str(e)}")

    print("="*60 + "\n")


if __name__ == "__main__":
    import sys

    check_devices()

    if "--benchmark" in sys.argv:
        benchmark_device()
