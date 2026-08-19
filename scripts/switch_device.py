#!/usr/bin/env python
"""
Switch compute device between CUDA, MPS (MacBook), and CPU
"""

import sys
import yaml
from pathlib import Path


def get_available_devices():
    """Check available devices."""
    import torch

    devices = {
        'cuda': torch.cuda.is_available(),
        'mps': torch.backends.mps.is_available(),
        'cpu': True
    }
    return devices


def switch_device(device_name):
    """Switch to specified device in config."""

    config_path = Path("config/config.yaml")

    if not config_path.exists():
        print(f"Error: {config_path} not found")
        sys.exit(1)

    # Check available devices
    available = get_available_devices()

    # Validate device
    valid_devices = ["cuda", "mps", "cpu"]
    if device_name not in valid_devices:
        print(f"Error: Invalid device '{device_name}'")
        print(f"Valid options: {', '.join(valid_devices)}")
        sys.exit(1)

    # Check if device is available
    if not available[device_name]:
        print(f"Error: Device '{device_name}' is NOT available on this machine")
        print(f"\nAvailable devices:")
        for dev, avail in available.items():
            status = "✓ Available" if avail else "✗ Not available"
            print(f"  - {dev}: {status}")
        sys.exit(1)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Update config
    config['model']['device'] = device_name

    # Save config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Switched device to: {device_name}")
    print(f"\nDevice information:")
    if device_name == "cuda":
        print("  - NVIDIA GPU (CUDA)")
    elif device_name == "mps":
        print("  - MacBook GPU (Metal Performance Shaders)")
    elif device_name == "cpu":
        print("  - CPU (Central Processing Unit)")

    print(f"\nRun training with:")
    print(f"  python scripts/run_pipeline.py config/config.yaml")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        import torch

        available = get_available_devices()
        print("Usage: python scripts/switch_device.py <device>")
        print("\nAvailable on this machine:")
        for dev, avail in available.items():
            status = "✓ AVAILABLE" if avail else "✗ Not available"
            print(f"  - {dev:6s} {status}")

        print("\nExamples:")
        print("  python scripts/switch_device.py cuda      # NVIDIA GPU")
        print("  python scripts/switch_device.py mps       # MacBook GPU")
        print("  python scripts/switch_device.py cpu       # CPU")
        sys.exit(1)

    device = sys.argv[1]
    switch_device(device)
