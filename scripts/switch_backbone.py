#!/usr/bin/env python
"""
Switch feature extraction backbone between DINOv2 and EfficientNet B4
"""

import sys
import yaml
from pathlib import Path


def switch_backbone(backbone_name):
    """Switch to specified backbone in config."""

    config_path = Path("config/config.yaml")

    if not config_path.exists():
        print(f"Error: {config_path} not found")
        sys.exit(1)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate backbone
    valid_backbones = ["dinov2_vitb14", "efficientnet_b4"]
    if backbone_name not in valid_backbones:
        print(f"Error: Invalid backbone '{backbone_name}'")
        print(f"Valid options: {', '.join(valid_backbones)}")
        sys.exit(1)

    # Update config
    config['model']['feature_extractor'] = backbone_name

    # Save config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Switched backbone to: {backbone_name}")
    print(f"\nFeature dimensions:")
    if backbone_name == "dinov2_vitb14":
        print("  - DINOv2 ViT-B14: 768 dimensions")
    else:
        print("  - EfficientNet B4: 1792 dimensions")

    print(f"\nRun training with:")
    print(f"  python scripts/run_pipeline.py config/config.yaml")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/switch_backbone.py <backbone>")
        print("\nAvailable backbones:")
        print("  - dinov2_vitb14    (DINOv2 Vision Transformer, 768-dim)")
        print("  - efficientnet_b4  (EfficientNet B4, 1792-dim)")
        print("\nExample:")
        print("  python scripts/switch_backbone.py efficientnet_b4")
        sys.exit(1)

    backbone = sys.argv[1]
    switch_backbone(backbone)
