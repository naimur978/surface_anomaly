#!/usr/bin/env python
"""
Complete ML Pipeline Runner
Runs data validation, training, and inference sequentially with MLflow logging.
Automatically starts MLflow UI in background.
"""

import subprocess
import sys
import os
import yaml
from pathlib import Path




def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{description}...")

    try:
        # Run with output streamed to console
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"✓ {description} completed\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}\n")
        return False


def get_model_path(config_file):
    """Get the correct model path based on current backbone in config."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    feature_extractor = config['model'].get('feature_extractor', 'dinov2_vitb14')
    model_name = f"anomaly_localization_surface_{feature_extractor}.pkl"
    return f"results/models/{model_name}"


def main():
    """Run the complete pipeline."""
    # Allow file store for MLflow (disable maintenance mode warning)
    os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

    print("\n" + "="*70)
    print("SURFACE ANOMALY DETECTION - TRAINING & INFERENCE PIPELINE")
    print("="*70)

    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"

    # Verify config exists
    if not Path(config_file).exists():
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)

    # Load config to get backbone
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    feature_extractor = config['model'].get('feature_extractor', 'dinov2_vitb14')
    print(f"\n🔧 Using backbone: {feature_extractor}")

    # Step 0: Validate data
    if Path("data/surface/test").exists():
        run_command(
            "python scripts/validation.py ./data surface",
            "Validating dataset"
        )
    else:
        print("⚠ Data directory not found, skipping validation\n")

    # Step 1: Training
    if not run_command(
        f"python scripts/train.py {config_file}",
        "Training model"
    ):
        print("✗ Training failed. Pipeline stopped.")
        sys.exit(1)

    # Get correct model path based on backbone
    model_path = get_model_path(config_file)

    # Step 2: Inference
    if not run_command(
        f"python scripts/inference.py "
        f"--model {model_path} "
        f"--config {config_file} "
        "--folder data/surface/test "
        "--output ./results/inference_latest "
        "--visualize",
        "Running inference"
    ):
        print("✗ Inference failed. Pipeline stopped.")
        sys.exit(1)


    # Step 4: Summary
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nResults:")
    print("  📁 Model:      results/models/patchcore_surface.pkl")
    print("  📊 Metrics:    results/metrics.json")
    print("  📈 Inference:  results/inference_latest/")
    print("  🔄 Comparison: results/figures/confusion_matrices_comparison.png")
    print("\nMLflow tracking:")
    print("  bash scripts/start_mlflow.sh")
    print("  then open http://localhost:5000")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
