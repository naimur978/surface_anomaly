#!/usr/bin/env python
"""
Complete ML Pipeline Runner
Runs data validation, training, and inference sequentially with MLflow logging.

Run: python scripts/run_pipeline.py [config/config.yaml]

Config path defaults to config/config.yaml if omitted. Start MLflow's UI
separately with `python scripts/mlflow_ui.py` to view logged runs.
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
        # Stream output directly to console rather than capturing it, since
        # these are long-running steps (training, inference) the user should
        # watch progress on in real time.
        subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"[OK] {description} completed\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] {description} failed with exit code {e.returncode}\n")
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
    # Use SQLite backend for MLflow (recommended over file-based store)
    os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///mlflow.db'

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
    print(f"\n[*] Using backbone: {feature_extractor}")

    # Step 1: Validate data
    if Path("data/surface/test").exists():
        run_command(
            f"{sys.executable} scripts/validation.py ./data surface",
            "Validating dataset"
        )
    else:
        print("[WARN] Data directory not found, skipping validation\n")

    # Step 2: Training
    if not run_command(
        f"{sys.executable} scripts/train.py {config_file}",
        "Training model"
    ):
        print("[FAIL] Training failed. Pipeline stopped.")
        sys.exit(1)

    # Get correct model path based on backbone
    model_path = get_model_path(config_file)

    # Step 3: Inference
    if not run_command(
        f"{sys.executable} scripts/inference.py "
        f"--model {model_path} "
        f"--config {config_file} "
        "--folder data/surface/test "
        "--output ./results/inference_latest "
        "--visualize",
        "Running inference"
    ):
        print("[FAIL] Inference failed. Pipeline stopped.")
        sys.exit(1)

    # Summary
    print("\n" + "="*70)
    print("[OK] PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nResults:")
    print(f"  Model:     {model_path}")
    print("  Metrics:   results/metrics.json")
    print("  Inference: results/inference_latest/")
    print("\nMLflow tracking:")
    print("  python scripts/mlflow_ui.py")
    print("  then open http://127.0.0.1:5000/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
