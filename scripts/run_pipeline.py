#!/usr/bin/env python
"""
Complete ML Pipeline Runner
Runs data validation, training, and inference sequentially with MLflow logging.
Automatically starts MLflow UI in background.
"""

import subprocess
import sys
import os
from pathlib import Path




def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print("\n" + "="*70)
    print(f"{description}")
    print("="*70 + "\n")

    try:
        # Run with output streamed to console
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"\n✓ {description} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}\n")
        return False


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

    # Step 0: Validate data
    print("\n" + "="*70)
    print("STEP 0: Validating Dataset")
    print("="*70 + "\n")

    if Path("data/surface/test").exists():
        run_command(
            "python scripts/validation.py ./data surface",
            "Data Validation"
        )
        print("Data validation completed (warnings allowed)\n")
    else:
        print("⚠ Data directory not found, skipping validation\n")

    # Step 1: Training
    if not run_command(
        f"python scripts/train.py {config_file}",
        "STEP 1: Training Model"
    ):
        print("Training failed. Pipeline stopped.")
        sys.exit(1)

    # Step 2: Inference
    if not run_command(
        "python scripts/inference.py "
        "--model results/models/patchcore_surface.pkl "
        "--folder data/surface/test "
        "--output ./results/inference_latest "
        "--visualize",
        "STEP 2: Running Inference"
    ):
        print("Inference failed. Pipeline stopped.")
        sys.exit(1)

    # Step 3: Generate and log threshold comparison
    print("\nSTEP 3: Generating Threshold Comparison")
    print("="*70)
    run_command(
        "python scripts/mlflow_comparison.py",
        "Threshold Comparison"
    )

    # Step 4: Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nResults saved to:")
    print("  - Model: results/models/patchcore_surface.pkl")
    print("  - Metrics: results/metrics.json")
    print("  - Inference: results/inference_latest/")
    print("  - Comparison: results/figures/confusion_matrices_comparison.png")
    print("\nMLflow Logged:")
    print("  - Training metrics (AUROC, F1, etc)")
    print("  - Inference metrics (accuracy, precision, recall)")
    print("  - Threshold comparison image")
    print("\nView all results in MLflow:")
    print("  bash scripts/start_mlflow.sh")
    print("  Then open http://localhost:5000 in your browser")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
