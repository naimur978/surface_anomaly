#!/usr/bin/env python
"""
Log threshold comparison to MLflow
Compares 99th percentile vs 100% recall thresholds and logs comparison image to MLflow.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import mlflow

def log_threshold_comparison_to_mlflow():
    """Generate and log threshold comparison to MLflow."""

    # Set MLflow tracking
    os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
    mlflow.set_tracking_uri('./mlruns')

    # Check files exist
    results_file = Path("results/inference_latest/results.json")
    metrics_file = Path("results/metrics.json")
    comparison_file = Path("results/figures/confusion_matrices_comparison.png")

    if not results_file.exists():
        print(f"Error: {results_file} not found")
        print("Run inference first: python scripts/inference.py ...")
        sys.exit(1)

    if not metrics_file.exists():
        print(f"Error: {metrics_file} not found")
        print("Run training first: python scripts/train.py config/config.yaml")
        sys.exit(1)

    # Run comparison script
    print("="*70)
    print("GENERATING THRESHOLD COMPARISON")
    print("="*70 + "\n")

    import subprocess
    result = subprocess.run(
        "python scripts/compare_thresholds.py",
        shell=True,
        check=True
    )

    # Log to MLflow
    if comparison_file.exists():
        print("\n" + "="*70)
        print("LOGGING TO MLFLOW")
        print("="*70 + "\n")

        # Get latest inference run
        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        # Find surface_anomaly_detection experiment
        experiment = client.get_experiment_by_name('surface-anomaly-detection')
        if not experiment:
            print("Error: Experiment not found")
            sys.exit(1)

        # Get latest run
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        if not runs:
            print("Error: No runs found")
            sys.exit(1)

        latest_run = runs[0]

        # Log comparison image to the run
        with mlflow.start_run(run_id=latest_run.info.run_id):
            mlflow.log_artifact(str(comparison_file), artifact_path='comparison')

        print(f"✓ Logged threshold comparison to MLflow")
        print(f"  Run ID: {latest_run.info.run_id}")
        print(f"  File: {comparison_file}")
        print("\n" + "="*70)
        print("View in MLflow: http://localhost:5000")
        print("="*70 + "\n")
    else:
        print(f"Error: Comparison image not generated: {comparison_file}")
        sys.exit(1)


if __name__ == "__main__":
    log_threshold_comparison_to_mlflow()
