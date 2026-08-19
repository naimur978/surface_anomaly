#!/usr/bin/env python
"""
Complete ML Pipeline Runner
Runs data validation, training, and inference sequentially with MLflow logging.
Automatically starts MLflow UI in background.
"""

import subprocess
import sys
import time
import os
import signal
from pathlib import Path


def is_port_in_use(port=5000):
    """Check if a port is already in use."""
    try:
        result = subprocess.run(
            f"lsof -ti:{port}",
            shell=True,
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def kill_port_process(port=5000):
    """Kill process running on a specific port."""
    try:
        subprocess.run(
            f"lsof -ti:{port} | xargs kill -9",
            shell=True,
            capture_output=True
        )
        print(f"✓ Killed existing process on port {port}")
        time.sleep(1)  # Wait for port to be released
        return True
    except Exception as e:
        print(f"✗ Could not kill process: {e}")
        return False


def start_mlflow_server(port=5000):
    """Start MLflow UI server in background."""
    print("\n" + "="*70)
    print("Starting MLflow Server")
    print("="*70 + "\n")

    # Check if port is in use
    if is_port_in_use(port):
        print(f"Port {port} is already in use. Killing existing process...")
        kill_port_process(port)

    # Start MLflow in background
    try:
        mlflow_cmd = f"mlflow ui --backend-store-uri ./mlruns --port {port}"
        mlflow_process = subprocess.Popen(
            mlflow_cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"✓ MLflow server started on http://localhost:{port}")
        print(f"  (Process ID: {mlflow_process.pid})\n")
        time.sleep(2)  # Wait for server to start
        return mlflow_process
    except Exception as e:
        print(f"✗ Failed to start MLflow: {e}")
        return None


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print("\n" + "="*70)
    print(f"{description}")
    print("="*70 + "\n")

    try:
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
    print("SURFACE ANOMALY DETECTION - COMPLETE PIPELINE")
    print("="*70)

    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"

    # Verify config exists
    if not Path(config_file).exists():
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)

    # Start MLflow server
    mlflow_process = start_mlflow_server(port=5000)

    # Step 0: Validate data
    print("\n" + "="*70)
    print("STEP 0: Validating Dataset")
    print("="*70)

    if Path("data/surface/test").exists():
        if not run_command(
            "python scripts/validation.py ./data surface",
            "Data Validation"
        ):
            print("Warning: Data validation had issues, but continuing...")
    else:
        print("⚠ Data directory not found, skipping validation")

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

    # Step 3: Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nResults saved to:")
    print("  - Model: results/models/patchcore_surface.pkl")
    print("  - Metrics: results/metrics.json")
    print("  - Inference: results/inference_latest/")
    print("\nMLflow Server:")
    print("  Open http://localhost:5000 in your browser")
    print("\nMLflow is running in background. To stop it:")
    print("  kill {mlflow_process.pid if mlflow_process else 'N/A'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
