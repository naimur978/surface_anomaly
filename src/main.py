"""
PatchCore Anomaly Detection Training Orchestration

Main entry point that coordinates data loading, model training, and evaluation.
"""

import sys
from pathlib import Path
import warnings
from io import StringIO

import torch
from torch.utils.data import DataLoader

from .dataset import MVTecDataset
from .config import setup_logging, load_config, setup_directories, set_seed, print_gpu_info
from .training import train_patchcore, evaluate_model, save_results
from .check_device import get_device
from .mlflow_utils import setup_mlflow, start_run, end_run, log_training_params, log_training_metrics, log_artifacts

warnings.filterwarnings('ignore')


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================
def main(config_path="config.yaml"):
    """Main training pipeline."""
    # Setup
    config = load_config(config_path)
    set_seed(config['training'].get('seed', 42))
    output_config = setup_directories(config)

    log_file = Path(output_config['logs_dir']) / "training.log"
    logger = setup_logging(log_file)

    logger.info("=" * 60)
    logger.info("PATCHCORE ANOMALY DETECTION - TRAINING")
    logger.info("=" * 60)

    # Initialize MLflow
    setup_mlflow(experiment_name='surface-anomaly-detection')
    run = start_run(run_name='surface_anomaly_training')
    logger.info(f"MLflow run started: {run.info.run_id}")

    device = get_device(logger=logger)
    print_gpu_info(logger, device)

    # Load data (suppress dataset prints)
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    train_dataset = MVTecDataset(
        config['data']['root_dir'],
        config['data']['category'],
        split='train',
        crop_size=config['image']['crop_size'],
        apply_roi_mask=config['image'].get('apply_roi_mask', False)
    )
    val_dataset = MVTecDataset(
        config['data']['root_dir'],
        config['data']['category'],
        split='test',
        crop_size=config['image']['crop_size'],
        apply_roi_mask=config['image'].get('apply_roi_mask', False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=config['training'].get('pin_memory', False)
    )
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    sys.stdout = old_stdout

    # Log training parameters
    log_training_params(config)

    # Train
    model, extractor = train_patchcore(train_loader, val_loader, device, config, output_config, logger)

    # Evaluate
    metrics, val_scores, val_labels, val_score_maps, _ = evaluate_model(
        val_loader, extractor, model, train_loader, config, output_config, logger
    )

    # Log metrics
    log_training_metrics(metrics)

    # Save
    save_results(model, metrics, val_scores, val_labels, val_score_maps, val_dataset,
                 output_config, config, logger)

    # Log artifacts
    log_artifacts(output_config['results_dir'])

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"MLflow UI: mlflow ui --backend-store-uri ./mlruns")

    end_run()
    return metrics


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    main(config_file)
