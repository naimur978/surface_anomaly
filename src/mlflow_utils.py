"""MLflow experiment tracking utilities."""

import os
from typing import Dict, Any, Optional
# Allow file store to avoid maintenance mode warning
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

import mlflow
import mlflow.pytorch
import json
from pathlib import Path


def setup_mlflow(experiment_name: str = 'surface-anomaly-detection', tracking_uri: str = './mlruns') -> str:
    """Initialize MLflow experiment.

    Args:
        experiment_name: Name of the experiment
        tracking_uri: Local directory for MLflow artifacts

    Returns:
        Experiment ID
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_training_params(config: Dict[str, Any]) -> None:
    """Log training hyperparameters to MLflow.

    Args:
        config: Config dictionary
    """
    # Log all important configuration parameters
    params = {
        # Model config
        'feature_extractor': config['model'].get('feature_extractor', 'dinov2_vitb14'),
        'coreset_ratio': config['model'].get('coreset_ratio', 0.15),
        'n_neighbors': config['model'].get('n_neighbors', 9),

        # Image config
        'crop_size': config['image'].get('crop_size', 224),
        'img_size': config['image'].get('img_size', 256),
        'pad_color': config['image'].get('pad_color', 0),
        'apply_roi_mask': config['image'].get('apply_roi_mask', False),
        'normalization': config['image'].get('normalization', 'imagenet'),

        # Training config
        'batch_size': config['training'].get('batch_size', 32),
        'num_workers': config['training'].get('num_workers', 0),
        'seed': config['training'].get('seed', 42),

        # Evaluation config
        'threshold_percentile': config['evaluation'].get('threshold_percentile', 99),
        'target_recall': config['evaluation'].get('target_recall', 1.0),
        'heatmap_smoothing': config['evaluation'].get('heatmap_smoothing', 4),

        # Data config
        'category': config['data'].get('category', 'surface'),
        'train_split': config['data'].get('train_split', 'train'),
        'test_split': config['data'].get('test_split', 'test'),

        # Device config
        'device': config['model'].get('device', 'cuda'),
    }

    mlflow.log_params(params)


def log_training_metrics(metrics: Dict[str, Any]) -> None:
    """Log training metrics to MLflow.

    Args:
        metrics: Dictionary of metrics
    """
    mlflow.log_metrics({
        'auroc_image': metrics.get('auroc_image', 0),
        'auroc_pixel': metrics.get('auroc_pixel', 0),
        'f1_score': metrics.get('f1', 0),
        'recall': metrics.get('recall', 0),
        'precision': metrics.get('precision', 0),
    })


def log_model(model_path: str | Path, model_name: str = 'surface_anomaly') -> None:
    """Log trained model to MLflow.

    Args:
        model_path: Path to saved model
        model_name: Name of the model
    """
    model_path = Path(model_path)
    if model_path.exists():
        mlflow.log_artifact(str(model_path), artifact_path='models')


def log_artifacts(results_dir: str | Path) -> None:
    """Log result artifacts (metrics, confusion matrix, etc).

    Args:
        results_dir: Path to results directory
    """
    results_dir = Path(results_dir)

    # Log metrics JSON
    metrics_file = results_dir / 'metrics.json'
    if metrics_file.exists():
        mlflow.log_artifact(str(metrics_file), artifact_path='metrics')

    # Log confusion matrix figure
    figures_dir = results_dir / 'figures'
    if figures_dir.exists():
        for fig in figures_dir.glob('*.png'):
            mlflow.log_artifact(str(fig), artifact_path='figures')


def log_inference_results(results_dir: str | Path, output_name: str = 'inference_latest') -> None:
    """Log inference results to MLflow.

    Args:
        results_dir: Path to results directory
        output_name: Name of inference output directory
    """
    inference_dir = results_dir / output_name
    if inference_dir.exists():
        # Log results.json
        results_file = inference_dir / 'results.json'
        if results_file.exists():
            mlflow.log_artifact(str(results_file), artifact_path='inference')

        # Log confusion matrix
        confusion_file = inference_dir / 'confusion_matrix.png'
        if confusion_file.exists():
            mlflow.log_artifact(str(confusion_file), artifact_path='inference')

        # Log heatmaps
        heatmaps_dir = inference_dir / 'heatmaps'
        if heatmaps_dir.exists():
            for heatmap in heatmaps_dir.glob('*.png'):
                mlflow.log_artifact(str(heatmap), artifact_path='inference/heatmaps')


def start_run(run_name: str, tags: Optional[Dict[str, str]] = None) -> Any:
    """Start a new MLflow run.

    Args:
        run_name: Name of the run
        tags: Dictionary of tags
    """
    run = mlflow.start_run(run_name=run_name)
    if tags:
        mlflow.set_tags(tags)
    return run


def end_run() -> None:
    """End the current MLflow run."""
    mlflow.end_run()


def log_dataset_info(dataset_stats: Dict[str, int]) -> None:
    """Log dataset statistics to MLflow.

    Args:
        dataset_stats: Dictionary with dataset statistics
    """
    mlflow.log_params({
        'train_images': dataset_stats.get('train_images', 0),
        'val_images': dataset_stats.get('val_images', 0),
        'test_images': dataset_stats.get('test_images', 0),
        'train_normal': dataset_stats.get('train_normal', 0),
        'train_defective': dataset_stats.get('train_defective', 0),
        'val_normal': dataset_stats.get('val_normal', 0),
        'val_defective': dataset_stats.get('val_defective', 0),
    })
