"""MLflow experiment tracking utilities."""

import os
# Allow file store to avoid maintenance mode warning
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'

import mlflow
import mlflow.pytorch
import json
from pathlib import Path


def setup_mlflow(experiment_name='surface-anomaly-detection', tracking_uri='./mlruns'):
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


def log_training_params(config):
    """Log training hyperparameters to MLflow.

    Args:
        config: Config dictionary
    """
    # Log model parameters
    mlflow.log_params({
        'feature_extractor': config['model'].get('feature_extractor', 'dinov2_vitb14'),
        'batch_size': config['training'].get('batch_size', 32),
        'crop_size': config['image'].get('crop_size', 224),
        'coreset_ratio': config['model'].get('coreset_ratio', 0.15),
        'n_neighbors': config['model'].get('n_neighbors', 9),
        'seed': config['training'].get('seed', 42),
    })


def log_training_metrics(metrics):
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


def log_model(model_path, model_name='surface_anomaly'):
    """Log trained model to MLflow.

    Args:
        model_path: Path to saved model
        model_name: Name of the model
    """
    model_path = Path(model_path)
    if model_path.exists():
        mlflow.log_artifact(str(model_path), artifact_path='models')


def log_artifacts(results_dir):
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


def log_inference_results(results_dir, output_name='inference_latest'):
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


def start_run(run_name, tags=None):
    """Start a new MLflow run.

    Args:
        run_name: Name of the run
        tags: Dictionary of tags
    """
    run = mlflow.start_run(run_name=run_name)
    if tags:
        mlflow.set_tags(tags)
    return run


def end_run():
    """End the current MLflow run."""
    mlflow.end_run()


def log_dataset_info(dataset_stats):
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
