"""
Inference script for PatchCore anomaly detection model.

This script provides utilities for loading a trained model and performing
inference on new images, including visualization and heatmap generation.

Usage:
    python inference.py --model <model_path> --image <image_path> --config config.yaml
"""

import argparse
import logging
import os
from pathlib import Path
import json
import pickle

import numpy as np
# Set non-interactive matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch

from .models import FeatureExtractor
from .dataset import MVTecDataset
from .main import setup_logging
from .check_device import get_device
from .mlflow_utils import setup_mlflow, start_run, end_run, log_inference_results
import mlflow

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Wrapper for PatchCore inference"""

    def __init__(self, model_path, config=None, device='cuda', logger=None):
        """
        Initialize anomaly detector.

        Args:
            model_path: Path to trained model
            config: Configuration dictionary
            device: Torch device
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device(device)

        # Load model
        self.logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Setup feature extractor
        model_name = config.get('model', {}).get('feature_extractor', 'dinov2_vitb14') \
            if config else 'dinov2_vitb14'

        self.extractor = FeatureExtractor(
            device=self.device,
            model_name=model_name,
            logger=self.logger
        )

        self.config = config or {}
        self.threshold = None

    def set_threshold(self, threshold):
        """Set anomaly detection threshold"""
        self.threshold = threshold
        self.logger.info(f"Threshold set to {threshold:.4f}")

    def detect(self, image_path, return_heatmap=True):
        """
        Detect anomalies in a single image.

        Args:
            image_path: Path to image
            return_heatmap: Whether to return heatmap

        Returns:
            Dictionary with scores and heatmap
        """
        # Load and preprocess image
        try:
            img = Image.open(image_path).convert('RGB')
        except (FileNotFoundError, OSError) as e:
            raise FileNotFoundError(f"Cannot open image: {image_path}") from e

        # Use MVTecDataset transform (suppress dataset info printing)
        import logging
        logging.getLogger('surface_anomaly_detection').setLevel(logging.WARNING)

        root_dir = self.config.get('data', {}).get('root_dir', './data')
        if not root_dir.startswith('/'):
            root_dir = str(Path(root_dir).resolve())

        dataset = MVTecDataset(
            root_dir,
            self.config.get('data', {}).get('category', 'surface'),
            split='test',
            crop_size=self.config.get('image', {}).get('crop_size', 224)
        )

        logging.getLogger('surface_anomaly_detection').setLevel(logging.INFO)

        img_t = dataset.transform(img)
        img_batch = img_t.unsqueeze(0)

        # Extract features
        patches, hw = self.extractor.extract(img_batch)
        patches = patches[0].numpy()

        # Compute scores
        image_score = self.model.score_image(patches)
        is_anomaly = image_score > self.threshold if self.threshold else None

        result = {
            'image_path': str(image_path),
            'anomaly_score': float(image_score),
            'is_anomaly': bool(is_anomaly) if is_anomaly is not None else None,
            'threshold': self.threshold,
        }

        if return_heatmap:
            heatmap = self.model.score_map(patches)
            result['heatmap'] = heatmap

        return result

    def batch_detect(self, image_folder, return_heatmaps=False):
        """
        Detect anomalies in multiple images.

        Args:
            image_folder: Folder containing images
            return_heatmaps: Whether to return heatmaps

        Returns:
            List of detection results
        """
        import sys
        from io import StringIO

        results = []
        image_paths = list(Path(image_folder).rglob("*.png")) + \
                     list(Path(image_folder).rglob("*.jpg"))

        self.logger.info(f"Processing {len(image_paths)} images...")

        # Suppress print output from dataset loading
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        for idx, img_path in enumerate(image_paths):
            try:
                result = self.detect(img_path, return_heatmap=return_heatmaps)
                results.append(result)

                if (idx + 1) % 10 == 0:
                    sys.stdout = old_stdout
                    self.logger.info(f"  Processed {idx + 1}/{len(image_paths)}")
                    sys.stdout = StringIO()
            except Exception as e:
                sys.stdout = old_stdout
                self.logger.error(f"Error processing {img_path}: {str(e)}")
                sys.stdout = StringIO()

        # Restore stdout
        sys.stdout = old_stdout
        return results


def visualize_detection(image_path, result, save_path=None):
    """
    Visualize anomaly detection result.

    Args:
        image_path: Path to original image
        result: Detection result dictionary
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    # Original image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0

    axes[0].imshow(img_array)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Heatmap
    if 'heatmap' in result:
        heatmap = result['heatmap']
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title("Anomaly Heatmap")
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(img_array)
        axes[2].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[2].set_title("Overlay")
        axes[2].axis('off')

    # Add title with score
    score = result['anomaly_score']
    is_anomaly = result['is_anomaly']
    status = "ANOMALY" if is_anomaly else "NORMAL"
    color = 'red' if is_anomaly else 'green'

    fig.suptitle(f"{status} - Score: {score:.4f}", fontsize=14, color=color, weight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")

    # Close figure to free memory and avoid interactive display
    plt.close(fig)


def main():
    """Main inference script"""
    parser = argparse.ArgumentParser(description='PatchCore Anomaly Detection Inference')

    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pkl)')
    parser.add_argument('--image', type=str,
                       help='Path to single image')
    parser.add_argument('--folder', type=str,
                       help='Path to folder with images')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--threshold', type=float,
                       help='Anomaly detection threshold')
    parser.add_argument('--output', type=str, default='./results/inference',
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')

    args = parser.parse_args()

    # Setup logging
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "inference.log"

    logger = setup_logging(log_file)

    logger.info("="*60)
    logger.info("PATCHCORE ANOMALY DETECTION - INFERENCE")
    logger.info("="*60)

    # Initialize MLflow
    setup_mlflow(experiment_name='surface-anomaly-detection')
    run = start_run(run_name='surface_anomaly_inference')
    logger.info(f"MLflow run started: {run.info.run_id}")

    # Load config
    import yaml
    config = {}
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    # Determine device
    if args.device == 'cuda':
        device = get_device()  # Auto-select best device
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Initialize detector
    detector = AnomalyDetector(args.model, config=config, device=str(device), logger=logger)

    # Load threshold from metrics.json or use provided threshold
    threshold_to_use = None

    if args.threshold:
        threshold_to_use = args.threshold
        logger.info(f"Using provided threshold: {threshold_to_use:.4f}")
    else:
        # Try to load from results/metrics.json
        metrics_file = Path("results/metrics.json")
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                # Use recall_threshold (100% recall) by default
                threshold_to_use = metrics.get('recall_threshold')
                if threshold_to_use:
                    logger.info(f"Loaded recall threshold from metrics.json: {threshold_to_use:.4f}")
                else:
                    threshold_to_use = metrics.get('threshold')
                    logger.info(f"Loaded threshold from metrics.json: {threshold_to_use:.4f}")
            except Exception as e:
                logger.warning(f"Could not load metrics.json: {str(e)}")
        else:
            logger.warning("No metrics.json found. Using threshold without it.")

    if threshold_to_use:
        detector.set_threshold(threshold_to_use)

    # Run inference
    results = []

    if args.image:
        logger.info(f"\nProcessing single image: {args.image}")
        result = detector.detect(args.image, return_heatmap=True)
        results = [result]

        logger.info(f"Score: {result['anomaly_score']:.4f}")
        logger.info(f"Status: {'ANOMALY' if result['is_anomaly'] else 'NORMAL'}")

        if args.visualize:
            viz_path = output_dir / "visualization.png"
            visualize_detection(args.image, result, save_path=str(viz_path))

    elif args.folder:
        logger.info(f"\nProcessing folder: {args.folder}")
        results = detector.batch_detect(args.folder, return_heatmaps=False)

        if args.visualize:
            logger.info("\nIdentifying mismatched predictions...")

            # Identify mismatches
            mismatches = []
            for idx, result in enumerate(results):
                path = result['image_path']
                if 'good' in path:
                    expected = 'normal'
                elif 'broken' in path or 'defect' in path:
                    expected = 'defective'
                else:
                    expected = None

                predicted = 'defective' if result['is_anomaly'] else 'normal'

                if expected and expected != predicted:
                    mismatches.append((idx, result, expected, predicted))

            # Generate visualizations for all mismatches
            if mismatches:
                logger.info(f"\nGenerating heatmap visualizations for {len(mismatches)} mismatched predictions...")
                mismatches_dir = output_dir / "mismatches"
                mismatches_dir.mkdir(exist_ok=True)

                for idx, result, expected, predicted in mismatches:
                    mismatch_subdir = mismatches_dir / "false_positives"
                    mismatch_subdir.mkdir(exist_ok=True)

                    try:
                        # Extract heatmap for this mismatch
                        mismatch_result = detector.detect(result['image_path'], return_heatmap=True)
                        mismatch_path = mismatch_subdir / f"mismatch_{idx:04d}.png"
                        visualize_detection(result['image_path'], mismatch_result, save_path=str(mismatch_path))
                    except Exception as e:
                        logger.error(f"Error visualizing {result['image_path']}: {str(e)}")

                logger.info(f"✓ Generated {len(mismatches)} heatmap visualizations")
            else:
                logger.info("\nNo mismatched predictions found!")

    else:
        logger.error("Either --image or --folder must be specified")
        return

    # Save results
    logger.info(f"\nSaving results to {output_dir}...")

    results_clean = []
    for r in results:
        clean = {k: v for k, v in r.items() if k != 'heatmap'}

        # Add ground truth label (expected) based on folder structure
        path = r['image_path']
        if 'good' in path:
            clean['expected'] = 'normal'
            clean['expected_label'] = 0
        elif 'broken' in path or 'defect' in path:
            clean['expected'] = 'defective'
            clean['expected_label'] = 1
        else:
            clean['expected'] = 'unknown'
            clean['expected_label'] = None

        # Add predicted label
        clean['predicted'] = 'defective' if clean['is_anomaly'] else 'normal'
        clean['predicted_label'] = 1 if clean['is_anomaly'] else 0

        # Add match status
        clean['correct'] = (clean['expected_label'] == clean['predicted_label'])

        results_clean.append(clean)

    with open(output_dir / "results.json", 'w') as f:
        json.dump(results_clean, f, indent=4)

    # Summary statistics
    if results:
        scores = [r['anomaly_score'] for r in results]
        anomalies = [r['is_anomaly'] for r in results if r['is_anomaly'] is not None]

        logger.info("\n" + "="*60)
        logger.info("INFERENCE SUMMARY")
        logger.info("="*60)
        logger.info(f"Total images: {len(results)}")
        logger.info(f"Anomalies detected: {sum(anomalies)}")
        logger.info(f"Normal images: {len(anomalies) - sum(anomalies)}")
        logger.info(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
        logger.info(f"Mean score: {np.mean(scores):.4f}")
        logger.info("="*60)

        # Check if we can generate confusion matrix (i.e., if ground truth is available)
        try:
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            import matplotlib.pyplot as plt

            # Extract ground truth from folder structure if available
            test_labels = []
            for result in results:
                path = result['image_path']
                # Check if path contains 'good' or defect category
                if 'good' in path:
                    test_labels.append(0)  # Normal
                elif 'broken' in path or 'defect' in path:
                    test_labels.append(1)  # Defective
                else:
                    test_labels.append(None)

            # Only generate confusion matrix if we have ground truth
            if None not in test_labels and len(test_labels) == len(anomalies):
                preds = [1 if a else 0 for a in anomalies]
                cm = confusion_matrix(test_labels, preds)

                # Save confusion matrix
                fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
                disp = ConfusionMatrixDisplay(cm, display_labels=["normal", "defective"])
                disp.plot(cmap="Blues", values_format="d", ax=ax)
                cm_path = output_dir / "confusion_matrix.png"
                plt.savefig(cm_path, dpi=150, bbox_inches="tight")
                plt.close()

                logger.info(f"\n✓ Saved confusion matrix to {cm_path}")
                logger.info(f"  True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
                logger.info(f"  False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")
        except Exception as e:
            logger.warning(f"Could not generate confusion matrix: {str(e)}")

    # Log results to MLflow
    log_inference_results(output_dir.parent, output_dir.name)

    # Log inference metrics if ground truth available
    if results:
        scores = [r['anomaly_score'] for r in results]
        try:
            # Extract ground truth
            test_labels = []
            for result in results:
                path = result['image_path']
                if 'good' in path:
                    test_labels.append(0)
                elif 'broken' in path or 'defect' in path:
                    test_labels.append(1)
                else:
                    test_labels.append(None)

            # Only log if we have ground truth
            if None not in test_labels:
                from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
                threshold = detector.threshold if hasattr(detector, 'threshold') else np.mean(scores)
                preds = (np.array(scores) > threshold).astype(int)

                cm = confusion_matrix(test_labels, preds)
                tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

                # Log metrics
                mlflow.log_metrics({
                    'inference_accuracy': accuracy_score(test_labels, preds),
                    'inference_precision': precision_score(test_labels, preds, zero_division=0),
                    'inference_recall': recall_score(test_labels, preds, zero_division=0),
                    'inference_f1': f1_score(test_labels, preds, zero_division=0),
                    'inference_tn': tn,
                    'inference_fp': fp,
                    'inference_fn': fn,
                    'inference_tp': tp,
                })
                logger.info("Inference metrics logged to MLflow")
        except Exception as e:
            logger.warning(f"Could not log inference metrics: {e}")

    # Print final summary with output locations (clean format, no timestamps)
    print("\n" + "="*70)
    print("INFERENCE COMPLETED")
    print("="*70)
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("\nOutput files:")

    # List all output files
    if (output_dir / "results.json").exists():
        print(f"  - results.json")
    if (output_dir / "confusion_matrix.png").exists():
        print(f"  - confusion_matrix.png")
    if (output_dir / "inference.log").exists():
        print(f"  - inference.log")

    logger.info(f"MLflow UI: mlflow ui --backend-store-uri ./mlruns")

    # Show false positives if any
    mismatches_dir = output_dir / "mismatches"
    if mismatches_dir.exists():
        false_positives_dir = mismatches_dir / "false_positives"
        if false_positives_dir.exists():
            count = len(list(false_positives_dir.glob("*.png")))
            print(f"\n  False Positive Heatmaps: {count} images")
            print(f"    - mismatches/false_positives/")

    print("="*70 + "\n")

    # End MLflow run
    end_run()


if __name__ == "__main__":
    main()
