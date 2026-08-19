# Surface Anomaly Localization

Detects and localizes surface defects using DINOv2/EfficientNet feature extractors with k-NN anomaly scoring. This documentation briefly explains the methods and rationale used in the implementation.

## Installation

1. Extract the dataset folder (`Anomaly Detection Data`) and place it under the `data/` directory:

```bash
# Your dataset structure should look like:
data/
└── Anomaly Detection Data/
    ├── Feature 1/
    │   ├── mask.png
    │   ├── training/
    │   ├── test_ok/
    │   └── test_nok/
    └── Feature 2/
        ├── mask.png
        ├── training/
        ├── test_ok/
        └── test_nok/
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Activate virtual environment and run the complete pipeline (data validation → training → inference):

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start MLflow UI (optional, for experiment tracking)
python scripts/mlflow_ui.py

# Run pipeline in another terminal
python scripts/run_pipeline.py config/config.yaml
```

**Note:** While Docker support is available, we recommend running with Python scripts directly. Docker containerization has limitations with MPS (MacBook GPU), so the container runs on CPU only. For best performance, use native Python execution which supports CUDA/MPS/CPU acceleration.

## Expected Output

The pipeline generates:

```
results/
├── models/
│   └── anomaly_localization_surface_dinov2_vitb14.pkl    # Trained model
├── metrics.json                                            # AUROC, F1, threshold
├── figures/
│   ├── confusion_matrix.png                               # Confusion matrix
│   ├── evaluation.png                                      # ROC curve + score distribution
│   ├── random_heatmaps.png                                # Sample predictions with heatmaps
│   └── preprocessing_pipeline.png                         # Data preprocessing steps
├── inference_latest/
│   ├── predictions.json                                   # Per-image scores and labels
│   ├── mismatches/
│   │   └── false_positives/                              # False positive heatmaps
│   └── inference.log                                      # Inference log
└── logs/
    └── surface_anomaly.log                                # Training log
```

**MLflow Tracking:**
Open `http://localhost:5000` to view experiment metrics, parameters, and timing benchmarks.

## My Plan

## MLflow

## GitFlow

## Docker

## Code Setup Plan

## References

Baseline model comparison can be found in `baseline_model.ipynb`. For a quick overview of the project, see the summarized notebook at `notebook.ipynb`.

