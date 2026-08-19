# MLflow Experiment Tracking

MLflow tracks all training runs, metrics, hyperparameters, and artifacts for the Surface Anomaly Detection project.

## Quick Start

### Option 1: Using Docker Compose (Recommended)

Start MLflow server and training together:
```bash
docker-compose up
```

This will:
1. Start MLflow UI on `http://localhost:5000`
2. Start training with automatic metric logging
3. Save all artifacts to `./mlruns`

Then open your browser to `http://localhost:5000` to view the experiments.

### Option 2: Local MLflow

Run MLflow server locally:
```bash
mlflow ui --backend-store-uri ./mlruns
```

Then train normally:
```bash
python scripts/train.py config/config.yaml
```

Open `http://localhost:5000` in your browser.

## What Gets Tracked

### Hyperparameters
- `model_name`: dinov2_vitb14
- `batch_size`: Training batch size
- `crop_size`: Image crop size
- `coreset_ratio`: Coreset selection ratio
- `n_neighbors`: Number of neighbors for k-NN

### Metrics
- `auroc_image`: Image-level AUROC
- `auroc_pixel`: Pixel-level AUROC
- `f1_score`: F1 score
- `recall`: Recall score
- `precision`: Precision score

### Artifacts
- `models/patchcore_surface.pkl`: Trained model
- `metrics/metrics.json`: Full metrics JSON
- `figures/confusion_matrix.png`: Confusion matrix visualization
- `inference/results.json`: Inference results
- `inference/confusion_matrix.png`: Inference confusion matrix
- `inference/heatmaps/*.png`: Anomaly heatmap visualizations

## Comparing Experiments

1. Open MLflow UI: `http://localhost:5000`
2. Go to **Experiments** tab
3. Select "surface-anomaly-detection"
4. Click **Compare** to see runs side-by-side
5. Compare metrics, parameters, and artifacts

## Example Workflow

```bash
# Start MLflow UI and training
docker-compose up

# In another terminal, monitor progress
open http://localhost:5000

# After training, compare runs
# Go to MLflow UI and select different runs to compare
```

## Directory Structure

```
mlruns/
├── 0/                           # Experiment ID
│   ├── 1a2b3c4d.../             # Run directory
│   │   ├── artifacts/           # Logged artifacts
│   │   │   ├── models/
│   │   │   ├── metrics/
│   │   │   ├── figures/
│   │   │   └── inference/
│   │   ├── params/              # Hyperparameters
│   │   ├── metrics/             # Metrics history
│   │   └── tags/                # Run metadata
│   └── 2b3c4d5e.../
└── meta.yaml                    # Experiment metadata
```

## Programmatic Access

Log additional metrics during training:
```python
from src.mlflow_utils import setup_mlflow, start_run, end_run
import mlflow

setup_mlflow()
run = start_run('my_run')

# Log custom metrics
mlflow.log_metric('custom_metric', value=0.95)
mlflow.log_param('custom_param', 'value')

end_run()
```

## Cleanup

Remove all MLflow runs:
```bash
rm -rf mlruns/
```

## Integration with CI/CD

GitHub Actions automatically logs runs to MLflow when using Docker:
```yaml
# .github/workflows/inference.yml runs inference and logs to MLflow
docker-compose run inference
```

## Troubleshooting

**MLflow UI not loading?**
```bash
# Check if server is running
curl http://localhost:5000

# Restart
mlflow ui --backend-store-uri ./mlruns
```

**Artifacts not showing?**
```bash
# Check mlruns directory exists
ls -la mlruns/

# Verify artifact paths
mlflow server --backend-store-uri file://$(pwd)/mlruns
```

**Port 5000 already in use?**
```bash
# Use different port
mlflow ui --backend-store-uri ./mlruns --port 5001
```
