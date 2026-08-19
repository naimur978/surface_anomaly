# Surface Anomaly Detection - PatchCore Implementation

A production-ready implementation of the PatchCore anomaly detection algorithm for identifying defects on surface inspection tasks.

## Features

✅ **DINOv2 Feature Extraction** - 768-dimensional embeddings using Vision Transformer  
✅ **Memory Bank Coreset Selection** - 15% compression ratio for efficient k-NN scoring  
✅ **Multi-Device Support** - CUDA GPU → Metal (MacBook) GPU → CPU with automatic selection  
✅ **Threshold Optimization** - 100% recall on validation set (no missed defects)  
✅ **Training Pipeline** - Complete setup from data loading to model evaluation  
✅ **Inference System** - Batch processing with automatic ground truth extraction  
✅ **Visualization** - Selective heatmap generation for false positives  
✅ **Results Export** - JSON format with expected/predicted labels and confusion matrix  

## Quick Start

### 1. Setup
```bash
# Activate virtual environment
source venv/bin/activate
```

### 2. Train Model
```bash
python scripts/train.py
```
Trains on `data/surface/train/` and saves model to `results/models/patchcore_surface.pkl`

### 3. Run Inference
```bash
python scripts/inference.py \
  --model results/models/patchcore_surface.pkl \
  --folder data/surface/test \
  --output ./results/inference_all \
  --visualize
```

### 4. Validate Data (Optional)
```bash
python scripts/validation.py
```

## Project Structure

```
test_final/
├── src/
│   ├── patchcore_anomaly_detection.py    # Main training/feature extraction
│   ├── inference.py                       # Inference & visualizations
│   ├── utils_data.py                      # Data validation utilities
│   └── check_device.py                    # Device detection
│
├── config/
│   └── config.yaml                        # Configuration parameters
│
├── data/
│   └── surface/
│       ├── train/                         # Training data (normal images only)
│       └── test/                          # Test data
│           ├── good/                      # 520 normal images
│           └── broken_*/                  # 12+ defective images
│
├── results/
│   ├── models/patchcore_surface.pkl       # Trained model
│   ├── metrics.json                       # Training metrics
│   └── inference_all/                     # Latest inference results
│       ├── results.json                   # 536 predictions
│       ├── confusion_matrix.png           # Confusion matrix
│       └── mismatches/false_positives/    # False positive heatmaps
│
└── scripts/
    ├── train.py                           # Training script
    ├── inference.py                       # Inference script
    └── validation.py                      # Data validation script
```

## Model Performance

**Test Dataset:** 536 images (524 normal, 12 defective)

- **Recall:** 100.0% (all defects detected)
- **Precision:** 75.0% (12 TP, 4 FP)
- **Specificity:** 99.2% (520 correct negatives)
- **Accuracy:** 99.3% (532/536 correct)

### Confusion Matrix
```
              Predicted
             Normal  Defective
Actual Normal  520 (TN)   4 (FP)
       Defect.   0 (FN)  12 (TP)
```

## Technical Details

### Feature Extraction
- **Model:** DINOv2-ViT-B/14 (768-dim embeddings)
- **Patches:** 14×14 = 196 patches per image
- **Training data:** 184,320 total embeddings

### Coreset Selection
- **Method:** Greedy selection
- **Ratio:** 15% (27,648 vectors selected)
- **Purpose:** Memory efficiency for k-NN scoring

### Anomaly Scoring
- **k-NN Distance:** Euclidean distance to nearest neighbor
- **Image Score:** Maximum patch distance
- **Threshold:** 36.9051 (optimized for 100% recall)

### Device Support
Priority order:
1. CUDA (NVIDIA GPU) - fastest
2. Metal (MacBook GPU) - Apple Silicon acceleration
3. CPU - fallback

## Output Files

### results.json
Contains 536 predictions with:
```json
{
  "image_path": "data/surface/test/...",
  "anomaly_score": 47.06,
  "is_anomaly": true,
  "threshold": 36.9051,
  "expected": "defective",
  "expected_label": 1,
  "predicted": "defective",
  "predicted_label": 1,
  "correct": true
}
```

### confusion_matrix.png
2×2 confusion matrix visualization with cell counts

### Heatmap Visualizations
False positive images with 3-panel visualization:
- Left: Original image (224×224)
- Middle: Anomaly score heatmap (jet colormap)
- Right: Overlay (image + heatmap at 50% alpha)

## Logs

- **Training:** `results/training.log`
- **Inference:** `results/inference_all/inference.log`

Each log entry includes timestamp, level, module, and message.

## Dependencies

- PyTorch
- DINOv2 (vision transformer)
- scikit-learn (metrics, coreset selection)
- matplotlib (visualizations)
- Pillow (image processing)
- PyYAML (configuration)

## Configuration

Edit `config/config.yaml` to adjust:
- Device selection
- Model architecture
- Batch size and workers
- Coreset ratio
- Input image size
- Augmentation parameters

## Notes

### Model Performance
- 100% recall prioritizes catching all defects (safety-critical)
- 75% precision means 4 normal images flagged as defective
- Adjustable via threshold parameter for precision/recall trade-off

### Visualization Strategy
- Only false positives are visualized (4 images)
- Reduces output size from 536+ to just 4 visualizations
- Heatmaps help understand model behavior

### Ground Truth Extraction
- Automatic extraction from folder names: 'good' → normal, 'broken'/'defect' → defective
- Enables automatic confusion matrix generation

## Debugging

Check execution logs:
```bash
tail results/inference_all/inference.log
```

Common issues:
- **CUDA not available:** Falls back to Metal/CPU automatically
- **Out of memory:** Reduce batch_size in config.yaml
- **Missing data:** Run `python validate.py` to check data structure

## License

MIT

## Author

Implemented: 2026-08-19
