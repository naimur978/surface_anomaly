# Dual Model System: DINOv2 & EfficientNet

## Overview
Each backbone (DINOv2 and EfficientNet B4) has its **own separate trained model** saved independently. This ensures **perfect consistency**: if you train with DINOv2, inference uses DINOv2; if you train with EfficientNet, inference uses EfficientNet.

## File Structure

### Trained Models
```
results/models/
├── patchcore_surface_dinov2_vitb14.pkl     # DINOv2 trained model
└── patchcore_surface_efficientnet_b4.pkl   # EfficientNet trained model
```

### Key Code Changes
1. **Training** (`src/training.py`): Saves model with backbone name
   ```python
   feature_extractor = config['model'].get('feature_extractor', 'dinov2_vitb14')
   model_name = f"patchcore_surface_{feature_extractor}.pkl"
   ```

2. **Inference** (`scripts/run_pipeline.py`): Loads correct model based on config
   ```python
   def get_model_path(config_file):
       # Reads config to find backbone
       # Returns path to corresponding trained model
   ```

## Usage

### Train & Infer with DINOv2
```bash
# Switch to DINOv2
python scripts/switch_backbone.py dinov2_vitb14

# Train (saves as patchcore_surface_dinov2_vitb14.pkl)
# Infer (loads patchcore_surface_dinov2_vitb14.pkl)
python scripts/run_pipeline.py config/config.yaml
```

### Train & Infer with EfficientNet
```bash
# Switch to EfficientNet
python scripts/switch_backbone.py efficientnet_b4

# Train (saves as patchcore_surface_efficientnet_b4.pkl)
# Infer (loads patchcore_surface_efficientnet_b4.pkl)
python scripts/run_pipeline.py config/config.yaml
```

## What Gets Loaded at Inference

| Config Setting | Training Saves | Inference Loads | Feature Dim |
|---|---|---|---|
| `feature_extractor: "dinov2_vitb14"` | `patchcore_surface_dinov2_vitb14.pkl` | `patchcore_surface_dinov2_vitb14.pkl` | 768 |
| `feature_extractor: "efficientnet_b4"` | `patchcore_surface_efficientnet_b4.pkl` | `patchcore_surface_efficientnet_b4.pkl` | 2224 |

## Why This Matters

❌ **OLD (broken)**: Train with DINOv2 → Switch config to EfficientNet → Inference tries to use 2224-dim extractor with 768-dim model → ERROR

✅ **NEW (working)**: Train with DINOv2 → Config saved model as `dinov2_vitb14.pkl` → Inference automatically loads `dinov2_vitb14.pkl` → PERFECT MATCH

## Both Models Coexist

You can have **both trained models** in `results/models/`:
- Train with DINOv2 → Get `patchcore_surface_dinov2_vitb14.pkl`
- Switch to EfficientNet → Train → Get `patchcore_surface_efficientnet_b4.pkl`
- Switch back to DINOv2 → Use existing `patchcore_surface_dinov2_vitb14.pkl` (no retraining needed!)

## Feature Extractor Implementations

Both use the **exact same code from the notebook**:

### DINOv2
- Loads pretrained Vision Transformer
- Produces 768-dim patch embeddings
- 16x16 spatial grid (256 patches per 224×224 image)

### EfficientNet B4
- Extracts layer2 (688 dims) and layer3 (1536 dims)
- Applies neighbourhood pooling
- Upsamples layer3 to match layer2 spatial resolution
- Concatenates: 688 + 1536 = **2224 dims**
- 28x28 spatial grid (784 patches per 224×224 image)
