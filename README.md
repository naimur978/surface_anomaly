# Surface Anomaly Localization

This work is about detecting and localizing surface defects using DINOv2/EfficientNet feature extractors with k-NN anomaly scoring. Below, I am going to explain my key decisions and rationale as briefly as possible.

**Quick Overview:** If you don't wanna go through all the scripts, I have made a brief notebook at `notebook.ipynb`. You can see my work at a glance in the notebook.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Expected Output](#expected-output)
- [My Plan](#my-plan)
  - [CI/CD Pipeline](#cicd-pipeline)
  - [Baseline Model Comparison](#baseline-model-comparison)
- [Problem Interpretation](#problem-interpretation)
- [Methodology](#methodology)
- [Key Design Decisions](#key-design-decisions)
- [Assumptions](#assumptions)
- [Evaluation Metrics](#evaluation-metrics)
- [Limitations](#limitations)
- [Potential Room for Improvements](#potential-room-for-improvements)
- [References](#references)

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

Open **two terminals** and activate virtual environment in both:

**Terminal 1 - MLflow UI (optional, for experiment tracking):**

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python scripts/mlflow_ui.py
```

I have put the code to solve port conflict. So you can directly run the URL on your browser: `http://localhost:5000`

**Terminal 2 - Run Pipeline:**

Before running the pipeline, you can change the parameters in `config/config.yaml` to customize the model, device, feature extractor, and other settings. Make sure to set the device to your hardware (cuda for NVIDIA GPU, mps for MacBook GPU, or cpu as fallback). I put mps as default as I am on MacBook.

Once you put the configuration, you can run:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python scripts/run_pipeline.py config/config.yaml
```

This will sequentially run: data validation → training → inference.

**Note:** I setup the code for both Python script and Docker. But it was challenging to run the model with GPU in the container. So if you want to run on Docker, you should use CPU. But if you want GPU, run with Python script like I gave above. Since I'm on MacBook, I ran with MPS (Metal Performance Shaders) for GPU acceleration using Python scripts.

**Optional - Docker Command:**

```bash
docker build -t surface_anomaly .
docker run -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results surface_anomaly python scripts/run_pipeline.py config/config.yaml
```

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
│   │   └── false_positives/                              # Put all false positives with heatmaps to compare
│   └── inference.log                                      # Inference log
└── logs/
    └── surface_anomaly.log                                # Training log
```

## My Plan

I wanted to make this like what I would do in an actual work setup. In DevOps, I can focus on codebase only. But MLOps is tricky because there are 3 key variables for versioning I tried to keep track of in this project:

1. **Codebase** - I used Git/GitHub with GitFlow for version control 
2. **Model Artifacts** - I used MLflow for experiment tracking and model versioning
3. **Dataset** - Dataset remains similar across experiments, so I didn't use versioning. Otherwise, I would use DVC (Data Version Control)

### CI/CD Pipeline

For continuous integration and deployment, I used:
- **Unit Testing & Automation** - Automated tests validate model loading, feature extraction, and inference pipelines
- **GitHub Actions** - Automated workflows run on every push/PR to ensure code quality and model performance:

![GitHub Actions Workflow](assets/github_actions.png)

- **Docker Containerization** - Dockerfile for reproducible deployments; handles CPU inference (GPU support recommended for production)
- **MLflow Experiment Tracking** - Centralized logging of model metrics, parameters, and artifacts:

![MLflow Tracking](assets/mlflow.png)


### Baseline Model Comparison

Initially, I read some recent CVPR papers, but then thought maybe I should start from basic approaches first. I found the **anomalib** library which allowed me to train several algorithms easily. As I used anomalib to train base models, I created the `data/surface/` folder structure following anomalib's documentation. I tried multiple models separately (PaDiM, PatchCore, GANomaly, Autoencoder) which you can find in `baseline_model.ipynb`.

Here are the results I got:

<div align="center">

| Model | AUROC |
|-------|-------|
| PaDiM | 0.6875 |
| **PatchCore** | **0.8750** |
| GANomaly | 0.5417 |
| Autoencoder | 0.3958 |

</div>

PatchCore seemed better, but that doesn't mean it will be better in the end. However, I took a leap of faith and decided to focus on improving PatchCore. My intention was to keep the ROC AUC above 0.98 to ensure it reliably separates defects from non-defects in production.

So I moved on with PatchCore as my core idea, and customized the idea, as I explained below.

## Problem Interpretation

**Objective:** Detect and localize surface defects on manufactured items using unsupervised anomaly detection. Images are captured in a controlled manner (fixed camera, consistent lighting, standardized setup) to eliminate environmental variables and focus purely on product surface anomalies.

**Challenge:**
- **No labeled defect examples** - Training data contains only "normal" samples; we have zero examples of actual defects
- **Unknown defect types** - New defect patterns may appear at inference time that were never seen during training
- **Subtle anomalies** - Some defects are barely visible as texture variations, not obvious visual flaws
- **Localization requirement** - Must pinpoint defect location (pixel-level heatmaps), not just classify image as defective
- **Limited training diversity** - Only normal surface variations in training set (material texture, manufacturing tolerances)

**Real-world Context:** This is a **one-class unsupervised classification problem** where my intention is to learn a tight boundary around "normal" samples and flag anything outside as anomalous. In manufacturing, defects reaching customers (false negatives) are catastrophic and costly. In contrast, false positives are acceptable since human reviewers filter them out at minimal cost. Therefore, the priority is **recall**: my plan is to optimize the threshold to catch every defect, even if it means accepting more false alarms. The model must err on the side of caution.

## Methodology

**Approach:** Patch-based anomaly detection using pre-trained feature extractors (DINOv2/EfficientNet) with k-NN scoring.

1. **Feature Extraction** - Extract deep features from image patches using a pre-trained vision model
2. **Coreset Selection** - Subsample representative training patches to reduce memory/compute (coreset ratio: 0.1)
3. **k-NN Scoring** - For each patch, compute anomaly score as distance to k-nearest neighbors in feature space
4. **Image-level Aggregation** - Combine patch scores to get image-level anomaly score
5. **Threshold Selection** - Use 100% recall threshold to catch all defects, accepting some false positives

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **DINOv2 backbone** | Vision Transformer features capture fine-grained structural patterns better than CNNs for surface anomalies |
| **k=9 neighbors** | Balances local context (small k too noisy) vs. generalization (large k loses sensitivity) |
| **Coreset ratio 0.1** | Reduces training patches from millions to ~100K while preserving patch diversity |
| **ROI masking** | Focuses on relevant surface area, ignoring background/borders that vary across images |
| **100% recall threshold** | Prioritizes catching defects; false positives filtered by human reviewers |
| **Patch overlap** | Sliding window with stride ensures complete coverage for smooth heatmaps |

## Assumptions

1. **Normal training samples are representative** - Training set contains diverse normal variations (lighting, angles, surface texture)
2. **Anomalies deviate significantly in feature space** - Defects create distinct patterns that k-NN can isolate
3. **Feature extractor generalizes** - Pre-trained DINOv2 transfers well without fine-tuning
4. **Threshold is stable** - 100% recall on validation generalizes to test set
5. **No label noise** - Training data correctly labeled as normal (critical for unsupervised methods)

## Evaluation Metrics

| Metric | Purpose | Target |
|--------|---------|--------|
| **AUROC (Image)** | Discrimination ability across all thresholds | > 0.98 |
| **AUROC (Pixel)** | Localization accuracy (per-pixel predictions) | > 0.90 |
| **F1 Score** | Balanced precision-recall at chosen threshold | > 0.85 |
| **F2 Score** | Recall-weighted (catch defects > precision) | > 0.80 |
| **Confusion Matrix** | TP/FP/TN/FN breakdown for business analysis | Low FN rate |

## Limitations

1. **Limited training diversity** - Model trained on 1-2 surface types; may not generalize to new defect patterns
2. **Threshold not adaptive** - Fixed threshold assumes similar defect severity; rare/subtle defects may be missed
3. **Computational cost** - Feature extraction + k-NN search slow for high-resolution images at inference time
4. **Hyperparameter sensitivity** - Coreset ratio and k-neighbors require tuning per dataset
5. **No temporal context** - Treats images independently; sequence anomalies (degradation trends) not captured
6. **Class imbalance** - Validation set may have imbalanced normal/defect ratios affecting threshold robustness

## Potential Room for Improvements

### Short-term
- **Adaptive thresholding** - Learn per-image or per-defect-type thresholds instead of global threshold
- **Ensemble methods** - Combine multiple backbones (DINOv2 + EfficientNet) for robustness
- **Hard negative mining** - Focus training on borderline false positives to tighten decision boundary

### Medium-term
- **Fine-tuning** - Adapt pre-trained features with contrastive learning on domain-specific data
- **Hierarchical k-NN** - Multi-scale patch analysis (local + global context) for better localization
- **Anomaly-specific clustering** - Separate "defect types" to learn better thresholds per category

### Long-term
- **One-class classifiers** - Replace k-NN with learned decision boundary (e.g., Support Vector Data Description)
- **Generative models** - Reconstruct normal images; anomaly = reconstruction error (GANomaly approach)
- **Semi-supervised learning** - Leverage few labeled examples to improve threshold selection
- **Active learning** - Iteratively select hard examples for human labeling to improve model

## Code Setup Plan

## References

[1] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, "MVTec AD — A comprehensive real-world dataset for unsupervised anomaly detection," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9584–9592.

[2] C. Zhou, R. C. Paffenroth, "Anomaly Detection with Robust Deep Autoencoders," in Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2017.

[3] J. Oquab, T. Darcet, T. Moutakanni, et al., "DINOv2: Learning Robust Visual Features without Supervision," arXiv preprint arXiv:2304.07193, 2023.

[4] D. Rippel, E. Mertens, and D. Merhav, "Modeling the distribution of normal data in pre-trained deep features for anomaly detection and localization," arXiv preprint arXiv:2005.14674, 2020.
