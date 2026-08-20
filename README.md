# Surface Anomaly Localization

This work is about detecting and localizing surface defects using DINOv2/EfficientNet feature extractors with k-NN anomaly scoring. Below, I am going to explain my key decisions and rationale as briefly as possible.

**Quick Overview:** If you don't wanna go through all the scripts, I have made a brief notebook at `notebook.ipynb`. You can see my work at a glance in the notebook.

## Table of Contents

1. [Installation](#1-installation)
2. [Usage](#2-usage)
3. [Expected Output](#3-expected-output)
4. [My Plan](#4-my-plan)
   - [4.1 CI/CD Pipeline](#41-cicd-pipeline)
   - [4.2 Baseline Model Comparison](#42-baseline-model-comparison)
5. [Problem Interpretation](#5-problem-interpretation)
6. [Methodology](#6-methodology)
   - [6.1 How It Works](#61-how-it-works)
   - [6.2 Why I Chose Patch-Based Approach](#62-why-i-chose-patch-based-approach)
7. [Key Design Decisions](#7-key-design-decisions)
   - [7.1 Memory Bank Strategy](#71-memory-bank-strategy-unified-vs-separate)
   - [7.2 Image Size](#72-image-size-224224)
   - [7.3 Data Augmentation](#73-data-augmentation-none-for-training)
   - [7.4 ROI Masking](#74-roi-masking)
   - [7.5 Feature Extractor](#75-feature-extractor-dinov2)
   - [7.6 Patch Overlap](#76-patch-overlap)
   - [7.7 Coreset Ratio](#77-coreset-sampling-ratio-01)
   - [7.8 Indexing Strategy](#78-indexing-strategy-in-memory-k-nn)
   - [7.9 Number of Neighbors](#79-number-of-neighbors-k9)
   - [7.10 Threshold Strategy](#710-threshold-strategy-100-recall)
8. [Assumptions](#8-assumptions)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Limitations](#10-limitations)
11. [Potential Room for Improvements](#11-potential-room-for-improvements)
    - [11.1 Short-term](#111-short-term)
    - [11.2 Medium-term](#112-medium-term)
    - [11.3 Long-term](#113-long-term)
12. [Code Setup Plan](#12-code-setup-plan)
13. [References](#13-references)

## 1. Installation

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

## 2. Usage

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

## 3. Expected Output

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

## 4. My Plan

I wanted to make this like what I would do in an actual work setup. In DevOps, I can focus on codebase only. But MLOps is tricky because there are 3 key variables for versioning I tried to keep track of in this project:

1. **Codebase** - I used Git/GitHub with GitFlow for version control 
2. **Model Artifacts** - I used MLflow for experiment tracking and model versioning
3. **Dataset** - Dataset remains similar across experiments, so I didn't use versioning. Otherwise, I would use DVC (Data Version Control)

### 4.1 CI/CD Pipeline

For continuous integration and deployment, I used:
- **Unit Testing & Automation** - Automated tests validate model loading, feature extraction, and inference pipelines
- **GitHub Actions** - Automated workflows run on every push/PR to ensure code quality and model performance:

![GitHub Actions Workflow](assets/github_actions.png)

- **Docker Containerization** - Dockerfile for reproducible deployments. Handles CPU inference (GPU support recommended for production)
- **MLflow Experiment Tracking** - Centralized logging of model metrics, parameters, and artifacts:

![MLflow Tracking](assets/mlflow.png)


### 4.2 Baseline Model Comparison

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

## 5. Problem Interpretation

**Objective:** Detect and localize surface defects on manufactured items using unsupervised anomaly detection. Images are captured in a controlled manner (fixed camera, consistent lighting, standardized setup) to eliminate environmental variables and focus purely on product surface anomalies.

**Challenge:**
- **No labeled defect examples** - Training data contains only "normal" samples. We have zero examples of actual defects
- **Unknown defect types** - New defect patterns may appear at inference time that were never seen during training
- **Subtle anomalies** - Some defects are barely visible as texture variations, not obvious visual flaws
- **Localization requirement** - Must pinpoint defect location (pixel-level heatmaps), not just classify image as defective
- **Limited training diversity** - Only normal surface variations in training set (material texture, manufacturing tolerances)

**Real-world Context:** This is a **one-class unsupervised classification problem** where my intention is to learn a tight boundary around "normal" samples and flag anything outside as anomalous. In manufacturing, defects reaching customers (false negatives) are catastrophic and costly. In contrast, false positives are acceptable since human reviewers filter them out at minimal cost. Therefore, the priority is **recall**: my plan is to optimize the threshold to catch every defect, even if it means accepting more false alarms. The model must err on the side of caution.

## 6. Methodology

**Approach:** Patch-based anomaly detection. Divide images into overlapping patches, extract features using pre-trained models, build a reference set of normal patches, then score test patches based on deviation from this normal distribution.

### 6.1 How It Works

1. Extract feature vectors from overlapping image patches using pre-trained models
2. Store all patch features from normal training data in a memory bank
3. Subsample the memory bank using coreset selection, keeping only the most representative patches
4. For test images, extract patch features and compute similarity/distance to the memory bank
5. Patches with low similarity (high distance) to normal patterns are flagged as anomalous
6. Map patch-level anomaly scores back to image space to create localization heatmaps
7. Aggregate patch scores to produce image-level prediction (defect or normal)

### 6.2 Why I Chose Patch-Based Approach

I needed localization (spatial heatmaps) not just classification. Patch-based methods provide spatial information, work with only normal samples (one-class learning), and are sensitive to subtle texture anomalies. Additionally, PatchCore offers many tunable hyperparameters (coreset ratio, k-neighbors, feature extractor) unlike autoencoders which rely mostly on fine-tuning. This flexibility is crucial given my limited training data. Given my limited time, I chose a technique that is algorithm-based (k-NN, feature engineering, threshold tuning) rather than model-based (requiring extensive training or fine-tuning). This allowed me to iterate faster and achieve good results without waiting for multiple training runs.

## 7. Key Design Decisions

### 7.1 Memory Bank Strategy (Unified vs. Separate)
**Decision:** Store all feature types (Feature 1 and Feature 2) in a single unified memory bank.  
**Rationale:** I tried both approaches separately and combined on my dataset. Both feature types represent normal surface variations of different metal surfaces. I didn't observe any significant performance drop when combining them into a unified memory bank. A unified memory bank learns a general "normal" boundary across both types, simplifying deployment and threshold management. Since combining them didn't hurt my model's performance, I decided to move on with the unified dataset approach.  
**Trade-off:** Risk of mixing feature distributions if they are too different. Single threshold may not optimize for both types. I would use separate memory banks only if:
  - The surfaces are made of fundamentally different materials (e.g., textile vs. metal)
  - Normal patterns diverge significantly
  - Feature-specific thresholds yield better results in production testing


### 7.2 Image Size (224×224)
**Decision:** Standardize all images to 224×224 pixels.  
**Rationale:** I chose this because it matches pre-trained model input dimensions (DINOv2 expects 224×224). I found it simplifies batch processing and feature extraction.  
**Trade-off:** Resizing may lose fine-grained details for very high-resolution defects, but I believe maintaining consistency and computational efficiency is more important.

### 7.3 Data Augmentation (None for Training)
**Decision:** No augmentation on training data. Train only on original normal samples.  
**Rationale:** I reason that training data is "normal only". Artificial augmentation could introduce unrealistic variations that distort the normal boundary in feature space. I want to avoid synthetic bias.  
**Trade-off:** I accept losing robustness to lighting/angle variations at test time, but I believe maintaining a tight normal boundary without synthetic bias is more critical.

### 7.4 ROI Masking
**Decision:** Apply region-of-interest (ROI) mask to focus on surface area only.  
**Rationale:** I observed that background/borders vary across images. Masking eliminates this noise, allowing the model to focus on what matters: product surface anomalies. My observation is that this significantly improves signal-to-noise ratio.  
**Trade-off:** This requires manual ROI definition and is inflexible to product shape changes, but I believe the improved signal-to-noise justifies it.

### 7.5 Feature Extractor (DINOv2)
**Decision:** Use Vision Transformer-based features (DINOv2) instead of CNN-based features.  
**Rationale:** I found that Vision Transformers capture fine-grained structural patterns better than CNNs. My observation is they excel at detecting subtle texture anomalies in surface defects, which is critical for my use case.  
**Trade-off:** I accept higher computational cost at inference compared to lightweight CNNs, but I believe better anomaly sensitivity justifies it.

### 7.6 Patch Overlap
**Decision:** Use overlapping patches with sliding window.  
**Rationale:** I observe that overlapping ensures complete spatial coverage and smooth heatmaps without blind spots. I think this is essential for precise anomaly localization.  
**Trade-off:** I accept increased computational cost due to redundant feature extraction, but I believe it's necessary for the localization accuracy I need.

### 7.7 Coreset Sampling Ratio (0.1)
**Decision:** Retain 10% of training patches after coreset sampling.  
**Rationale:** I need to reduce memory and computation from millions of patches to ~100K for practical deployment. I believe coreset algorithms preserve the most representative samples while maintaining anomaly detection capability.  
**Trade-off:** Aggressive sampling (10%) risks losing important normal patterns, but I think modern coreset selection is smart enough to capture the essential distribution.

### 7.8 Indexing Strategy (In-Memory k-NN)
**Decision:** Store coreset patches in memory and use exact k-NN search.  
**Rationale:** I want fast inference with guaranteed correctness. My observation is that ~100K coreset patches fit comfortably in memory, and I prefer the simplicity and interpretability of exact search over approximate methods.  
**Trade-off:** I acknowledge this is memory-intensive for very large coresets (millions). Approximate methods (FAISS, LSH) could scale better, but I believe added complexity isn't worth it for my current scale.

### 7.9 Number of Neighbors (k=9)
**Decision:** Set k=9 for nearest neighbor search.  
**Rationale:** I need to balance two tensions: small k is too noisy, large k loses sensitivity to anomalies. Through experimentation, I found k=9 works well for my specific dataset and recall requirements.  
**Trade-off:** I've observed higher k gives smoother but less sensitive results. Lower k is more sensitive but noisier. I chose k=9 as the empirical sweet spot for my use case.

### 7.10 Threshold Strategy (100% Recall)
**Decision:** Use 100% recall threshold to catch every defect.  
**Rationale:** I believe in manufacturing, missing a defect is catastrophic and unacceptable. My observation is that false positives are acceptable since human reviewers can filter them out. I prioritize recall above all else.  
**Trade-off:** I accept high false positive rate as the necessary cost, but I gain the guarantee of zero false negatives. No defective products slip through.

## 8. Assumptions

1. **Normal training samples are representative** - Training set contains diverse normal variations (lighting, angles, surface texture)
2. **Anomalies deviate significantly in feature space** - Defects create distinct patterns that k-NN can isolate
3. **Feature extractor generalizes** - Pre-trained DINOv2 transfers well without fine-tuning
4. **Threshold is stable** - 100% recall on validation generalizes to test set
5. **No label noise** - Training data correctly labeled as normal (critical for unsupervised methods)

## 9. Evaluation Metrics

| Metric | Purpose | Target |
|--------|---------|--------|
| **AUROC (Image)** | Discrimination ability across all thresholds | > 0.98 |
| **AUROC (Pixel)** | Localization accuracy (per-pixel predictions) | > 0.90 |
| **F1 Score** | Balanced precision-recall at chosen threshold | > 0.85 |
| **F2 Score** | Recall-weighted (catch defects > precision) | > 0.80 |
| **Confusion Matrix** | TP/FP/TN/FN breakdown for business analysis | Low FN rate |

## 10. Limitations

1. **Limited training diversity** - Model trained on 1-2 surface types. May not generalize to new defect patterns
2. **Threshold not adaptive** - Fixed threshold assumes similar defect severity. Rare/subtle defects may be missed
3. **Computational cost** - Feature extraction + k-NN search slow for high-resolution images at inference time
4. **Hyperparameter sensitivity** - Coreset ratio and k-neighbors require tuning per dataset
5. **No temporal context** - Treats images independently. Sequence anomalies (degradation trends) not captured
6. **Class imbalance** - Validation set may have imbalanced normal/defect ratios affecting threshold robustness

## 11. Potential Room for Improvements

### 11.1 Short-term
- **Adaptive thresholding** - Learn per-image or per-defect-type thresholds instead of global threshold
- **Ensemble methods** - Combine multiple backbones (DINOv2 + EfficientNet) for robustness
- **Hard negative mining** - Focus training on borderline false positives to tighten decision boundary

### 11.2 Medium-term
- **Fine-tuning** - Adapt pre-trained features with contrastive learning on domain-specific data
- **Hierarchical k-NN** - Multi-scale patch analysis (local + global context) for better localization
- **Anomaly-specific clustering** - Separate "defect types" to learn better thresholds per category

### 11.3 Long-term
- **One-class classifiers** - Replace k-NN with learned decision boundary (e.g., Support Vector Data Description)
- **Generative models** - Reconstruct normal images. Anomaly = reconstruction error (GANomaly approach)
- **Semi-supervised learning** - Leverage few labeled examples to improve threshold selection
- **Active learning** - Iteratively select hard examples for human labeling to improve model

## 12. Code Setup Plan

## 13. References

[1] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, "MVTec AD — A comprehensive real-world dataset for unsupervised anomaly detection," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9584–9592.

[2] C. Zhou, R. C. Paffenroth, "Anomaly Detection with Robust Deep Autoencoders," in Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2017.

[3] J. Oquab, T. Darcet, T. Moutakanni, et al., "DINOv2: Learning Robust Visual Features without Supervision," arXiv preprint arXiv:2304.07193, 2023.

[4] D. Rippel, E. Mertens, and D. Merhav, "Modeling the distribution of normal data in pre-trained deep features for anomaly detection and localization," arXiv preprint arXiv:2005.14674, 2020.
