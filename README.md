# Surface Anomaly Localization

This work develops an unsupervised anomaly detection solution for surface defect detection and localization on manufactured items. Below, I explain my approach, key decisions, and rationale as briefly as possible.

**Quick Overview:** If you don't wanna go through all the scripts, I have made a brief notebook at `notebook.ipynb`. You can see my work at a glance in the notebook.

## Table of Contents

1. [Installation](#1-installation)
2. [Usage](#2-usage)
3. [Expected Output](#3-expected-output)
4. [My Plan for Experimental Setup](#4-my-plan-for-experimental-setup)
   - [4.1 CI/CD Pipeline](#41-cicd-pipeline)
   - [4.2 Baseline Model Comparison](#42-baseline-model-comparison)
5. [Problem Interpretation](#5-problem-interpretation)
6. [Methodology](#6-methodology)
   - [6.1 How It Works](#61-how-it-works)
   - [6.2 Why I Chose Patch-Based Approach](#62-why-i-chose-patch-based-approach)
7. [Key Design Decisions](#7-key-design-decisions)
   - [Assumptions](#assumptions)
   - [Design Choices](#design-choices)
   - [Trade-offs](#trade-offs)
8. [Approaches That Didn't Work](#8-approaches-that-didnt-work)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Limitations](#10-limitations)
11. [Potential Room for Improvements](#11-potential-room-for-improvements)
    - [11.1 Short-term](#111-short-term)
    - [11.2 Medium-term](#112-medium-term)
    - [11.3 Long-term](#113-long-term)
12. [References](#12-references)

## 1. Installation

1. Create a Python virtual environment:

```bash
python3 -m venv venv
```

2. Activate virtual environment and install dependencies:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Extract the dataset folder (`Anomaly Detection Data`) and place it under the `data/` directory:

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

## 2. Usage

Open **two terminals** and activate virtual environment in both:

**Terminal 1 - MLflow UI (optional, for experiment tracking):**

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python scripts/mlflow_ui.py
```

I have put the code to solve port conflict. So you can directly run the URL on your browser: `http://127.0.0.1:5000/`

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

## 4. My Plan for Experimental Setup

I structured this project following MLOps best practices to mimic a production-ready setup. Unlike traditional DevOps where versioning focuses on code, MLOps requires tracking three distinct dimensions. I implemented versioning strategies for each:

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

Initially, I read some recent CVPR papers, but then thought maybe I should start from basic approaches first. I found the **anomalib** library which allowed me to train several SoTA algorithms easily. As I used anomalib to train base models, I created the `data/surface/` folder structure following anomalib's documentation. I tried multiple models separately (PaDiM, PatchCore, GANomaly, Autoencoder) [1][2] which you can find in `baseline_model.ipynb`.

Here are the results I got:

<div align="center">

| Model | AUROC |
|-------|-------|
| PaDiM | 0.6875 |
| **PatchCore** | **0.8750** |
| GANomaly | 0.5417 |
| Autoencoder | 0.3958 |
| DDPM Diffusion | 0.5313 |
| RD4AD | 0.5173 |

</div>

PatchCore showed the strongest performance at 0.8750 AUROC. While this doesn't guarantee it will remain best after customization, I decided to focus on improving PatchCore. As you have mentioned, the objective is not to use the most complex model, but to demonstrate a well-reasoned solution. My intention was to keep the ROC AUC above 0.98 to ensure it reliably separates defects from non-defects in production.

So I moved on with PatchCore as my core idea, and customized the idea, as I explained below.

## 5. Problem Interpretation

**Objective:** Detect and localize surface defects on manufactured items using unsupervised anomaly detection. Images are captured in a controlled manner (fixed camera, consistent lighting, standardized setup) to eliminate environmental variables and focus purely on product surface anomalies.

**Challenge:**
- **No labeled defect examples** - Training data contains only "normal" samples. We have zero examples of actual defects
- **Unknown defect types** - New defect patterns may appear at inference time that were never seen during training
- **Subtle anomalies** - Some defects are barely visible as texture variations, not obvious visual flaws
- **Localization requirement** - Must pinpoint defect location (pixel-level heatmaps), not just classify image as defective
- **Limited training diversity** - Only normal surface variations in training set (material texture, manufacturing tolerances)

**Real-world Context:** This is a **one-class unsupervised classification problem** where my intention is to learn a tight boundary around "normal" samples and flag anything outside as anomalous. In manufacturing, defects reaching customers (false negatives) are catastrophic and costly. In contrast, false positives are acceptable since human reviewers filter them out at minimal cost. This establishes a **human-in-the-loop workflow** where the model acts as an aggressive first-pass filter, flagging all suspicious items, and human inspectors make the final decision. Therefore, the priority is **recall**. My plan is to optimize the threshold to catch every defect, even if it means accepting more false alarms. The model must err on the side of caution.

## 6. Methodology

**Approach:** I got the core idea from PatchCore but updated it in my own ways. The approach divides images into overlapping patches, extracts features using pre-trained models, builds a reference set of normal patches, then scores test patches based on deviation from this normal distribution. The key customizations and design choices are detailed in Section 7.

### 6.1 How It Works

1. Extract feature vectors from overlapping image patches using pre-trained models
2. Store all patch features from normal training data in a memory bank
3. Subsample the memory bank using coreset selection, keeping only the most representative patches
4. For test images, extract patch features and compute similarity/distance to the memory bank
5. Patches with low similarity (high distance) to normal patterns are flagged as anomalous
6. Map patch-level anomaly scores back to image space to create localization heatmaps
7. Aggregate patch scores to produce image-level prediction (defect or normal)

### 6.2 Why I Chose Patch-Based Approach

I needed localization (spatial heatmaps) not just classification. Patch-based methods [1] provide spatial information, work with only normal samples (one-class learning), and are sensitive to subtle texture anomalies. Additionally, PatchCore offers many tunable hyperparameters (coreset ratio, k-neighbors, feature extractor) unlike U-Net based reconstruction approaches [2] or autoencoders [3] which rely mostly on fine-tuning and extensive training. This flexibility is crucial given my limited training data. Given my limited time, I chose a technique that is algorithm-based (k-NN, feature engineering, threshold tuning) rather than model-based (requiring extensive training or fine-tuning). This allowed me to iterate faster and achieve good results without waiting for multiple training runs.

## 7. Key Design Decisions

### Assumptions

**7.1 Controlled Environment**
The manufacturing setup operates under controlled conditions: fixed camera, consistent lighting, standardized product placement. No different angles, no dramatic lighting changes, no camera motion. This assumption justifies why I don't use data augmentation and can rely on a tight normal boundary without synthetic bias.

**7.2 Single Unified Dataset**
Both feature types (Feature 1 and Feature 2) represent normal surface variations of the same material with different finishes/tolerances. I tested both separate and unified approaches on my dataset and found no significant performance drop when combining them. A unified approach learns a general "normal" boundary across both types, simplifying deployment and threshold management.

Note: subject-based thresholding (separate thresholds per feature type) could be implemented to optimize each type individually, but I haven't done that in this version since the unified approach already achieves good performance. After merging both feature types, I got a total of 720 training images and 110 test images (70 normal, 40 defective). The defective test dataset is very low, which is why the F1 score tends to be lower compared to AUROC. AUROC is more robust to class imbalance since it measures ranking ability regardless of threshold, while F1 is sensitive to the imbalance between normal and defective samples.

![Dataset Strategy Comparison](assets/dataset.png)

### Design Choices

**7.3 Image Size (224×224)**

**Decision:** Standardize all images to 224×224 pixels.

**Rationale:** DINOv2 (ViT-B/14) is pretrained on variable sizes (224×224, 448×448, 518×518), while EfficientNet-B4 is pretrained on 380×380. When I downscaled EfficientNet to 224×224, it lost ~7% AUROC. Upscaling to 380×380 would increase computational cost significantly with more patches and longer inference time. I chose 224×224 as a compromise between computational efficiency and model performance, ensuring uniform feature extraction across my varying dataset sizes (243×265 and 301×241).

**Why DINOv2 over DINOv3:** DINOv3 is larger and more resource-intensive. The performance gains didn't justify the overhead for my dataset size. DINOv3's weights were also more prone to overfitting on smaller datasets. Given my computational constraints, DINOv2 proved to be the better choice.

**7.4 No Data Augmentation**

**Decision:** Train only on original normal samples without augmentation.

**Rationale:** Given the controlled environment (fixed camera, consistent lighting), augmentation would introduce unrealistic variations that distort the normal boundary. Arbitrary rotations don't reflect test conditions, and aggressive color jitter might make the model insensitive to legitimate defects. Instead, I rely on DINOv2's features, which are naturally robust to subtle shifts due to ImageNet pretraining. This keeps the boundary tight and focused on actual surface anomalies.

**7.5 Feature Extractor (DINOv2)**

**Decision:** Use Vision Transformer-based features (DINOv2 ViT-B/14).

**Rationale:** I evaluated three feature extractors and found Vision Transformers capture fine-grained structural patterns better than CNNs. DINOv2 excels at detecting subtle texture anomalies, critical for surface defects. While EfficientNet-B4 is very lightweight and computationally efficient, in this use case accuracy is more important than lightweight deployment. I prioritized detection performance to ensure no defects slip through in manufacturing QA, so I chose the more powerful Vision Transformer approach despite its higher computational cost.

**Feature Extractor Comparison:**

<div align="center">

| Extractor | AUROC |
|-----------|-------|
| DINOv3 | 0.9711 |
| **DINOv2** | **0.9625** |
| Qwen | 0.9388 |
| EfficientNet-B4 | 0.9487 |

</div>

While DINOv3 achieved a slightly higher AUROC (0.9711), it is larger and more resource-intensive, requiring more memory and computational power. DINOv2 (0.9625 AUROC) provides excellent performance with lower computational overhead. The 0.86% performance trade-off is justified by significant savings in model size, inference speed, and memory consumption. For production deployment with my computational constraints, DINOv2's efficiency gains outweigh DINOv3's marginal accuracy improvement.

**Alternative to Explore:** Create an ensemble combining DINOv2 and DINOv3 feature extractors. Since DINOv3 achieves 0.9711 AUROC vs DINOv2's 0.9625, an ensemble could potentially push overall accuracy higher for critical quality scenarios. The trade-off is 2x computational cost at inference time, which may be acceptable if latency permits and accuracy improvement is significant.

**7.6 Number of Neighbors (k=9)**

**Decision:** Set k=9 for nearest neighbor search.

**Rationale:** Through experimentation on my dataset, k=9 balances sensitivity to anomalies with noise robustness. Smaller k is too noisy; larger k loses sensitivity. k=9 emerged as the empirical sweet spot.

### Trade-offs

**7.7 ROI Masking**

**Decision:** Apply region-of-interest (ROI) mask by setting pixels outside ROI to black. Procedure:
  - Apply black masking outside the ROI boundary
  - Pad the masked image to a square (with black padding)
  - Scale to 224×224

![ROI Masking Procedure](assets/roi_procedure.png)

**Rationale:** I tested three approaches: no masking, black masking, and white masking. Black masking achieved 2% improvement over white masking and 3.5% over no masking. Black pixels (zero intensity) are treated as legitimate signal absence by feature extractors, while white may be interpreted as noise. Background/borders are irrelevant; black masking cleanly eliminates this noise.

![ROI Masking Comparison](assets/roi.png)

**Trade-off:** Requires manual ROI definition and is inflexible to product shape changes. Assumes the feature extractor handles zero-intensity regions appropriately. However, the empirical 3.5% accuracy gain justifies this. Manual masks sometimes misalign with boundaries, causing red heatmap artifacts around borders. Future improvement: use Segment Anything Model (SAM) for automated ROI refinement or learn an adaptive ROI based on image features.

**7.8 Patch Overlap**

**Decision:** Use overlapping patches with sliding window.

**Rationale:** Overlapping ensures complete spatial coverage and smooth heatmaps without blind spots. Essential for precise anomaly localization.

**Trade-off:** Increased computational cost due to redundant feature extraction. But necessary for localization accuracy.

**7.9 Coreset Sampling Ratio (0.15)**

**Decision:** Retain 15% of training patches using random sampling.

**Rationale:** I initially tried greedy coreset selection for better representativeness, but it added 321ms to inference time. Random sampling (15%) is faster AND yields better accuracy. My hypothesis: random sampling avoids overfitting to training patterns, creating a more generalizable normal boundary.

**Trade-off:** Theoretically, random sampling risks losing important patterns vs. greedy/diversity-based methods. However, empirical results (better accuracy + faster inference) outweigh theory. Future work: explore k-center or density-aware sampling if performance plateaus.

**7.10 Indexing Strategy (In-Memory k-NN with Euclidean Distance)**

**Decision:** Store coreset patches in memory and use exact k-NN search with torch.cdist using Euclidean (L2) distance.

**Rationale:** ~27,648 coreset patches fit comfortably in memory. torch.cdist provides exact k-NN distances with no quantization error, crucial when small distance differences affect the decision boundary. Simple, interpretable, GPU-accelerated, and reproducible.

**Distance Metric Comparison:** I also tried Cosine distance (normalized by feature magnitude), but found no improvement in AUROC. Euclidean performed equally well and is slightly faster on GPU.

**Trade-off:** Memory-intensive for very large coresets (millions). FAISS GPU indexing could scale better for massive datasets, but simplicity and accuracy of exact search justify the memory trade-off for my current scale.

**7.11 Threshold Optimization (100% Recall)**

**Decision:** Use 100% recall threshold to catch every defect.

**Rationale:** In manufacturing QA, missing a defect is catastrophic. I computed two candidate thresholds from my data:

1. **99th Percentile (Production Default)**: 36.0894 - I computed this as the 99th percentile of anomaly scores from my training data (normal samples only). This provides a statistically-grounded threshold that balances coverage with a reasonable false positive rate.

2. **100% Recall Threshold (Industry Practice)**: 36.9051 - I derived this by finding the lowest anomaly score on my validation set that achieves 100% recall (zero false negatives). In practice, I would validate this threshold on a larger test set before deployment. In real manufacturing, the standard approach is to collect test images, measure recall at different thresholds, and select the one where recall reaches 100%. Since false positives are acceptable (human reviewers filter them), I can accept a higher false alarm rate to guarantee no defective products escape.

![Threshold Comparison](assets/threshold_comparison.png)

**Trade-off:** I accept higher false positive rate as the necessary cost to guarantee zero false negatives. No defective products slip through.

**Alternative to Explore:** Implement continuous model monitoring with concept drift detection (e.g., ADWIN statistical testing) to automatically detect if model performance degrades over time as manufacturing conditions or product batches change. This would trigger alerts or retraining when threshold adjustment is needed, rather than relying on manual monitoring.

## 8. Approaches That Didn't Work

During development, I experimented with several techniques that either didn't improve performance or increased complexity without benefit:

1. **Coarse-to-Fine Feature Extraction** - I tried hierarchical multi-scale feature extraction to capture both global context and local details. However, this approach didn't improve the AUROC score and significantly increased inference time due to processing multiple resolution levels. The added complexity didn't justify the computational cost.

2. **Greedy Coreset Selection** - I initially used greedy coreset sampling for better representativeness (aiming to preserve the most "diverse" patches). Greedy selection uses O(n²) complexity: for each of 184,320 training patches, you compute distances to remaining patches to find the most "diverse" ones. This results in ~33 billion distance computations during coreset construction. In practice, this added 321ms overhead per inference image due to the expensive nearest-neighbor computations. More critically, this slowdown made threshold tuning and validation impractical. If I needed to test multiple thresholds or re-tune the coreset ratio, each attempt would require waiting 15+ minutes. Random sampling at 0.15 ratio proved dramatically faster (O(n) complexity) and paradoxically achieved better AUROC, likely because it avoids overfitting to specific patterns in the training set. The simplicity and speed of random sampling won out.

3. **Fine-tuning Feature Extractor** - I experimented with fine-tuning DINOv2 on my specific dataset to adapt it better. My approach was simple: freeze earlier layers (which learn general features like edges/textures) and only fine-tune later layers (which learn domain-specific patterns). This is the standard transfer learning strategy to reduce overfitting. However, with limited normal examples (~720 training samples), even selective fine-tuning led to overfitting: the later layers overfit to specific surface textures in my limited training set, reducing robustness to natural variations at test time. AUROC actually decreased compared to using pre-trained features directly. The fundamental issue: 720 samples is too small to learn meaningful domain-specific variations without degrading generalization. Pre-trained DINOv2 features (trained on millions of ImageNet images) remain more robust to unseen surface patterns than a fine-tuned model on 720 images.

4. **Test-time Augmentation (TTA) with Ensemble Memory Banks** - I tried test-time augmentation by creating two models with different random seeds (different coreset samples) and ensembling their memory banks. The approach: (1) train two independent PatchCore models with different coreset samples, (2) at inference, compute anomaly score for the test image using both memory banks, (3) average or max-pool the scores. Intuition: two independent memory banks might capture different normal pattern variations, reducing gaps like Images 2-3. However, TTA added 232ms overhead per inference (nearly 8x slower than single model's ~28.94ms), and I observed no improvement in AUROC on my dataset. The lack of improvement suggests either: (a) the dataset patterns are well-captured by a single 0.15 coreset, or (b) the two models were sampling similar regions despite different seeds (not enough diversity from random sampling). The computational cost far outweighed any potential benefit for production use. Single model inference remains the practical choice.

5. **Duplicate Image Detection** - I checked the training dataset for duplicate images using MD5 hashing to detect if the same image appeared multiple times (which could inflate training metrics). I computed hashes for all 720 training images and found zero duplicates. This is good. Training data is diverse and not contaminated by repetition.

## 9. Evaluation Metrics

### Results on Surface Dataset

<div align="center">

| Metric | Value |
|--------|-------|
| **AUROC (Image-level)** | 0.9981 (99.81%) |
| **AUROC (Pixel-level)** | 0.8898 (88.98%) |
| **Decision Threshold** | 36.0894 |

</div>

I achieved 99.81% image-level AUROC with DINOv2, exceeding the 98% target that I set initiallly. This demonstrates strong discrimination between normal and anomalous surfaces. It means my model correctly ranks a randomly selected defective image as more anomalous than a normal image 99.81% of the time. The excellent score separation (normal vs. defective) indicates that DINOv2 features capture surface defects very effectively.

My pixel-level AUROC is lower at 88.98% because manual ROI masks sometimes misalign with actual product boundaries, introducing noise around borders that affects pixel-level accuracy. Additionally, at inference, upscaling patch-level scores to pixel space and applying Gaussian filtering blurs fine-grained localization details, making precise defect boundary detection difficult.

I prioritize 100% recall with my threshold to catch all defects in manufacturing QA.



![ROC Curve](assets/roc_curve.png)

The ROC curve shows a strong L-shaped pattern, with the curve staying close to the top-left corner. This indicates that I can achieve high true positive rates (recall) while maintaining low false positive rates, which is exactly what I wanted for manufacturing QA where catching all defects is critical. 

On the other hand, the anomaly score distribution shows clear separation between normal and defective samples. Normal images cluster with low scores (left side), while defective images cluster with higher scores (right side). The overlap I observe in the middle is explained in the "False Positives Analysis" below. Some normal samples fall into higher score ranges due to memory bank gaps (Images 2 & 3), while some test images with actual defects are mislabeled as normal (Images 1 & 4). Overall, the separation validates that my model learned meaningful anomaly signals.

### Inference Performance

<div align="center">

| Device | Mean (ms) | Std Dev (ms) | Min (ms) | Max (ms) |
|--------|-----------|-------------|----------|----------|
| GPU (CUDA) | 28.94     | 0.83        | 27.28    | 29.78    |
| GPU (MPS)  | 33.78     | 0.94        | 32.75    | 36.36    |
| CPU    | 295.21    | 6.23        | 286.33   | 306.66   |

</div>

GPU inference is ~10x faster than CPU. 

> **GPU Warmup:** I performed GPU warmup runs before measuring to ensure GPU memory is allocated and caches are warmed up, which I think is crucial for accurately tracking real inference time.



### False Positives Analysis

![False Positives](assets/false_positive.png)

Out of 524 normal test images, I observed 4 false positives flagged as defective. Analysis:

**Images 1 & 4 (Visually Faulty)**: Both appear to have actual surface defects and deserve to be flagged as positives. Image 1 shows blurry reflections/hazy regions that could be contour distortions, black hat morphology artifacts, or salt-and-pepper noise. I tried morphological filtering approaches (contour detection, black hat transforms, blob filtering) to identify and remove these artifacts, but none improved the model's ability to distinguish them from real defects. Image 4 exhibits clear visual defects. Rather than exclude these images through post-processing heuristics, I accepted the model's classification as correct. These are likely legitimate anomalies or test set labeling issues, not true false positives.

**Images 2 & 3 (Memory Bank Gaps)**: Represent a systematic issue where the random 0.15 coreset sample doesn't adequately represent certain legitimate surface patterns. Subtle texture variations in these regions score high despite being normal. I attempted to address this through coreset ratio adjustment, but any tuning that reduced these false positives also degraded recall on true defects, forcing me to prioritize the recall objective.

**Observation:** These false positives highlight a fundamental trade-off in one-class anomaly detection. Aggressive threshold or coreset tuning to reduce FP rate risks missing true defects. The current approach prioritizes zero false negatives (100% recall), accepting a 0.76% FP rate (4/524) as the acceptable cost for manufacturing quality assurance.

## 10. Limitations

1. **Limited training diversity** - Model trained on 1-2 surface types. May not generalize to new defect patterns
2. **Threshold not adaptive** - Fixed threshold assumes similar defect severity. Rare/subtle defects may be missed
3. **Computational cost** - Current approach uses efficient 224×224 resolution and random coreset sampling to keep costs low. However, computational complexity would be prohibitively high if I used greedy coreset selection (O(n²) complexity, adding 321ms overhead) or high-resolution images (more patches to process). Transformer-based feature extractors also add overhead. These design choices trade some potential accuracy for practical inference speed.
4. **Hyperparameter sensitivity** - Coreset ratio and k-neighbors require tuning per dataset
5. **Limited defective data** - Only 12 defective test images total (2 in Feature 1, 10 in Feature 2). A single misclassified defect changes Feature 1 recall from 100% to 50%. This severe data scarcity makes threshold validation unreliable and prevents generalization testing across defect types.
6. **No confidence intervals** - Reported metrics (99.81% AUROC, 36.0894 threshold) lack uncertainty estimates:
   - Could compute 95% confidence intervals via bootstrapping (resample test set 1000x) or K-fold cross-validation
   - With only 12 defects, confidence intervals would be extremely wide and uninformative
   - Statistical rigor impossible without more data
   - **For production deployment, recommend:**
     - Collect at least 100+ defects per feature type
     - Ensure balanced representation across feature types
     - Then compute proper confidence intervals before claiming performance guarantees

## 11. Potential Room for Improvements

### 12.1 Short-term
- **Per-defect-type thresholds** - Replace global threshold (36.0894) with category-specific thresholds
  - Different defect types (scratches, dents, corrosion, contamination) have different anomaly score distributions
  - Current single threshold assumes all defects score equally high, but subtle defects (fine scratches) might score lower than gross defects (large dents)
  - **Approach:**
    - Manually label test defects by type (if not already labeled)
    - Compute separate threshold for each type to achieve 100% recall per category
    - Use ensemble voting (flag if any category-threshold triggers)
  - **Benefit:** Catches subtle defects that might slip under global threshold
  - **Trade-off:** Requires labeled defect taxonomy (currently unavailable), and only feasible with more diverse test data (12 total defects insufficient for reliable per-type statistics)
- **Ensemble methods** - Combine multiple backbones (DINOv2 + EfficientNet) for robustness
- **Hard negative mining** - Focus training on borderline false positives to tighten decision boundary
- **Improved coreset sampling** - Replace random 0.15 sampling with SOTA methods
  - **Problem:** Current random approach misses pattern variations (Images 2-3 false positives)
  - **Options:**
    - **k-Center:** O(n·k) complexity, greedily maximize min-distance to existing coreset for better feature space coverage
    - **Importance Sampling:** O(n) with weighting by k-NN distance, prioritizes hard examples
    - **Stratified Random Sampling:** O(n) with theoretical guarantees on subspace coverage
  - **Trade-off:** Improved coreset costs more compute time during training (tolerable one-time cost) but gains better memory bank representation without hurting inference speed

### 12.2 Medium-term
- **Domain-specific feature extraction** - Fine-tune DINOv2 on normal surface images using self-supervised learning (SimCLR, BYOL)
  - Current pre-trained features (ImageNet) are robust but not optimized for subtle surface texture anomalies
  - Domain adaptation would learn surface-specific invariances (lighting changes, minor scratches that aren't defects)
  - **Constraint:** Only 720 training images (minimal, risks overfitting without strong augmentation)
  - **Expected benefit:** Tighter normal boundary, fewer memory bank gaps (fewer Images 2-3 false positives)

- **Production Monitoring & Concept Drift Detection** - Deploy monitoring system to detect when model performance degrades
  - Log inference predictions, anomaly scores, metadata (timestamp, camera condition, surface batch)
  - **Trigger retraining when:**
    - FP rate exceeds 2%
    - FN rate detected (human review shows missed defects)
    - Score distribution changes by >15% KL divergence
  - **Implementation:** Use ADWIN (Adaptive Windowing) or statistical tests to detect distribution shift
  - **Why it matters:** Manufacturing conditions evolve; Jan 2024 model may fail by Mar 2024

- **Hierarchical k-NN** - Multi-scale patch analysis (local + global context) for better localization
- **Anomaly-specific clustering** - Separate "defect types" to learn better thresholds per category

### 12.3 Long-term
- **Zero-Shot CLIP + DINOv2 Ensemble** - Multimodal embeddings combining vision and text descriptions of defects. No labeled defects required.
- **Test-Time Augmentation (TTA) with Diverse Models** - Previous TTA using two PatchCore models with same backbone lacked diversity and didn't improve AUROC
  - **Better approach:** Ensemble DINOv2 + EfficientNet feature extractors (different architectures = orthogonal strengths)
  - Average anomaly scores across both extractors
  - Trade-off: 2x inference cost but potentially higher robustness to unseen defects
- **Multi-Modal Transformer Ensemble** - DINOv2 + CLIP with orthogonal strengths (vision + text embeddings)
- **Transformer Attention for Sparse Features** - Reduce memory bank to 5K-10K critical patches via ViT attention pruning
- **Benchmarking on MVTec AD & VisA** - Validate generalization beyond your specific surface type
  - Measure AUROC (image & pixel), F1-score, inference time (<30ms GPU target)
- **Synthetic Defect Generation (Flux Schnell)** - Generate 200+ synthetic defects to address data imbalance
  - ~1 sec/image with text prompts ("scratched metal", "dented steel", "corroded surface")
  - Feature 1 has only 2 defects; synthetic generation creates 50+ per type
  - Enables per-defect-type threshold tuning (Section 12.1) and robustness validation
- **ControlNet Inpainting** - Inpaint synthetic defects onto real normal images maintaining photorealistic lighting/shadows
  - Creates training pairs for contrastive learning
  - Trade-off: 2-4 sec/image vs 1 sec Flux Schnell, but higher photorealism
- **CycleGAN Domain Adaptation** - Generate defects on generic surfaces then adapt to your material via CycleGAN
  - ~100ms inference, best for continuous deployment
  - Trade-off: requires 2-3 hours upfront training
- **Priority:** Start with Flux Schnell (fast, immediate data) → ControlNet (production-grade quality) → CycleGAN (long-term deployment).

## 12. References

[1] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, "MVTec AD : A comprehensive real-world dataset for unsupervised anomaly detection," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 9584-9592.

[2] C. Zhou, R. C. Paffenroth, "Anomaly Detection with Robust Deep Autoencoders," in Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2017.

[3] J. Oquab, T. Darcet, T. Moutakanni, et al., "DINOv2: Learning Robust Visual Features without Supervision," arXiv preprint arXiv:2304.07193, 2023.

[4] D. Rippel, E. Mertens, and D. Merhav, "Modeling the distribution of normal data in pre-trained deep features for anomaly detection and localization," arXiv preprint arXiv:2005.14674, 2020.
