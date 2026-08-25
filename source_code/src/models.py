"""
PatchCore model components (Feature Extractor and PatchCore scoring).
"""

from typing import Tuple, Optional
import logging
import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import os

# Patch torch.hub to skip GitHub repo validation to avoid rate limit issues
def _patched_validate(*args, **kwargs):
    pass

torch.hub._validate_not_a_forked_repo = _patched_validate


class FeatureExtractor(nn.Module):
    """Feature extractor with support for DINOv2 and EfficientNet backbones."""

    def __init__(self, device: torch.device, model_name: str = 'dinov2_vitb14',
                 logger: Optional[logging.Logger] = None) -> None:
        super().__init__()
        self.device = device
        self.logger = logger
        self.model_name = model_name

        if model_name.startswith('dinov2_'):
            self._init_dinov2(device, model_name, logger)
        elif model_name == 'efficientnet_b4':
            self._init_efficientnet(device, logger)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _init_dinov2(self, device: torch.device, model_name: str,
                     logger: Optional[logging.Logger]) -> None:
        """Initialize DINOv2 feature extractor."""
        os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, trust_repo=True).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.feature_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size
        self.backbone_type = 'dinov2'
        if logger:
            logger.info(f"{model_name} loaded successfully")

    def _init_efficientnet(self, device: torch.device,
                          logger: Optional[logging.Logger]) -> None:
        """Initialize EfficientNet B4 feature extractor."""
        from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
        from torchvision.models.feature_extraction import create_feature_extractor

        backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

        # Extract intermediate layers (layer2 and layer3)
        self.model = create_feature_extractor(
            backbone,
            return_nodes={
                "features.4": "layer2",   # 40x40 spatial resolution
                "features.6": "layer3",   # 20x20 spatial resolution
            },
        )
        self.model = self.model.to(device).eval()

        for p in self.model.parameters():
            p.requires_grad = False

        # For 224x224 input: layer2 is 28x28, layer3 is 14x14
        # After concatenation: 14x14 spatial, 688+1536=2224 channels
        self.feature_dim = 2224  # layer2 + layer3 concatenated
        self.patch_size = 16  # 224 / 14 = 16
        self.backbone_type = 'efficientnet'
        self.neighbourhood_size = 3

        if logger:
            logger.info("EfficientNet B4 loaded successfully")

    @torch.no_grad()
    def extract(self, imgs: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Extract patch embeddings from images.

        Args:
            imgs: Input images tensor (B, C, H, W)

        Returns:
            Tuple of (patch_embeddings, spatial_grid_shape)
        """
        imgs = imgs.to(self.device)
        B, C, H, W = imgs.shape
        gh, gw = H // self.patch_size, W // self.patch_size

        if self.backbone_type == 'dinov2':
            tokens = self.model.forward_features(imgs)['x_norm_patchtokens']
        else:  # efficientnet
            # Forward through EfficientNet to get layer2 and layer3 features
            feats = self.model(imgs)
            f2 = feats["layer2"]  # (B, 688, 28, 28)
            f3 = feats["layer3"]  # (B, 1536, 14, 14)

            # Apply neighbourhood pooling to layer2
            if self.neighbourhood_size > 1:
                f2 = torch.nn.functional.avg_pool2d(
                    f2, kernel_size=self.neighbourhood_size, stride=1,
                    padding=self.neighbourhood_size // 2
                )

            # Upsample layer3 to match layer2 spatial dimensions
            f3_up = torch.nn.functional.interpolate(
                f3, size=f2.shape[-2:], mode="bilinear", align_corners=False
            )

            # Concatenate layer2 and upsampled layer3
            combined = torch.cat([f2, f3_up], dim=1)  # (B, 2224, 28, 28)

            # Reshape to patch format: (B, H*W, C)
            B, C, h, w = combined.shape
            tokens = combined.permute(0, 2, 3, 1).reshape(B, h * w, C)

        return tokens.cpu(), (gh, gw)


class PatchCore:
    """PatchCore anomaly detection with memory bank and k-NN scoring."""

    def __init__(self, coreset_ratio: float = 0.15, n_neighbors: int = 9,
                 device: str = 'cuda', logger: Optional[logging.Logger] = None,
                 heatmap_smoothing: float = 4.0) -> None:
        self.coreset_ratio = coreset_ratio
        self.n_neighbors = n_neighbors
        self.device = device
        self.logger = logger
        self.heatmap_smoothing = heatmap_smoothing
        self.memory_bank: Optional[torch.Tensor] = None
        self.hw_shape: Optional[Tuple[int, int]] = None

    def fit(self, all_patches: torch.Tensor, hw_shape: Tuple[int, int], seed: int = 42) -> None:
        """Fit model with coreset selection."""
        self.hw_shape = hw_shape
        feats = all_patches if isinstance(all_patches, torch.Tensor) else torch.from_numpy(all_patches)

        if self.logger:
            self.logger.info(f"Memory bank: {len(feats):,} vectors, dim {feats.shape[1]}")

        n_keep = max(1, int(len(feats) * self.coreset_ratio))

        if self.logger:
            self.logger.info(f"Coreset selection: {self.coreset_ratio*100:.0f}% ratio, keeping {n_keep:,} vectors")

        selected_idx = self._greedy_coreset(feats, n_keep, seed)
        self.memory_bank = feats[selected_idx].to(self.device)

        if self.logger:
            self.logger.info(f"Final memory bank: {len(self.memory_bank):,} vectors")

    def _greedy_coreset(self, feats: torch.Tensor, n_keep: int, seed: int = 42) -> np.ndarray:
        """Random sampling for coreset selection."""
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(feats), size=n_keep, replace=False)
        return selected

    def _to_tensor(self, patch_features: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert to tensor if needed."""
        if isinstance(patch_features, np.ndarray):
            return torch.from_numpy(patch_features)
        return patch_features

    def _compute_patch_scores(self, patch_features: np.ndarray | torch.Tensor) -> np.ndarray:
        """Compute k-NN distances for patches."""
        x = self._to_tensor(patch_features).to(self.device)
        dists = torch.cdist(x, self.memory_bank)
        knn_dists, _ = torch.topk(dists, self.n_neighbors, dim=1, largest=False)
        return knn_dists[:, 0].cpu().numpy()

    def score_image(self, patch_features: np.ndarray | torch.Tensor) -> float:
        """Score entire image (max patch distance)."""
        scores = self._compute_patch_scores(patch_features)
        return float(scores.max())

    def score_map(self, patch_features: np.ndarray | torch.Tensor) -> np.ndarray:
        """Generate pixel-level anomaly map."""
        H, W = self.hw_shape
        scores = self._compute_patch_scores(patch_features).reshape(H, W)
        # Resize using scipy zoom
        zoom_factors = (224 / H, 224 / W)
        scores_up = ndimage.zoom(scores, zoom_factors, order=1)
        return gaussian_filter(scores_up, sigma=self.heatmap_smoothing)
