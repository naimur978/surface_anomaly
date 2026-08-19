"""
PatchCore model components (Feature Extractor and PatchCore scoring).
"""

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
    """DINOv2 feature extractor for patch embeddings."""

    def __init__(self, device, model_name='dinov2_vitb14', logger=None):
        super().__init__()
        self.device = device
        self.logger = logger
        os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, trust_repo=True).to(device).eval()

        for p in self.model.parameters():
            p.requires_grad = False

        self.feature_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size

        if logger:
            logger.info(f"{model_name} loaded successfully")

    @torch.no_grad()
    def extract(self, imgs):
        """Extract patch embeddings from images."""
        imgs = imgs.to(self.device)
        B, C, H, W = imgs.shape
        gh, gw = H // self.patch_size, W // self.patch_size
        tokens = self.model.forward_features(imgs)['x_norm_patchtokens']
        return tokens.cpu(), (gh, gw)


class PatchCore:
    """PatchCore anomaly detection with memory bank and k-NN scoring."""

    def __init__(self, coreset_ratio=0.15, n_neighbors=9, device='cuda', logger=None):
        self.coreset_ratio = coreset_ratio
        self.n_neighbors = n_neighbors
        self.device = device
        self.logger = logger
        self.memory_bank = None
        self.hw_shape = None

    def fit(self, all_patches, hw_shape, seed=42):
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

    def _greedy_coreset(self, feats, n_keep, seed=42):
        """Random sampling for coreset selection."""
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(feats), size=n_keep, replace=False)
        return selected

    def _to_tensor(self, patch_features):
        """Convert to tensor if needed."""
        if isinstance(patch_features, np.ndarray):
            return torch.from_numpy(patch_features)
        return patch_features

    def _compute_patch_scores(self, patch_features):
        """Compute k-NN distances for patches."""
        x = self._to_tensor(patch_features).to(self.device)
        dists = torch.cdist(x, self.memory_bank)
        knn_dists, _ = torch.topk(dists, self.n_neighbors, dim=1, largest=False)
        return knn_dists[:, 0].cpu().numpy()

    def score_image(self, patch_features):
        """Score entire image (max patch distance)."""
        scores = self._compute_patch_scores(patch_features)
        return float(scores.max())

    def score_map(self, patch_features):
        """Generate pixel-level anomaly map."""
        H, W = self.hw_shape
        scores = self._compute_patch_scores(patch_features).reshape(H, W)
        # Resize using scipy zoom
        zoom_factors = (224 / H, 224 / W)
        scores_up = ndimage.zoom(scores, zoom_factors, order=1)
        return gaussian_filter(scores_up, sigma=4)
