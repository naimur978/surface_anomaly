"""
MVTec dataset loader with preprocessing and ROI masking.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class MVTecDataset(Dataset):
    """MVTec dataset loader with preprocessing, ROI masking, and augmentation."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, root: str | Path, category: str, split: str = "train", crop_size: int = 224, apply_roi_mask: bool = False) -> None:
        root = Path(root).resolve()  # Convert to absolute path
        self.root = root / category / split
        self.mask_root = root / category / "ground_truth"
        self.category = category
        self.split = split
        self.apply_roi_mask = apply_roi_mask

        # Load ROI masks if enabled
        self.roi_masks = {}
        if self.apply_roi_mask:
            self.roi_masks = self._load_roi_masks(root)

        self.transform = self._build_transform(crop_size, normalize=True)
        self.mask_transform = self._build_transform(crop_size, normalize=False)
        self.samples = self._build_samples()
        self._print_summary(category)

    def _load_roi_masks(self, root: Path) -> Dict[str, np.ndarray]:
        """Load ROI masks from parent data folder (mask.png for each feature)."""
        roi_masks: Dict[str, np.ndarray] = {}
        parent = root.parent.parent  # Go up to find data root with features

        # Look for mask.png files in Feature 1, Feature 2, etc.
        for feature_dir in parent.glob("Feature *"):
            if feature_dir.is_dir():
                mask_path = feature_dir / "mask.png"
                if mask_path.exists():
                    mask_img = Image.open(mask_path).convert("L")
                    mask_array = np.array(mask_img) > 127  # Binary mask (True = ROI)
                    feature_name = feature_dir.name.replace(" ", "").lower()  # "Feature 1" -> "feature1"
                    roi_masks[feature_name] = mask_array

        return roi_masks

    def _apply_roi_mask(self, img: Image.Image) -> Image.Image:
        """Apply ROI mask to image (black out non-ROI regions)."""
        if not self.apply_roi_mask or not self.roi_masks:
            return img

        # Determine which feature this image belongs to (feature1 or feature2)
        img_array = np.array(img)

        # Try to find matching ROI mask
        for feature_prefix, mask in self.roi_masks.items():
            # Resize mask to match image size
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img = mask_img.resize(img.size, Image.Resampling.NEAREST)
            mask_array = np.array(mask_img) > 127

            # Apply mask: set non-ROI pixels to 0 (black)
            img_array[~mask_array] = 0

            return Image.fromarray(img_array)

        return img

    def _pad_to_square(self, img: Image.Image, pad_color: int = 0) -> Image.Image:
        """Pad image to square."""
        w, h = img.size
        size = max(w, h)
        new_img = Image.new(img.mode, (size, size), pad_color)
        new_img.paste(img, ((size - w) // 2, (size - h) // 2))
        return new_img

    def _build_transform(self, crop_size: int, normalize: bool) -> transforms.Compose:
        """Build image transformation pipeline."""
        ops = [
            transforms.Lambda(lambda img: self._apply_roi_mask(img) if self.apply_roi_mask else img),
            transforms.Lambda(lambda img: self._pad_to_square(img, 0)),
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
        ]
        if normalize:
            ops.append(transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD))
        return transforms.Compose(ops)

    def _resolve_mask_path(self, class_name: str, img_path: Path) -> Optional[Path]:
        """Find mask file for defective image."""
        mask_path = self.mask_root / class_name / (img_path.stem + "_mask.png")
        return mask_path if mask_path.exists() else None

    def _build_samples(self) -> List[Tuple[Path, Optional[Path], int]]:
        """Build list of (image_path, mask_path, label) tuples."""
        samples: List[Tuple[Path, Optional[Path], int]] = []
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = 0 if class_dir.name == "good" else 1
            for img_path in sorted(class_dir.glob("*.png")):
                mask_path = self._resolve_mask_path(class_dir.name, img_path) if label == 1 else None
                samples.append((img_path, mask_path, label))
        return samples

    def _print_summary(self, category: str) -> None:
        """Print dataset summary."""
        n_normal = sum(1 for _, _, l in self.samples if l == 0)
        n_defect = sum(1 for _, _, l in self.samples if l == 1)
        roi_status = "✓ ROI masking enabled" if self.apply_roi_mask else "✗ No ROI masking"
        print(f"Dataset '{self.split}' [{category}]: {n_normal} normal | {n_defect} defective | {roi_status}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_mask(self, mask_path: Optional[Path], img_t: torch.Tensor) -> torch.Tensor:
        """Load ground truth mask."""
        if mask_path is None:
            return torch.zeros(1, img_t.shape[1], img_t.shape[2])
        try:
            mask = Image.open(mask_path).convert("L")
            mask_t = self.mask_transform(mask)
            return (mask_t > 0.5).float()
        except (FileNotFoundError, OSError):
            # Mask not found, return empty mask
            return torch.zeros(1, img_t.shape[1], img_t.shape[2])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        img_path, mask_path, label = self.samples[idx]
        img_t = self.transform(Image.open(img_path).convert("RGB"))
        mask_t = self._load_mask(mask_path, img_t)
        return img_t, mask_t, label, str(img_path.absolute())
