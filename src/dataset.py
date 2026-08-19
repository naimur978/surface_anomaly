"""
MVTec dataset loader with preprocessing.
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class MVTecDataset(Dataset):
    """MVTec dataset loader with preprocessing and augmentation."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, root, category, split="train", crop_size=224):
        root = Path(root).resolve()  # Convert to absolute path
        self.root = root / category / split
        self.mask_root = root / category / "ground_truth"
        self.split = split
        self.transform = self._build_transform(crop_size, normalize=True)
        self.mask_transform = self._build_transform(crop_size, normalize=False)
        self.samples = self._build_samples()
        self._print_summary(category)

    def _pad_to_square(self, img, pad_color=0):
        """Pad image to square."""
        w, h = img.size
        size = max(w, h)
        new_img = Image.new(img.mode, (size, size), pad_color)
        new_img.paste(img, ((size - w) // 2, (size - h) // 2))
        return new_img

    def _build_transform(self, crop_size, normalize):
        """Build image transformation pipeline."""
        ops = [
            transforms.Lambda(lambda img: self._pad_to_square(img, 0)),
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
        ]
        if normalize:
            ops.append(transforms.Normalize(self.IMAGENET_MEAN, self.IMAGENET_STD))
        return transforms.Compose(ops)

    def _resolve_mask_path(self, class_name, img_path):
        """Find mask file for defective image."""
        mask_path = self.mask_root / class_name / (img_path.stem + "_mask.png")
        return mask_path if mask_path.exists() else None

    def _build_samples(self):
        """Build list of (image_path, mask_path, label) tuples."""
        samples = []
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            label = 0 if class_dir.name == "good" else 1
            for img_path in sorted(class_dir.glob("*.png")):
                mask_path = self._resolve_mask_path(class_dir.name, img_path) if label == 1 else None
                samples.append((img_path, mask_path, label))
        return samples

    def _print_summary(self, category):
        """Print dataset summary."""
        n_normal = sum(1 for _, _, l in self.samples if l == 0)
        n_defect = sum(1 for _, _, l in self.samples if l == 1)
        print(f"Dataset '{self.split}' [{category}]: {n_normal} normal | {n_defect} defective")

    def __len__(self):
        return len(self.samples)

    def _load_mask(self, mask_path, img_t):
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

    def __getitem__(self, idx):
        img_path, mask_path, label = self.samples[idx]
        img_t = self.transform(Image.open(img_path).convert("RGB"))
        mask_t = self._load_mask(mask_path, img_t)
        return img_t, mask_t, label, str(img_path.absolute())
