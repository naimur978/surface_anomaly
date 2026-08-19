"""
Data preparation and validation utilities for PatchCore anomaly detection.

This module provides functions for:
- Data structure validation
- Image format checking
- Dataset statistics
- Data splitting
"""

import logging
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import hashlib

logger = logging.getLogger(__name__)


def validate_data_structure(root_dir, category):
    """
    Validate that data follows the expected directory structure.

    Args:
        root_dir: Root data directory
        category: Product category

    Returns:
        Dictionary with validation results
    """
    cat_path = Path(root_dir) / category
    issues = []
    warnings = []

    # Check required directories
    required_dirs = [
        cat_path / "train" / "good",
        cat_path / "test" / "good",
        cat_path / "ground_truth"
    ]

    for dir_path in required_dirs:
        if not dir_path.exists():
            issues.append(f"Missing directory: {dir_path}")

    # Check for at least one defect category
    test_dir = cat_path / "test"
    if test_dir.exists():
        defect_dirs = [d for d in test_dir.iterdir() if d.is_dir() and d.name != "good"]
        if not defect_dirs:
            warnings.append("No defect images found in test directory")

    # Check for ground truth masks
    gt_dir = cat_path / "ground_truth"
    if gt_dir.exists():
        mask_count = len(list(gt_dir.rglob("*_mask.png")))
        if mask_count == 0:
            warnings.append("No ground truth masks found")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings
    }


def check_image_properties(root_dir, category, extensions=['.png', '.jpg', '.jpeg']):
    """
    Check properties of all images in dataset.

    Args:
        root_dir: Root data directory
        category: Product category
        extensions: List of valid image extensions

    Returns:
        Dictionary with image statistics
    """
    cat_path = Path(root_dir) / category
    sizes = []
    modes = []
    ranges = []
    corrupted = []
    blank = []

    files = list(cat_path.rglob("*.png")) + list(cat_path.rglob("*.jpg"))

    logger.info(f"Checking {len(files)} images...")

    for f in files:
        if f.name.endswith('_mask.png'):
            continue

        try:
            img = Image.open(f)
            arr = np.array(img)
            sizes.append(img.size)
            modes.append(img.mode)
            ranges.append((arr.min(), arr.max()))

            if arr.max() == arr.min():
                blank.append(f)
        except Exception as e:
            corrupted.append((f, str(e)))

    stats = {
        'total_images': len(files),
        'unique_sizes': set(sizes),
        'unique_modes': set(modes),
        'pixel_range': (min(r[0] for r in ranges) if ranges else None,
                       max(r[1] for r in ranges) if ranges else None),
        'corrupted': corrupted,
        'blank': blank,
    }

    return stats


def print_dataset_stats(root_dir, category):
    """Print comprehensive dataset statistics."""
    cat_path = Path(root_dir) / category

    print("\n" + "="*60)
    print(f"DATASET STATISTICS - {category.upper()}")
    print("="*60)

    # Count images
    train_good = list((cat_path / "train" / "good").glob("*.png"))
    test_good = list((cat_path / "test" / "good").glob("*.png"))

    print(f"\nTraining Set:")
    print(f"  Normal images: {len(train_good)}")

    print(f"\nTest Set:")
    print(f"  Normal images: {len(test_good)}")

    test_dir = cat_path / "test"
    if test_dir.exists():
        for defect_dir in sorted(test_dir.iterdir()):
            if defect_dir.is_dir() and defect_dir.name != "good":
                count = len(list(defect_dir.glob("*.png")))
                print(f"  Defective images ({defect_dir.name}): {count}")

    # Check image properties
    stats = check_image_properties(root_dir, category)

    print(f"\nImage Properties:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Unique sizes: {stats['unique_sizes']}")
    print(f"  Unique modes: {stats['unique_modes']}")
    print(f"  Pixel range: {stats['pixel_range'][0]} - {stats['pixel_range'][1]}")

    if stats['corrupted']:
        print(f"\n  ⚠ Corrupted images: {len(stats['corrupted'])}")
        for path, error in stats['corrupted'][:3]:
            print(f"    - {path.name}: {error}")

    if stats['blank']:
        print(f"\n  ⚠ Blank images: {len(stats['blank'])}")
        for path in stats['blank'][:3]:
            print(f"    - {path.name}")

    print("="*60 + "\n")


def find_duplicates(folder, hash_type='md5'):
    """
    Find duplicate images in a folder using file hashing.

    Args:
        folder: Folder to check
        hash_type: Hash algorithm to use

    Returns:
        Dictionary of duplicate groups
    """
    hashes = defaultdict(list)

    for f in Path(folder).glob("*.png"):
        with open(f, "rb") as file:
            h = hashlib.md5(file.read()).hexdigest()
            hashes[h].append(f)

    dupes = {h: files for h, files in hashes.items() if len(files) > 1}

    if dupes:
        logger.warning(f"Found {len(dupes)} duplicate groups in {folder}")
        for h, files in dupes.items():
            logger.warning(f"  - {[f.name for f in files]}")

    return dupes


def split_dataset(root_dir, category, train_ratio=0.8, seed=42):
    """
    Split good images into train/val sets.

    Note: This creates a backup and modifies directory structure!

    Args:
        root_dir: Root data directory
        category: Product category
        train_ratio: Ratio for train/val split
        seed: Random seed

    Returns:
        Paths to train and val directories
    """
    import random
    import shutil

    random.seed(seed)

    cat_path = Path(root_dir) / category
    train_good = cat_path / "train" / "good"
    val_good = cat_path / "val" / "good"

    # Create val directory
    val_good.mkdir(parents=True, exist_ok=True)

    # Get all training images
    images = list(train_good.glob("*.png"))
    n_train = int(len(images) * train_ratio)

    # Randomly select validation images
    val_images = random.sample(images, len(images) - n_train)

    # Move to val directory
    for img in val_images:
        shutil.move(str(img), str(val_good / img.name))

    logger.info(f"Split: {n_train} train, {len(val_images)} val")

    return train_good, val_good


def verify_dataset_integrity(root_dir, category):
    """
    Perform comprehensive dataset verification.

    Args:
        root_dir: Root data directory
        category: Product category

    Returns:
        Boolean indicating if dataset is valid
    """
    # Validate structure
    struct_result = validate_data_structure(root_dir, category)

    if struct_result['issues']:
        logger.error("Dataset structure validation failed:")
        for issue in struct_result['issues']:
            logger.error(f"  - {issue}")
        return False

    if struct_result['warnings']:
        logger.warning("Dataset warnings:")
        for warning in struct_result['warnings']:
            logger.warning(f"  - {warning}")

    # Check images
    stats = check_image_properties(root_dir, category)

    if stats['corrupted']:
        logger.error(f"Found {len(stats['corrupted'])} corrupted images")
        return False

    if stats['blank']:
        logger.warning(f"Found {len(stats['blank'])} blank images")

    # Check duplicates
    train_good = Path(root_dir) / category / "train" / "good"
    duplicates = find_duplicates(train_good)

    if duplicates:
        logger.warning(f"Found {len(duplicates)} duplicate groups")

    logger.info("Dataset integrity verification completed")
    return True


if __name__ == "__main__":
    import sys

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python utils_data.py <root_dir> <category>")
        print("Example: python utils_data.py ./data surface")
        sys.exit(1)

    root_dir = sys.argv[1]
    category = sys.argv[2]

    # Run verification
    print_dataset_stats(root_dir, category)
    is_valid = verify_dataset_integrity(root_dir, category)

    if is_valid:
        print("✓ Dataset validation passed!")
    else:
        print("✗ Dataset validation failed!")
        sys.exit(1)
