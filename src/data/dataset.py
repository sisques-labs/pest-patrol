"""Dataset classes for pest classification."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class PestDataset(Dataset):
    """Dataset class for pest classification images.

    Expected directory structure:
        data/
        ├── raw/
        │   ├── class1/
        │   │   ├── img1.jpg
        │   │   └── img2.jpg
        │   ├── class2/
        │   │   ├── img1.jpg
        │   │   └── img2.jpg
    """

    def __init__(
        self,
        images: List[str],
        labels: List[int],
        class_to_idx: Dict[str, int],
        transform: Optional[Callable] = None,
    ):
        """Initialize PestDataset.

        Args:
            images: List of image file paths
            labels: List of integer labels corresponding to images
            class_to_idx: Mapping from class names to integer indices
            transform: Optional transform to apply to images
        """
        self.images = images
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item at index.

        Args:
            idx: Index of the item

        Returns:
            Tuple of (image tensor, label)
        """
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading image {img_path}: {e}")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_name(self, idx: int) -> str:
        """Get class name from index.

        Args:
            idx: Class index

        Returns:
            Class name
        """
        return self.idx_to_class.get(idx, "unknown")


def _discover_classes(data_dir: Path) -> Dict[str, int]:
    """Discover class names from directory structure.

    Args:
        data_dir: Root directory containing class subdirectories

    Returns:
        Dictionary mapping class names to indices
    """
    class_dirs = [
        d
        for d in data_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    class_dirs.sort()  # Ensure consistent ordering
    return {class_dir.name: idx for idx, class_dir in enumerate(class_dirs)}


def _collect_images_and_labels(
    data_dir: Path, class_to_idx: Dict[str, int]
) -> Tuple[List[str], List[int]]:
    """Collect all image paths and corresponding labels.

    Args:
        data_dir: Root directory containing class subdirectories
        class_to_idx: Mapping from class names to indices

    Returns:
        Tuple of (image paths, labels)
    """
    images = []
    labels = []

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    for class_name, class_idx in class_to_idx.items():
        class_dir = data_dir / class_name
        if not class_dir.exists():
            continue

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in valid_extensions:
                images.append(str(img_path))
                labels.append(class_idx)

    return images, labels


def _split_dataset(
    images: List[str],
    labels: List[int],
    train_split: float,
    val_split: float,
    test_split: float,
    random_seed: int = 42,
) -> Tuple[
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
    Tuple[List[str], List[int]],
]:
    """Split dataset into train, validation, and test sets.

    Args:
        images: List of image paths
        labels: List of labels
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train, val, test) splits, each containing (images, labels)
    """
    # Normalize splits
    total = train_split + val_split + test_split
    train_split /= total
    val_split /= total
    test_split /= total

    # First split: train vs (val + test)
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        images,
        labels,
        test_size=(1 - train_split),
        random_state=random_seed,
        stratify=labels,
    )

    # Second split: val vs test
    val_size = val_split / (val_split + test_split)
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images,
        temp_labels,
        test_size=(1 - val_size),
        random_state=random_seed,
        stratify=temp_labels,
    )

    return (
        (train_images, train_labels),
        (val_images, val_labels),
        (test_images, test_labels),
    )


def get_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    test_transform: Optional[Callable] = None,
    num_workers: int = 4,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Create data loaders for train, validation, and test sets.

    Args:
        data_dir: Path to directory containing class subdirectories
        batch_size: Batch size for data loaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        train_transform: Transform to apply to training images
        val_transform: Transform to apply to validation images
        test_transform: Transform to apply to test images
        num_workers: Number of worker processes for data loading
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_to_idx)
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")

    # Discover classes
    class_to_idx = _discover_classes(data_dir)
    if not class_to_idx:
        raise ValueError(
            f"No class directories found in {data_dir}. "
            "Expected structure: data_dir/class_name/images.jpg"
        )

    # Collect images and labels
    images, labels = _collect_images_and_labels(data_dir, class_to_idx)
    if not images:
        raise ValueError(f"No images found in {data_dir}")

    # Split dataset
    train_data, val_data, test_data = _split_dataset(
        images, labels, train_split, val_split, test_split, random_seed
    )

    # Create datasets
    train_dataset = PestDataset(
        train_data[0], train_data[1], class_to_idx, transform=train_transform
    )
    val_dataset = PestDataset(
        val_data[0], val_data[1], class_to_idx, transform=val_transform
    )
    test_dataset = PestDataset(
        test_data[0], test_data[1], class_to_idx, transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_to_idx
