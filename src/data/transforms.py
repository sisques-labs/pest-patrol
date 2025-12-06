"""Image transforms and data augmentation for pest classification."""

from typing import Dict, Any, Optional
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F


class RandomRotation:
    """Random rotation transform."""

    def __init__(self, degrees: int = 15):
        """Initialize random rotation.

        Args:
            degrees: Maximum rotation angle in degrees
        """
        self.degrees = degrees

    def __call__(self, img):
        """Apply rotation."""
        angle = torch.randint(-self.degrees, self.degrees + 1, (1,)).item()
        return F.rotate(img, angle)


class RandomBrightness:
    """Random brightness adjustment."""

    def __init__(self, brightness_range: tuple = (0.8, 1.2)):
        """Initialize random brightness.

        Args:
            brightness_range: Tuple of (min, max) brightness factors
        """
        self.brightness_range = brightness_range

    def __call__(self, img):
        """Apply brightness adjustment."""
        brightness_factor = torch.empty(1).uniform_(
            self.brightness_range[0], self.brightness_range[1]
        ).item()
        return F.adjust_brightness(img, brightness_factor)


class RandomContrast:
    """Random contrast adjustment."""

    def __init__(self, contrast_range: tuple = (0.8, 1.2)):
        """Initialize random contrast.

        Args:
            contrast_range: Tuple of (min, max) contrast factors
        """
        self.contrast_range = contrast_range

    def __call__(self, img):
        """Apply contrast adjustment."""
        contrast_factor = torch.empty(1).uniform_(
            self.contrast_range[0], self.contrast_range[1]
        ).item()
        return F.adjust_contrast(img, contrast_factor)


class RandomSaturation:
    """Random saturation adjustment."""

    def __init__(self, saturation_range: tuple = (0.8, 1.2)):
        """Initialize random saturation.

        Args:
            saturation_range: Tuple of (min, max) saturation factors
        """
        self.saturation_range = saturation_range

    def __call__(self, img):
        """Apply saturation adjustment."""
        saturation_factor = torch.empty(1).uniform_(
            self.saturation_range[0], self.saturation_range[1]
        ).item()
        return F.adjust_saturation(img, saturation_factor)


def get_transforms(
    image_size: tuple = (224, 224),
    augmentation_config: Optional[Dict[str, Any]] = None,
    is_training: bool = True,
) -> transforms.Compose:
    """Get image transforms for training or validation.

    Args:
        image_size: Tuple of (height, width) for image resizing
        augmentation_config: Configuration dictionary for data augmentation
        is_training: Whether these are training transforms (includes augmentation)

    Returns:
        Composed transform pipeline
    """
    transform_list = []

    if is_training and augmentation_config and augmentation_config.get("enabled", True):
        # Training transforms with augmentation
        if augmentation_config.get("random_crop", True):
            transform_list.append(
                transforms.RandomResizedCrop(
                    size=image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)
                )
            )
        else:
            transform_list.append(transforms.Resize(image_size))

        if augmentation_config.get("horizontal_flip", True):
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        if augmentation_config.get("rotation_range", 0) > 0:
            transform_list.append(
                RandomRotation(degrees=augmentation_config["rotation_range"])
            )

        if augmentation_config.get("brightness_range"):
            transform_list.append(
                RandomBrightness(
                    brightness_range=tuple(augmentation_config["brightness_range"])
                )
            )

        if augmentation_config.get("contrast_range"):
            transform_list.append(
                RandomContrast(
                    contrast_range=tuple(augmentation_config["contrast_range"])
                )
            )

        if augmentation_config.get("saturation_range"):
            transform_list.append(
                RandomSaturation(
                    saturation_range=tuple(augmentation_config["saturation_range"])
                )
            )
    else:
        # Validation/test transforms (no augmentation)
        transform_list.append(transforms.Resize(image_size))

    # Convert to tensor and normalize
    transform_list.append(transforms.ToTensor())
    transform_list.append(
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    )  # ImageNet normalization

    return transforms.Compose(transform_list)
