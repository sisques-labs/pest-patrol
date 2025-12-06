"""Model factory for creating different architectures."""

import ssl
from typing import Any, Dict, Optional

import timm
import torch
import torchvision.models as models


def list_available_models() -> list:
    """List all available model architectures.

    Returns:
        List of model names
    """
    torchvision_models = [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "efficientnet_b4",
    ]
    timm_models = [
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "efficientnet_b4",
        "efficientnetv2_s",
        "efficientnetv2_m",
        "efficientnetv2_l",
    ]
    return sorted(set(torchvision_models + timm_models))


def create_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.5,
) -> torch.nn.Module:
    """Create a model instance.

    Args:
        model_name: Name of the model architecture
        num_classes: Number of classes to classify
        pretrained: Whether to use pretrained weights
        dropout: Dropout probability for classifier head

    Returns:
        Model instance

    Raises:
        ValueError: If model name is not supported
    """
    model_name_lower = model_name.lower()

    # Try torchvision models first
    if hasattr(models, model_name_lower):
        return _create_torchvision_model(
            model_name_lower, num_classes, pretrained, dropout
        )

    # Try timm models
    if timm.is_model(model_name_lower):
        return _create_timm_model(model_name_lower, num_classes, pretrained, dropout)

    # Try with different naming conventions
    model_aliases = {
        "resnet50": "resnet50",
        "resnet34": "resnet34",
        "resnet18": "resnet18",
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet_b4": "efficientnet_b4",
        "mobilenet_v3": "mobilenet_v3_large",
        "mobilenet_v3_small": "mobilenet_v3_small",
        "mobilenet_v3_large": "mobilenet_v3_large",
    }

    if model_name_lower in model_aliases:
        alias = model_aliases[model_name_lower]
        if hasattr(models, alias):
            return _create_torchvision_model(alias, num_classes, pretrained, dropout)
        if timm.is_model(alias):
            return _create_timm_model(alias, num_classes, pretrained, dropout)

    raise ValueError(
        f"Model '{model_name}' not supported. "
        f"Available models: {list_available_models()}"
    )


def _create_torchvision_model(
    model_name: str, num_classes: int, pretrained: bool, dropout: float
) -> torch.nn.Module:
    """Create a torchvision model.

    Args:
        model_name: Name of the model
        num_classes: Number of classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout probability

    Returns:
        Model instance
    """
    # Get model creation function
    model_fn = getattr(models, model_name)

    # Map model names to their weight enums (new API)
    weights_map = {
        "resnet18": models.ResNet18_Weights,
        "resnet34": models.ResNet34_Weights,
        "resnet50": models.ResNet50_Weights,
        "resnet101": models.ResNet101_Weights,
        "resnet152": models.ResNet152_Weights,
        "mobilenet_v3_small": models.MobileNet_V3_Small_Weights,
        "mobilenet_v3_large": models.MobileNet_V3_Large_Weights,
        "efficientnet_b0": models.EfficientNet_B0_Weights,
        "efficientnet_b1": models.EfficientNet_B1_Weights,
        "efficientnet_b2": models.EfficientNet_B2_Weights,
        "efficientnet_b3": models.EfficientNet_B3_Weights,
        "efficientnet_b4": models.EfficientNet_B4_Weights,
    }

    # Determine weights to use
    weights = None
    if pretrained:
        if model_name in weights_map:
            try:
                weights = weights_map[model_name].DEFAULT
            except AttributeError:
                weights = weights_map[model_name].IMAGENET1K_V1
        else:
            # Try to get weights dynamically
            try:
                weights_class = getattr(models, f"{model_name.replace('_', '').title()}_Weights")
                weights = getattr(weights_class, "DEFAULT", None) or getattr(weights_class, "IMAGENET1K_V1", None)
            except AttributeError:
                weights = None

    # Handle SSL certificate issues - disable SSL verification for macOS
    # This is a workaround for macOS SSL certificate issues with PyTorch downloads
    original_context = ssl._create_default_https_context
    ssl_context_modified = False
    
    if pretrained:
        # Disable SSL verification before attempting download to avoid certificate errors
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
            ssl_context_modified = True
            model = _try_create_model(model_fn, model_name, weights, pretrained, dropout, num_classes)
        except Exception as e:
            # Restore SSL context before re-raising
            if ssl_context_modified:
                ssl._create_default_https_context = original_context
            raise
        finally:
            # Always restore SSL context
            if ssl_context_modified:
                ssl._create_default_https_context = original_context
    else:
        # No download needed, create without SSL modification
        model = _try_create_model(model_fn, model_name, weights, pretrained, dropout, num_classes)

    return model


def _try_create_model(model_fn, model_name, weights, pretrained, dropout, num_classes):
    """Helper function to try creating a model with different APIs."""
    try:
        # Try new API with weights parameter
        if weights is not None:
            model = model_fn(weights=weights)
        else:
            model = model_fn(weights=None)
    except (TypeError, AttributeError):
        # Fallback to old API if new one doesn't work
        model = model_fn(pretrained=pretrained)

    # Modify classifier head
    if model_name.startswith("efficientnet"):
        # EfficientNet has classifier attribute
        in_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_features, num_classes),
        )
    elif model_name.startswith("mobilenet"):
        # MobileNet has classifier attribute
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_features, num_classes),
        )
    else:
        # ResNet and similar models
        in_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_features, num_classes),
        )

    return model


def _create_timm_model(
    model_name: str, num_classes: int, pretrained: bool, dropout: float
) -> torch.nn.Module:
    """Create a timm model.

    Args:
        model_name: Name of the model
        num_classes: Number of classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout probability

    Returns:
        Model instance
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )
    return model
