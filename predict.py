"""Prediction script for pest classification."""

import argparse
import base64
import io
import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image

from src.config import load_config
from src.data import get_transforms
from src.models import create_model
from src.utils import setup_logger
from src.utils.checkpoints import CheckpointManager


def load_model(
    checkpoint_path: str,
    config_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, dict]:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        device: Device to load model on

    Returns:
        Tuple of (model, class_to_idx)
    """
    config = load_config(config_path)

    # Load class mapping
    output_dir = Path(config.paths.outputs_dir)
    class_mapping_path = output_dir / "class_mapping.json"
    if not class_mapping_path.exists():
        raise FileNotFoundError(
            f"Class mapping not found at {class_mapping_path}. "
            "Make sure you have trained a model first."
        )

    with open(class_mapping_path, "r") as f:
        class_to_idx = json.load(f)

    num_classes = len(class_to_idx)

    # Create model
    model = create_model(
        model_name=config.model.name,
        num_classes=num_classes,
        pretrained=False,
        dropout=config.model.dropout,
    )

    # Load checkpoint
    checkpoint_manager = CheckpointManager(output_dir / "checkpoints")
    checkpoint_manager.load_checkpoint(checkpoint_path, model)
    model.to(device)
    model.eval()

    return model, class_to_idx


def load_image_from_source(image_source: str) -> Image.Image:
    """Load image from various sources: file path, URL, or base64 string.

    Args:
        image_source: Path to image file, URL, or base64 encoded string

    Returns:
        PIL Image object
    """
    # Check if it's a URL first (most specific check)
    if image_source.startswith("http://") or image_source.startswith("https://"):
        try:
            response = requests.get(image_source, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image from URL {image_source}: {e}")

    # Check if it's a base64 string (data URI format or raw base64)
    if image_source.startswith("data:image"):
        try:
            # Handle data URI format: data:image/png;base64,...
            base64_data = image_source.split(",")[1]
            image_bytes = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return image
        except Exception as e:
            raise ValueError(f"Failed to decode base64 data URI: {e}")

    # Try as file path first (before attempting raw base64)
    image_path = Path(image_source)
    if image_path.exists():
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image from file {image_source}: {e}")

    # Last resort: try as raw base64 string
    # Only attempt if string is reasonably long and doesn't look like a path
    if len(image_source) > 50 and not any(c in image_source for c in ["\\", "/", ":"]):
        try:
            # Try to decode as base64
            image_bytes = base64.b64decode(image_source)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return image
        except Exception:
            # If base64 decoding fails, fall through to error
            pass

    raise ValueError(f"Invalid image source: {image_source}")


def predict_image(
    image_source: Union[str, Image.Image],
    model: torch.nn.Module,
    class_to_idx: dict,
    config,
    device: torch.device,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Predict pest class for a single image.

    Args:
        image_source: Path to image file, URL, base64 string, or PIL Image
        model: Trained model
        class_to_idx: Mapping from class names to indices
        config: Configuration object
        device: Device to run inference on
        top_k: Number of top predictions to return

    Returns:
        List of (class_name, probability) tuples, sorted by probability
    """
    # Load image if it's a string (path, URL, or base64)
    if isinstance(image_source, str):
        image = load_image_from_source(image_source)
    elif isinstance(image_source, Image.Image):
        image = image_source.convert("RGB")
    else:
        raise ValueError(f"Invalid image_source type: {type(image_source)}")

    # Get transforms
    transform = get_transforms(
        image_size=tuple(config.dataset.image_size),
        augmentation_config=None,
        is_training=False,
    )

    # Preprocess
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(class_to_idx)))

    # Convert to class names
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    predictions = [
        (idx_to_class[idx.item()], prob.item())
        for prob, idx in zip(top_probs[0], top_indices[0])
    ]

    return predictions


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Predict pest class from image")
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to image file or directory containing images",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (default: best_model.pth)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to show",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions (JSON format)",
    )

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger()

    # Load configuration
    config = load_config(args.config)

    # Device setup
    use_gpu = config.get("device", {}).get("use_gpu", True)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Determine checkpoint path
    if args.checkpoint is None:
        # Try new structure first: outputs/models/{model_name}/checkpoints/best_model.pth
        model_name = config.model.name
        new_path = Path(config.paths.outputs_dir) / "models" / model_name / "checkpoints" / "best_model.pth"
        # Fallback to old structure: outputs/checkpoints/best_model.pth
        old_path = Path(config.paths.outputs_dir) / "checkpoints" / "best_model.pth"
        
        if new_path.exists():
            checkpoint_path = new_path
        elif old_path.exists():
            checkpoint_path = old_path
            logger.warning(f"Using old checkpoint structure. Consider using: {new_path}")
        else:
            raise FileNotFoundError(
                f"Checkpoint not found. Tried:\n  - {new_path}\n  - {old_path}"
            )
    else:
        checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load model
    logger.info(f"Loading model from {checkpoint_path}")
    model, class_to_idx = load_model(str(checkpoint_path), args.config, device)

    # Process image(s)
    image_path_str = args.image_path
    all_predictions = {}

    # Check if it's a directory
    image_path = Path(image_path_str)
    if image_path.is_dir():
        # Directory of images
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_files = [
            f for f in image_path.iterdir()
            if f.suffix.lower() in valid_extensions
        ]

        logger.info(f"Found {len(image_files)} images")
        for img_file in image_files:
            predictions = predict_image(
                str(img_file),
                model,
                class_to_idx,
                config,
                device,
                args.top_k,
            )
            all_predictions[str(img_file)] = predictions

            # Print results
            print(f"\nPredictions for {img_file.name}:")
            print("-" * 50)
            for class_name, prob in predictions:
                print(f"  {class_name}: {prob*100:.2f}%")
    else:
        # Single image (file, URL, or base64)
        logger.info(f"Predicting for image: {image_path_str[:100]}...")
        try:
            predictions = predict_image(
                image_path_str,
                model,
                class_to_idx,
                config,
                device,
                args.top_k,
            )
            # Use a descriptive key for the predictions dict
            if image_path.is_file():
                display_name = image_path.name
            elif image_path_str.startswith("http"):
                display_name = image_path_str.split("/")[-1][:50]
            elif image_path_str.startswith("data:image") or len(image_path_str) > 100:
                display_name = "base64_image"
            else:
                display_name = str(image_path_str)[:50]
            
            all_predictions[display_name] = predictions

            # Print results
            print(f"\nPredictions for {display_name}:")
            print("-" * 50)
            for class_name, prob in predictions:
                print(f"  {class_name}: {prob*100:.2f}%")
        except Exception as e:
            raise ValueError(f"Failed to process image: {e}")

    # Save predictions if output path specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_predictions, f, indent=2)
        logger.info(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    main()
