"""Training script for pest classification model."""

import os
import random
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.data import get_data_loaders, get_transforms
from src.models import create_model
from src.training import Trainer, create_loss, create_metrics
from src.utils import create_optimizer, create_scheduler, setup_logger


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""
    # Load configuration
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    config = load_config(config_path)

    # Set random seed
    seed = config.get("seed", 42)
    set_seed(seed)

    # Setup paths
    project_root = Path(__file__).parent

    # Create output directories (organized by model name)
    base_output_dir = Path(config.paths.outputs_dir)
    model_name = config.model.name
    output_dir = base_output_dir / "models" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(
        level=config.get("logging", {}).get("level", "INFO"),
        log_file=output_dir / "logs" / "training.log",
    )
    logger.info("=" * 50)
    logger.info("Starting Pest Classification Training")
    logger.info("=" * 50)
    logger.info(f"Model: {model_name}")
    logger.info(f"Output directory: {output_dir}")

    # Device setup
    use_gpu = config.get("device", {}).get("use_gpu", True)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Get data transforms
    train_transform = get_transforms(
        image_size=tuple(config.dataset.image_size),
        augmentation_config=config.augmentation.train,
        is_training=True,
    )
    val_transform = get_transforms(
        image_size=tuple(config.dataset.image_size),
        augmentation_config=config.augmentation.val,
        is_training=False,
    )

    # Create data loaders
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader, class_to_idx = get_data_loaders(
        data_dir=config.paths.raw_data_dir,
        batch_size=config.training.batch_size,
        train_split=config.dataset.train_split,
        val_split=config.dataset.val_split,
        test_split=config.dataset.test_split,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=val_transform,
        random_seed=seed,
    )

    num_classes = len(class_to_idx)
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {list(class_to_idx.keys())}")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Save class mapping (save in base output dir for shared access)
    import json

    base_output_dir = Path(config.paths.outputs_dir)
    class_mapping_path = base_output_dir / "class_mapping.json"
    with open(class_mapping_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    logger.info(f"Class mapping saved to: {class_mapping_path}")

    # Create model
    logger.info(f"Creating model: {config.model.name}")
    model = create_model(
        model_name=config.model.name,
        num_classes=num_classes,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create loss function
    criterion = create_loss(config.loss)

    # Create optimizer
    optimizer = create_optimizer(model, config.training)

    # Create scheduler
    scheduler = create_scheduler(optimizer, config.training, config.training.num_epochs)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config.to_dict(),
        output_dir=str(output_dir),
    )

    # Train model
    history = trainer.train(config.training.num_epochs)

    logger.info("Training completed successfully!")
    logger.info(f"Best validation accuracy: {trainer.best_val_acc*100:.2f}%")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
