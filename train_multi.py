"""Multi-model training script for pest classification.

This script trains multiple models sequentially and generates a comparison report.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

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


def train_single_model(
    model_name: str,
    config,
    train_loader,
    val_loader,
    class_to_idx: dict,
    device: torch.device,
    base_output_dir: Path,
    logger,
) -> Dict:
    """Train a single model.

    Args:
        model_name: Name of the model architecture
        config: Configuration object
        train_loader: Training data loader
        val_loader: Validation data loader
        class_to_idx: Class to index mapping
        device: Device to train on
        base_output_dir: Base output directory
        logger: Logger instance

    Returns:
        Dictionary with model training results
    """
    logger.info("=" * 70)
    logger.info(f"Training model: {model_name}")
    logger.info("=" * 70)

    # Create model-specific output directory
    model_output_dir = base_output_dir / f"models" / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    (model_output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (model_output_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Create model
    logger.info(f"Creating model: {model_name}")
    model = create_model(
        model_name=model_name,
        num_classes=len(class_to_idx),
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
    )
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Create loss function
    criterion = create_loss(config.loss)

    # Create optimizer
    optimizer = create_optimizer(model, config.training)

    # Create scheduler
    scheduler = create_scheduler(optimizer, config.training, config.training.num_epochs)

    # Create trainer with model-specific output directory
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config.to_dict(),
        output_dir=str(model_output_dir),
    )

    # Train model
    history = trainer.train(config.training.num_epochs)

    # Get final results
    results = {
        "model_name": model_name,
        "num_parameters": num_params,
        "best_val_acc": trainer.best_val_acc,
        "best_val_loss": trainer.best_val_loss,
        "final_train_acc": history["train_acc"][-1] if history["train_acc"] else 0.0,
        "final_val_acc": history["val_acc"][-1] if history["val_acc"] else 0.0,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else float("inf"),
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else float("inf"),
        "checkpoint_path": str(model_output_dir / "checkpoints" / "best_model.pth"),
        "output_dir": str(model_output_dir),
    }

    logger.info(f"Model {model_name} training completed!")
    logger.info(f"  Best validation accuracy: {trainer.best_val_acc*100:.2f}%")
    logger.info(f"  Best validation loss: {trainer.best_val_loss:.4f}")

    return results


def generate_comparison_report(results: List[Dict], output_dir: Path, logger):
    """Generate a comparison report for all trained models.

    Args:
        results: List of training results for each model
        output_dir: Output directory to save report
        logger: Logger instance
    """
    # Sort by validation accuracy (best first)
    sorted_results = sorted(results, key=lambda x: x["best_val_acc"], reverse=True)

    # Create report
    report = {
        "summary": {
            "total_models": len(results),
            "best_model": sorted_results[0]["model_name"] if sorted_results else None,
            "best_accuracy": sorted_results[0]["best_val_acc"] if sorted_results else 0.0,
        },
        "models": sorted_results,
    }

    # Save JSON report
    report_path = output_dir / "model_comparison.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Comparison report saved to: {report_path}")

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Model':<20} {'Params':<15} {'Val Acc':<12} {'Val Loss':<12} {'Checkpoint':<30}")
    logger.info("-" * 70)
    for result in sorted_results:
        logger.info(
            f"{result['model_name']:<20} "
            f"{result['num_parameters']:>14,} "
            f"{result['best_val_acc']*100:>10.2f}% "
            f"{result['best_val_loss']:>11.4f} "
            f"{result['checkpoint_path']:<30}"
        )
    logger.info("=" * 70)
    logger.info(f"\nBest model: {sorted_results[0]['model_name']} "
                f"(Val Acc: {sorted_results[0]['best_val_acc']*100:.2f}%)")
    logger.info("=" * 70 + "\n")


def main():
    """Main function for multi-model training."""
    parser = argparse.ArgumentParser(
        description="Train multiple models sequentially for comparison"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="List of model names to train (overrides config)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have checkpoints",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    seed = config.get("seed", 42)
    set_seed(seed)

    # Setup logger
    logger = setup_logger(
        level=config.get("logging", {}).get("level", "INFO"),
        log_file=Path(config.paths.logs_dir) / "multi_training.log",
    )
    logger.info("=" * 70)
    logger.info("Starting Multi-Model Training")
    logger.info("=" * 70)

    # Determine which models to train
    if args.models:
        models_to_train = args.models
    elif hasattr(config, "multi_model") and hasattr(config.multi_model, "models"):
        models_to_train = config.multi_model.models
    else:
        # Default models if nothing specified
        models_to_train = [
            "resnet34",
            "resnet50",
            "efficientnet_b0",
            "efficientnet_b4",
            "mobilenet_v3_large",
        ]
        logger.warning(
            "No models specified in config or arguments. Using default models: "
            f"{models_to_train}"
        )

    logger.info(f"Models to train: {models_to_train}")
    logger.info(f"Total models: {len(models_to_train)}")

    # Device setup
    use_gpu = config.get("device", {}).get("use_gpu", True)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directories
    output_dir = Path(config.paths.outputs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)

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

    # Create data loaders (shared across all models)
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

    # Save class mapping (once for all models)
    class_mapping_path = output_dir / "class_mapping.json"
    with open(class_mapping_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    logger.info(f"Class mapping saved to: {class_mapping_path}")

    # Train each model
    all_results = []
    for i, model_name in enumerate(models_to_train, 1):
        logger.info(f"\n[{i}/{len(models_to_train)}] Processing model: {model_name}")

        # Check if model already exists
        if args.skip_existing:
            model_checkpoint = output_dir / "models" / model_name / "checkpoints" / "best_model.pth"
            if model_checkpoint.exists():
                logger.info(f"Skipping {model_name} - checkpoint already exists")
                continue

        try:
            results = train_single_model(
                model_name=model_name,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                class_to_idx=class_to_idx,
                device=device,
                base_output_dir=output_dir,
                logger=logger,
            )
            all_results.append(results)
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}", exc_info=True)
            continue

    # Generate comparison report
    if all_results:
        generate_comparison_report(all_results, output_dir, logger)
        logger.info("Multi-model training completed successfully!")
    else:
        logger.warning("No models were successfully trained.")


if __name__ == "__main__":
    main()

