"""Checkpoint management utilities."""

import torch
from pathlib import Path
from typing import Dict, Optional
import json


class CheckpointManager:
    """Manage model checkpoints."""

    def __init__(self, checkpoint_dir: str):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.checkpoint_dir / "best_model.pth"

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict,
        filename: Optional[str] = None,
    ):
        """Save model checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dictionary of metrics
            filename: Optional custom filename
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch + 1}.pth"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def save_best(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict,
    ):
        """Save best model.

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dictionary of metrics
        """
        self.save_checkpoint(
            model, optimizer, epoch, metrics, filename="best_model.pth"
        )

    def load_checkpoint(
        self, checkpoint_path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into

        Returns:
            Dictionary containing checkpoint information
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint

    def load_best(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
        """Load best model checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into

        Returns:
            Dictionary containing checkpoint information
        """
        return self.load_checkpoint(self.best_model_path, model, optimizer)


