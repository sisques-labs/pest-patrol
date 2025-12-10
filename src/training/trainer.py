"""Trainer class for training pest classification models."""

import os
from pathlib import Path
from typing import Dict, Optional, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import logging

from ..utils.logging import setup_logger, TensorBoardLogger
from ..utils.checkpoints import CheckpointManager


class Trainer:
    """Trainer class for model training and validation."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = None,
        config: Optional[Dict] = None,
        output_dir: Optional[str] = None,
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            config: Configuration dictionary
            output_dir: Directory to save outputs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config or {}
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")

        # Device setup
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.model.to(self.device)

        # Mixed precision training
        self.use_amp = self.config.get("training", {}).get("mixed_precision", True)
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient clipping
        self.gradient_clip = self.config.get("training", {}).get("gradient_clip", None)

        # Logging
        self.logger = setup_logger(
            name="Trainer", level=self.config.get("logging", {}).get("level", "INFO")
        )
        self.log_interval = self.config.get("logging", {}).get("log_interval", 10)
        use_tensorboard = self.config.get("logging", {}).get("tensorboard", True)
        if use_tensorboard:
            log_dir = self.output_dir / "logs" / "tensorboard"
            self.tb_logger = TensorBoardLogger(log_dir)
        else:
            self.tb_logger = None

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(self.output_dir / "checkpoints")

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                self.logger.info(
                    f"Epoch [{self.current_epoch + 1}] "
                    f"Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Acc: {100 * correct / total:.2f}%"
                )

                if self.tb_logger:
                    self.tb_logger.log_scalar(
                        "train/batch_loss", loss.item(), self.current_epoch * len(self.train_loader) + batch_idx
                    )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total

        metrics = {"loss": avg_loss, "accuracy": accuracy / 100}
        return metrics

    def validate(self) -> Dict[str, float]:
        """Validate the model.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total

        metrics = {"loss": avg_loss, "accuracy": accuracy / 100}
        return metrics

    def train(self, num_epochs: int) -> Dict[str, list]:
        """Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train

        Returns:
            Dictionary containing training history
        """
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        # Early stopping
        early_stopping = self.config.get("training", {}).get("early_stopping", {})
        if early_stopping.get("enabled", False):
            patience = early_stopping.get("patience", 10)
            min_delta = early_stopping.get("min_delta", 0.001)
            monitor = early_stopping.get("monitor", "val_loss")
            best_metric = float("inf") if "loss" in monitor else 0.0
            patience_counter = 0

        self.logger.info(f"Starting training for {num_epochs} epochs...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed precision: {self.use_amp}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])

            # Validate
            val_metrics = self.validate()
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(
                    self.scheduler,
                    optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Logging
            self.logger.info(
                f"Epoch [{epoch + 1}/{num_epochs}] - "
                f"Train Loss: {train_metrics['loss']:.4f} "
                f"Train Acc: {train_metrics['accuracy']*100:.2f}% - "
                f"Val Loss: {val_metrics['loss']:.4f} "
                f"Val Acc: {val_metrics['accuracy']*100:.2f}%"
            )

            # TensorBoard logging
            if self.tb_logger:
                self.tb_logger.log_scalar("train/epoch_loss", train_metrics["loss"], epoch)
                self.tb_logger.log_scalar("train/epoch_accuracy", train_metrics["accuracy"], epoch)
                self.tb_logger.log_scalar("val/epoch_loss", val_metrics["loss"], epoch)
                self.tb_logger.log_scalar("val/epoch_accuracy", val_metrics["accuracy"], epoch)
                if self.scheduler:
                    self.tb_logger.log_scalar(
                        "train/learning_rate",
                        self.optimizer.param_groups[0]["lr"],
                        epoch,
                    )

            # Save best model
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.checkpoint_manager.save_best(
                    self.model, self.optimizer, epoch, val_metrics
                )

            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]

            # Checkpoint saving
            save_interval = self.config.get("logging", {}).get(
                "save_checkpoint_interval", 5
            )
            if (epoch + 1) % save_interval == 0:
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics
                )

            # Early stopping
            if early_stopping.get("enabled", False):
                current_metric = (
                    val_metrics["loss"]
                    if "loss" in monitor
                    else val_metrics["accuracy"]
                )
                is_better = (
                    current_metric < (best_metric - min_delta)
                    if "loss" in monitor
                    else current_metric > (best_metric + min_delta)
                )

                if is_better:
                    best_metric = current_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.logger.info(
                            f"Early stopping triggered after {epoch + 1} epochs"
                        )
                        break

        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc*100:.2f}%")

        return history


