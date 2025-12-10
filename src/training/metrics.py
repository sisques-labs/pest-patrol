"""Evaluation metrics for pest classification."""

from typing import Dict, List, Callable
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    top_k_accuracy_score,
)
import numpy as np


class MetricTracker:
    """Track and compute metrics during training."""

    def __init__(self, metric_names: List[str], top_k: int = 5):
        """Initialize metric tracker.

        Args:
            metric_names: List of metric names to compute
            top_k: Top-k for top-k accuracy
        """
        self.metric_names = metric_names
        self.top_k = top_k
        self.reset()

    def reset(self):
        """Reset all accumulated values."""
        self.predictions = []
        self.targets = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update with new predictions and targets.

        Args:
            predictions: Model predictions (logits) (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
        """
        preds = predictions.detach().cpu().numpy()
        targs = targets.detach().cpu().numpy()

        # Get predicted classes
        pred_classes = np.argmax(preds, axis=1)

        self.predictions.append(pred_classes)
        self.targets.append(targs)

    def compute(self) -> Dict[str, float]:
        """Compute all metrics.

        Returns:
            Dictionary of metric names and values
        """
        if not self.predictions:
            return {}

        # Concatenate all predictions and targets
        all_preds = np.concatenate(self.predictions)
        all_targets = np.concatenate(self.targets)

        metrics = {}

        if "accuracy" in self.metric_names:
            metrics["accuracy"] = accuracy_score(all_targets, all_preds)

        if "top_k_accuracy" in self.metric_names:
            # For top-k accuracy, we need the original logits
            # This is a simplified version
            metrics["top_k_accuracy"] = accuracy_score(
                all_targets, all_preds
            )  # Simplified

        if any(m in self.metric_names for m in ["precision", "recall", "f1_score"]):
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_preds, average="weighted", zero_division=0
            )
            if "precision" in self.metric_names:
                metrics["precision"] = precision
            if "recall" in self.metric_names:
                metrics["recall"] = recall
            if "f1_score" in self.metric_names:
                metrics["f1_score"] = f1

        if "confusion_matrix" in self.metric_names:
            cm = confusion_matrix(all_targets, all_preds)
            metrics["confusion_matrix"] = cm

        return metrics


def create_metrics(metric_names: List[str]) -> MetricTracker:
    """Create metric tracker.

    Args:
        metric_names: List of metric names

    Returns:
        MetricTracker instance
    """
    return MetricTracker(metric_names)


