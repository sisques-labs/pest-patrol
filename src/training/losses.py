"""Loss functions for pest classification."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """Initialize Focal Loss.

        Args:
            alpha: Weighting factor for rare class
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Predicted logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross Entropy Loss with Label Smoothing."""

    def __init__(self, smoothing: float = 0.1):
        """Initialize Label Smoothing Cross Entropy.

        Args:
            smoothing: Label smoothing factor
        """
        super().__init__()
        self.smoothing = smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label smoothing cross entropy loss.

        Args:
            inputs: Predicted logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)

        Returns:
            Label smoothing cross entropy loss
        """
        log_probs = F.log_softmax(inputs, dim=1)
        num_classes = inputs.size(1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


def create_loss(loss_config: dict) -> nn.Module:
    """Create loss function from configuration.

    Args:
        loss_config: Loss configuration dictionary

    Returns:
        Loss function module

    Raises:
        ValueError: If loss name is not supported
    """
    loss_name = loss_config.get("name", "cross_entropy").lower()

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()

    elif loss_name == "focal_loss":
        alpha = loss_config.get("focal_alpha", 0.25)
        gamma = loss_config.get("focal_gamma", 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_name == "label_smoothing":
        smoothing = loss_config.get("label_smoothing", 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)

    else:
        raise ValueError(
            f"Unsupported loss function: {loss_name}. "
            "Supported: cross_entropy, focal_loss, label_smoothing"
        )
