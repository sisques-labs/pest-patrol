"""Base model class for pest classification."""

from abc import ABC, abstractmethod
import torch.nn as nn


class BaseClassifier(nn.Module, ABC):
    """Base class for pest classification models."""

    def __init__(self, num_classes: int, dropout: float = 0.5):
        """Initialize base classifier.

        Args:
            num_classes: Number of pest classes to classify
            dropout: Dropout probability
        """
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout

    @abstractmethod
    def forward(self, x):
        """Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Model output logits
        """
        pass

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters.

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
