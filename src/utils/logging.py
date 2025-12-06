"""Logging utilities for the project."""

import logging
import sys
from pathlib import Path
from typing import Optional
from torch.utils.tensorboard import SummaryWriter


def setup_logger(
    name: str = "pest_patrol",
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Setup logger with console and file handlers.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class TensorBoardLogger:
    """Wrapper for TensorBoard logging."""

    def __init__(self, log_dir: str):
        """Initialize TensorBoard logger.

        Args:
            log_dir: Directory to save TensorBoard logs
        """
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_path))

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value.

        Args:
            tag: Tag for the scalar
            value: Scalar value
            step: Step number
        """
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars.

        Args:
            main_tag: Main tag
            tag_scalar_dict: Dictionary of tag-value pairs
            step: Step number
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_image(self, tag: str, img_tensor, step: int):
        """Log an image.

        Args:
            tag: Tag for the image
            img_tensor: Image tensor
            step: Step number
        """
        self.writer.add_image(tag, img_tensor, step)

    def close(self):
        """Close the writer."""
        self.writer.close()
