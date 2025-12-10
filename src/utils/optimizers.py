"""Optimizer and scheduler creation utilities."""

from typing import Dict, Optional
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def create_optimizer(
    model: torch.nn.Module, optimizer_config: Dict
) -> optim.Optimizer:
    """Create optimizer from configuration.

    Args:
        model: Model to optimize
        optimizer_config: Optimizer configuration dictionary

    Returns:
        Optimizer instance

    Raises:
        ValueError: If optimizer name is not supported
    """
    optimizer_name = optimizer_config.get("optimizer", "adam").lower()
    learning_rate = optimizer_config.get("learning_rate", 0.001)
    weight_decay = optimizer_config.get("weight_decay", 0.0001)

    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "adamw":
        return optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "sgd":
        momentum = optimizer_config.get("momentum", 0.9)
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}. "
            "Supported: adam, adamw, sgd"
        )


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_config: Dict,
    num_epochs: int,
) -> Optional[lr_scheduler._LRScheduler]:
    """Create learning rate scheduler from configuration.

    Args:
        optimizer: Optimizer to schedule
        scheduler_config: Scheduler configuration dictionary
        num_epochs: Total number of epochs

    Returns:
        Scheduler instance or None if scheduler is disabled
    """
    scheduler_name = scheduler_config.get("scheduler", "none").lower()

    if scheduler_name == "none":
        return None

    scheduler_params = scheduler_config.get("scheduler_params", {})

    if scheduler_name == "cosine":
        params = scheduler_params.get("cosine", {})
        T_max = params.get("T_max", num_epochs)
        eta_min = params.get("eta_min", 0.0)
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    elif scheduler_name == "step":
        params = scheduler_params.get("step", {})
        step_size = params.get("step_size", 10)
        gamma = params.get("gamma", 0.1)
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_name == "plateau":
        params = scheduler_params.get("plateau", {})
        mode = params.get("mode", "min")
        factor = params.get("factor", 0.5)
        patience = params.get("patience", 5)
        min_lr = params.get("min_lr", 0.0)
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

    else:
        raise ValueError(
            f"Unsupported scheduler: {scheduler_name}. "
            "Supported: cosine, step, plateau, none"
        )


