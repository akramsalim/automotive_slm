import torch
import torch.nn as nn
from typing import Dict, Optional
import numpy as np

class Optimizer:
    """Optimization utilities for model training."""
    
    @staticmethod
    def get_optimizer(model: nn.Module,
                     config: Dict) -> torch.optim.Optimizer:
        """
        Get optimizer based on configuration.
        
        Args:
            model: Model to optimize
            config: Optimizer configuration
        
        Returns:
            Configured optimizer
        """
        optimizer_type = config["type"].lower()
        params = config["params"]
        
        if optimizer_type == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params.get("weight_decay", 0.01)
            )
        elif optimizer_type == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=params["learning_rate"]
            )
        elif optimizer_type == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=params["learning_rate"],
                momentum=params.get("momentum", 0.9)
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    @staticmethod
    def get_scheduler(optimizer: torch.optim.Optimizer,
                     config: Dict,
                     num_training_steps: int):
        """
        Get learning rate scheduler.
        
        Args:
            optimizer: Optimizer
            config: Scheduler configuration
            num_training_steps: Total number of training steps
        
        Returns:
            Configured scheduler
        """
        scheduler_type = config["type"].lower()
        
        if scheduler_type == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=config.get("start_factor", 1.0),
                end_factor=config.get("end_factor", 0.1),
                total_iters=num_training_steps
            )
        elif scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_training_steps,
                eta_min=config.get("eta_min", 0)
            )
        elif scheduler_type == "constant":
            return torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=config.get("factor", 1.0)
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")