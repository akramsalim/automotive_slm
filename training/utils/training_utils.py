# training/utils/training_utils.py
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json

def save_training_state(state: Dict,
                       output_dir: Path,
                       filename: str):
    """Save training state to file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    torch.save(state, output_path)

def load_training_state(checkpoint_path: Path) -> Dict:
    """Load training state from file."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return torch.load(checkpoint_path)

def compute_gradient_norm(model: nn.Module) -> float:
    """Compute gradient norm for model parameters."""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def get_learning_rates(optimizer: torch.optim.Optimizer) -> List[float]:
    """Get current learning rates from optimizer."""
    return [group["lr"] for group in optimizer.param_groups]

def calculate_steps_per_epoch(dataset_size: int,
                            batch_size: int,
                            gradient_accumulation_steps: int) -> int:
    """Calculate steps per epoch."""
    return dataset_size // (batch_size * gradient_accumulation_steps)

def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"