# models/utils/model_utils.py
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedTokenizer, PreTrainedModel


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / (1024 * 1024)

def calculate_flops(model: PreTrainedModel,
                   input_shape: tuple) -> int:
    """Estimate FLOPs for model."""
    # Implement FLOPs calculation
    return 0  # Placeholder

def optimize_model_memory(model: PreTrainedModel,
                        dtype: Optional[torch.dtype] = None) -> PreTrainedModel:
    """Optimize model memory usage."""
    if dtype is None:
        dtype = torch.float16
        
    # Convert model to specified dtype
    model = model.to(dtype)
    
    return model

def add_automotive_tokens(model: PreTrainedModel,
                        tokenizer: PreTrainedTokenizer,
                        tokens: List[str]) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Add automotive-specific tokens to model and tokenizer."""
    # Add tokens to tokenizer
    num_added_tokens = tokenizer.add_tokens(tokens)
    
    # Resize model embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer