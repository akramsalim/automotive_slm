from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: List[str] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "query_key_value",
                "dense",
                "dense_h_to_4h",
                "dense_4h_to_h"
            ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type
        }

@dataclass
class ModelConfig:
    model_key: str
    max_length: int
    lora_config: LoRAConfig
    device: str
    dtype: str
    num_labels: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "max_length": self.max_length,
            "lora_config": self.lora_config.to_dict(),
            "device": self.device,
            "dtype": self.dtype,
            "num_labels": self.num_labels
        }