import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional, Union
import logging
import numpy as np

class AutomotiveDataset(Dataset):
    def __init__(self,
                 data: List[Dict],
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 512,
                 command_labels: Optional[List[str]] = None):
        """
        Initialize automotive dataset.
        
        Args:
            data: List of command dictionaries
            tokenizer: Tokenizer for encoding commands
            max_length: Maximum sequence length
            command_labels: List of possible command labels
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.command_labels = command_labels or self._extract_labels()
        self.label2id = {label: i for i, label in enumerate(self.command_labels)}
        self.logger = logging.getLogger(__name__)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item with encoded command and labels."""
        item = self.data[idx]
        
        # Prepare input text
        input_text = self._prepare_input_text(item)
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Prepare labels
        label_id = self.label2id[item["intent"]]
        
        # Add safety information if available
        safety_info = torch.tensor(self._get_safety_score(item))
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": label_id,
            "safety_score": safety_info
        }

    def _prepare_input_text(self, item: Dict) -> str:
        """Prepare input text from command item."""
        parameters = " ".join(f"{k}={v}" for k, v in item["parameters"].items())
        return f"<{item['intent']}> {item['command']} <params> {parameters}"

    def _extract_labels(self) -> List[str]:
        """Extract unique command labels from data."""
        return sorted(list(set(item["intent"] for item in self.data)))

    def _get_safety_score(self, item: Dict) -> float:
        """Calculate safety score based on command rules."""
        if "safety_rules" not in item:
            return 1.0
        
        safety_rules = item["safety_rules"]
        parameters = item["parameters"]
        
        # Implement safety scoring logic based on rules
        # This is a simplified example
        score = 1.0
        for rule, value in safety_rules.items():
            if isinstance(value, dict):
                if "min" in value and "max" in value:
                    param_value = float(parameters.get(rule, 0))
                    if param_value < value["min"] or param_value > value["max"]:
                        score *= 0.5
            elif isinstance(value, bool):
                if not value:
                    score *= 0.8
        
        return score
