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
        # If intent is not in the label map, use default (first label)
        intent = item.get("intent", self.command_labels[0])
        label_id = self.label2id.get(intent, 0)
        
        # Add safety information if available
        safety_info = self._get_safety_score(item)
        
        # Create output dictionary
        result = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label_id, dtype=torch.long),
            "safety_score": torch.tensor(safety_info, dtype=torch.float)
        }
        
        # Add category information if available
        if "category" in item:
            # Get category label index
            category_labels = ["climate_control", "navigation", "vehicle_control", 
                            "media_control", "system_control"]
            category_idx = category_labels.index(item["category"]) if item["category"] in category_labels else 0
            result["category_labels"] = torch.tensor(category_idx, dtype=torch.long)
        
        # Add vehicle state information if available
        if "vehicle_state" in item:
            if "speed" in item["vehicle_state"]:
                result["vehicle_speed"] = torch.tensor(item["vehicle_state"]["speed"], dtype=torch.float)
            if "time_of_day" in item["vehicle_state"]:
                time_mapping = {"day": 0, "night": 1}
                result["time_of_day"] = torch.tensor(time_mapping.get(item["vehicle_state"]["time_of_day"], 0), dtype=torch.long)
            if "weather" in item["vehicle_state"]:
                weather_mapping = {"clear": 0, "rain": 1, "snow": 2, "fog": 3}
                result["weather_condition"] = torch.tensor(weather_mapping.get(item["vehicle_state"]["weather"], 0), dtype=torch.long)
                
        return result

    def _prepare_input_text(self, item: Dict) -> str:
        """Prepare input text from command item."""
        # Check if command key exists
        if "command" not in item:
            # Create default command from intent and parameters
            intent = item.get("intent", "unknown")
            parameters = item.get("parameters", {})
            command = f"{intent} {' '.join(f'{k}={v}' for k, v in parameters.items())}"
        else:
            command = item["command"]
        
        # Format parameters if available
        parameters = ""
        if "parameters" in item and item["parameters"]:
            parameters = " ".join(f"{k}={v}" for k, v in item["parameters"].items())
            
        # Construct the formatted input
        if "intent" in item:
            return f"<{item['intent']}> {command} <params> {parameters}"
        else:
            return f"{command} <params> {parameters}"

    def _extract_labels(self) -> List[str]:
        """Extract unique command labels from data."""
        # Default labels if none found in data
        default_labels = ["climate_control", "navigation", "vehicle_control", 
                         "media_control", "system_control"]
                         
        # Try to extract from data
        unique_intents = set()
        for item in self.data:
            if "intent" in item and item["intent"]:
                unique_intents.add(item["intent"])
        
        # If no intents found, use defaults
        if not unique_intents:
            return default_labels
            
        return sorted(list(unique_intents))

    def _get_safety_score(self, item: Dict) -> float:
        """Calculate safety score based on command rules."""
        if "safety_rules" not in item:
            return 1.0
        
        safety_rules = item["safety_rules"]
        parameters = item.get("parameters", {})
        
        # Implement safety scoring logic
        score = 1.0
        
        # Check parameter range rules
        for rule, value in safety_rules.items():
            if isinstance(value, dict):
                if "min" in value and "max" in value:
                    # Get parameter value, convert to float if possible
                    param_value = parameters.get(rule)
                    if param_value is not None:
                        try:
                            param_value = float(param_value)
                            if param_value < value["min"] or param_value > value["max"]:
                                score *= 0.5
                        except (ValueError, TypeError):
                            # Not a numeric parameter, can't check range
                            pass
            elif isinstance(value, bool):
                # Boolean rule check (e.g., input_while_moving: false)
                rule_param = rule.split('_')[0] if '_' in rule else rule
                if not value and rule_param in parameters:
                    score *= 0.8
        
        # Apply additional safety penalties if specified
        if "safety_level" in item:
            safety_levels = {
                "safe": 1.0,
                "warning": 0.7,
                "unsafe": 0.3,
                "critical": 0.0
            }
            level_score = safety_levels.get(item["safety_level"], 1.0)
            score = min(score, level_score)
        
        return score