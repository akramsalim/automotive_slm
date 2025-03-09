# data/utils/data_utils.py
from typing import Dict, List, Optional
import torch
import numpy as np
from collections import Counter

def analyze_data_distribution(data: List[Dict]) -> Dict:
    """Analyze distribution of commands and parameters in dataset."""
    analysis = {
        "total_samples": len(data),
        "intent_distribution": Counter(),
        "parameter_statistics": {}
    }

    # Analyze intents
    for item in data:
        if "intent" in item:
            analysis["intent_distribution"][item["intent"]] += 1

    # Analyze parameters
    all_params = {}
    for item in data:
        if "parameters" in item:
            for param, value in item["parameters"].items():
                if param not in all_params:
                    all_params[param] = []
                all_params[param].append(value)

    for param, values in all_params.items():
        analysis["parameter_statistics"][param] = {
            "unique_values": len(set(values)),
            "distribution": Counter(values)
        }

    # Add safety level analysis if present
    if any("safety_level" in item for item in data):
        analysis["safety_levels"] = Counter(
            item.get("safety_level", "unknown") for item in data
        )

    # Add vehicle state statistics if present
    if any("vehicle_state" in item for item in data):
        analysis["vehicle_states"] = {}
        
        # Extract all vehicle state fields
        state_fields = set()
        for item in data:
            if "vehicle_state" in item:
                state_fields.update(item["vehicle_state"].keys())
        
        # Analyze each field
        for field in state_fields:
            values = [
                item["vehicle_state"].get(field) 
                for item in data 
                if "vehicle_state" in item and field in item["vehicle_state"]
            ]
            
            # For numeric fields, provide statistics
            if all(isinstance(v, (int, float)) for v in values if v is not None):
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    analysis["vehicle_states"][field] = {
                        "mean": np.mean(numeric_values),
                        "min": min(numeric_values),
                        "max": max(numeric_values),
                        "distribution": None  # Too many unique values for numeric fields
                    }
            else:
                # For categorical fields, provide distribution
                analysis["vehicle_states"][field] = {
                    "unique_values": len(set(values)),
                    "distribution": Counter(values)
                }

    return analysis

def compute_class_weights(labels: List[str], smoothing_factor: float = 0.1) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets with smoothing.
    
    Args:
        labels: List of label strings or indices
        smoothing_factor: Factor to smooth weights (0 = no smoothing)
        
    Returns:
        Tensor of weights for each class
    """
    label_counts = Counter(labels)
    total = len(labels)
    num_classes = len(set(labels))
    
    # Initialize weights tensor
    weights = torch.zeros(num_classes)
    
    # Compute inverse frequency with smoothing
    for label, count in label_counts.items():
        if isinstance(label, str):
            # Convert string label to index if needed
            try:
                label_idx = int(label)
            except ValueError:
                # If we can't convert to int, skip this label
                continue
        else:
            label_idx = label
            
        if 0 <= label_idx < num_classes:
            # Apply smoothing: weight = 1 / (count + alpha * total / num_classes)
            # This smooths the weights toward uniform distribution
            smoothed_count = count + smoothing_factor * (total / num_classes)
            weights[label_idx] = total / (num_classes * smoothed_count)
    
    # Normalize weights
    if weights.sum() > 0:
        weights = weights / weights.sum() * num_classes
    else:
        # If all weights are zero, use uniform weights
        weights = torch.ones(num_classes)
    
    return weights

def check_data_quality(data: List[Dict]) -> Dict:
    """
    Check quality of dataset.
    
    Args:
        data: List of data items
        
    Returns:
        Dictionary of quality metrics
    """
    quality = {
        "total_samples": len(data),
        "missing_fields": {},
        "empty_fields": {},
        "issues_found": 0
    }
    
    # Required fields to check
    required_fields = ["command", "intent", "parameters"]
    
    for field in required_fields:
        missing = sum(1 for item in data if field not in item)
        quality["missing_fields"][field] = missing
        quality["issues_found"] += missing
        
        # Check for empty fields
        if field in ["parameters"]:
            empty = sum(1 for item in data if field in item and not item[field])
            quality["empty_fields"][field] = empty
            quality["issues_found"] += empty
    
    # Check for duplicate commands
    commands = [item.get("command") for item in data if "command" in item]
    command_counts = Counter(commands)
    quality["duplicate_commands"] = sum(count - 1 for count in command_counts.values() if count > 1)
    quality["issues_found"] += quality["duplicate_commands"]
    
    # Calculate overall quality score (0-100)
    if len(data) > 0:
        quality_score = 100 - (quality["issues_found"] / len(data) * 100)
        quality["quality_score"] = min(100, max(0, quality_score))
    else:
        quality["quality_score"] = 0
        
    return quality

def augment_data(data: List[Dict], augmentation_factor: float = 0.2) -> List[Dict]:
    """
    Augment dataset with variations.
    
    Args:
        data: Original dataset
        augmentation_factor: Fraction of data to augment
        
    Returns:
        Augmented dataset
    """
    augmented_data = data.copy()
    num_to_augment = int(len(data) * augmentation_factor)
    
    if num_to_augment <= 0:
        return augmented_data
    
    # Select random samples to augment
    indices_to_augment = np.random.choice(len(data), num_to_augment, replace=False)
    
    for idx in indices_to_augment:
        item = data[idx]
        
        # Create augmented copy
        augmented_item = item.copy()
        
        # Modify command by adding variations
        if "command" in item:
            command = item["command"]
            augmented_command = command
            
            # Add variations like "please", changing word order, etc.
            variations = [
                f"Please {command.lower()}",
                f"I want to {command.lower()}",
                f"Can you {command.lower()}"
            ]
            
            augmented_command = np.random.choice(variations)
            augmented_item["command"] = augmented_command
        
        # Modify parameters slightly
        if "parameters" in item and item["parameters"]:
            augmented_params = item["parameters"].copy()
            
            # Find numeric parameters and slightly modify them
            for param, value in augmented_params.items():
                try:
                    # Modify numeric parameters slightly
                    if isinstance(value, str) and value.isdigit():
                        numeric_value = int(value)
                        modified_value = numeric_value + np.random.randint(-2, 3)
                        augmented_params[param] = str(max(0, modified_value))
                except (ValueError, TypeError):
                    # Not a numeric parameter, leave unchanged
                    pass
            
            augmented_item["parameters"] = augmented_params
        
        # Add the augmented item to the dataset
        augmented_data.append(augmented_item)
    
    return augmented_data