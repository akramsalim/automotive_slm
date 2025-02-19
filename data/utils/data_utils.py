from typing import Dict, List, Optional
import torch
import numpy as np
from collections import Counter

def analyze_data_distribution(data: List[Dict]) -> Dict:
    """Analyze distribution of commands and parameters in dataset."""
    analysis = {
        "total_samples": len(data),
        "intent_distribution": Counter(item["intent"] for item in data),
        "parameter_statistics": {}
    }

    # Analyze parameters
    all_params = {}
    for item in data:
        for param, value in item["parameters"].items():
            if param not in all_params:
                all_params[param] = []
            all_params[param].append(value)

    for param, values in all_params.items():
        analysis["parameter_statistics"][param] = {
            "unique_values": len(set(values)),
            "distribution": Counter(values)
        }

    return analysis

def compute_class_weights(labels: List[str]) -> torch.Tensor:
    """Compute class weights for imbalanced datasets."""
    label_counts = Counter(labels)
    total = len(labels)
    weights = torch.zeros(len(set(labels)))
    
    for label, count in label_counts.items():
        weights[int(label)] = total / (len(label_counts) * count)
    
    return weights