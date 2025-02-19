from typing import Dict, List, Optional
import numpy as np
import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_evaluation_report(metrics: Dict,
                           output_dir: Path,
                           include_plots: bool = True):
    """Create comprehensive evaluation report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    if include_plots:
        create_metric_plots(metrics, output_dir)

def create_metric_plots(metrics: Dict, output_dir: Path):
    """Create visualization plots for metrics."""
    # Convert metrics to DataFrame
    df = pd.DataFrame(metrics)
    
    # Create plots
    plt.figure(figsize=(12, 6))
    
    # General metrics
    sns.barplot(data=pd.melt(df[["accuracy", "precision", "recall", "f1"]]))
    plt.title("General Performance Metrics")
    plt.tight_layout()
    plt.savefig(output_dir / "general_metrics.png")
    plt.close()
    
    # Safety metrics
    if "safety_compliance" in df.columns:
        plt.figure(figsize=(8, 6))
        sns.barplot(data=pd.melt(df[["safety_compliance", "critical_error_rate"]]))
        plt.title("Safety Metrics")
        plt.tight_layout()
        plt.savefig(output_dir / "safety_metrics.png")
        plt.close()

def calculate_confidence_intervals(metrics: List[float],
                                confidence: float = 0.95) -> Dict:
    """Calculate confidence intervals for metrics."""
    import scipy.stats as stats
    
    mean = np.mean(metrics)
    sem = stats.sem(metrics)
    ci = stats.t.interval(confidence, len(metrics)-1, mean, sem)
    
    return {
        "mean": mean,
        "ci_lower": ci[0],
        "ci_upper": ci[1]
    }

def compare_models(model_metrics: Dict[str, Dict]) -> pd.DataFrame:
    """Compare metrics across different models."""
    comparison_data = []
    
    for model_name, metrics in model_metrics.items():
        model_data = {"model": model_name}
        for category, category_metrics in metrics.items():
            for metric_name, value in category_metrics.items():
                model_data[f"{category}_{metric_name}"] = value
        comparison_data.append(model_data)
    
    return pd.DataFrame(comparison_data)