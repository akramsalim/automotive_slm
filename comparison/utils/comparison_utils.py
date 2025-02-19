# comparison/utils/comparison_utils.py
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_comparison_matrix(results: Dict[str, Dict],
                           metrics: List[str]) -> pd.DataFrame:
    """Create comparison matrix for multiple metrics."""
    data = []
    
    for model_name, model_results in results.items():
        row = {"model": model_name}
        for metric in metrics:
            category, metric_name = metric.split(".")
            if category in model_results and metric_name in model_results[category]:
                row[metric] = model_results[category][metric_name]
            else:
                row[metric] = None
        data.append(row)
    
    return pd.DataFrame(data)

def visualize_model_comparison(results: Dict[str, Dict],
                             output_dir: Path,
                             plot_types: Optional[List[str]] = None):
    """Create visualization plots for model comparison."""
    if plot_types is None:
        plot_types = ["performance", "resource", "radar"]
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for plot_type in plot_types:
        if plot_type == "performance":
            _create_performance_plot(results, output_dir)
        elif plot_type == "resource":
            _create_resource_plot(results, output_dir)
        elif plot_type == "radar":
            _create_radar_plot(results, output_dir)

def _create_performance_plot(results: Dict[str, Dict], output_dir: Path):
    """Create performance comparison plot."""
    plt.figure(figsize=(12, 6))
    
    performance_data = []
    for model, metrics in results.items():
        if "performance" in metrics:
            for metric, value in metrics["performance"].items():
                performance_data.append({
                    "Model": model,
                    "Metric": metric,
                    "Value": value
                })
    
    df = pd.DataFrame(performance_data)
    sns.barplot(data=df, x="Model", y="Value", hue="Metric")
    plt.title("Model Performance Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "performance_comparison.png")
    plt.close()

def _create_resource_plot(results: Dict[str, Dict], output_dir: Path):
    """Create resource utilization plot."""
    plt.figure(figsize=(10, 6))
    
    resource_data = []
    for model, metrics in results.items():
        if "resource" in metrics:
            for metric, value in metrics["resource"].items():
                resource_data.append({
                    "Model": model,
                    "Metric": metric,
                    "Value": value
                })
    
    df = pd.DataFrame(resource_data)
    sns.barplot(data=df, x="Model", y="Value", hue="Metric")
    plt.title("Resource Utilization Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "resource_comparison.png")
    plt.close()

def _create_radar_plot(results: Dict[str, Dict], output_dir: Path):
    """Create radar plot for model comparison."""
    metrics = ["accuracy", "safety", "latency", "memory"]
    num_metrics = len(metrics)
    
    angles = [n / float(num_metrics) * 2 * np.pi for n in range(num_metrics)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    for model, metrics_dict in results.items():
        values = []
        for metric in metrics:
            # Normalize values between 0 and 1
            value = _get_normalized_value(metrics_dict, metric)
            values.append(value)
        values += values[:1]
        
        ax.plot(angles, values, linewidth=1, label=model)
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], metrics)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title("Model Comparison Radar Chart")
    plt.tight_layout()
    plt.savefig(output_dir / "radar_comparison.png")
    plt.close()

def _get_normalized_value(metrics_dict: Dict, metric: str) -> float:
    """Get normalized value for metric."""
    # Implement normalization logic based on metric type
    return 0.5  # Placeholder

def calculate_relative_improvements(baseline_results: Dict,
                                 comparison_results: Dict[str, Dict]) -> Dict[str, Dict]:
    """Calculate relative improvements over baseline."""
    improvements = {}
    
    for model, metrics in comparison_results.items():
        improvements[model] = {}
        for category, category_metrics in metrics.items():
            if category in baseline_results:
                improvements[model][category] = {}
                for metric, value in category_metrics.items():
                    if metric in baseline_results[category]:
                        baseline = baseline_results[category][metric]
                        if baseline != 0:
                            rel_improvement = (value - baseline) / baseline * 100
                            improvements[model][category][metric] = rel_improvement
    
    return improvements

def rank_models(results: Dict[str, Dict],
               weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Rank models based on weighted metrics."""
    if weights is None:
        weights = {
            "performance": 0.4,
            "safety": 0.3,
            "resource": 0.2,
            "automotive": 0.1
        }
    
    rankings = []
    for model, metrics in results.items():
        score = 0
        for category, weight in weights.items():
            if category in metrics:
                category_score = np.mean(list(metrics[category].values()))
                score += weight * category_score
        
        rankings.append({
            "model": model,
            "score": score
        })
    
    return pd.DataFrame(rankings).sort_values("score", ascending=False)

def generate_latex_tables(results: Dict[str, Dict]) -> Dict[str, str]:
    """Generate LaTeX tables for paper/thesis."""
    latex_tables = {}
    
    # Performance table
    performance_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{l"
    
    metrics = set()
    for metrics_dict in results.values():
        for category in metrics_dict:
            metrics.update(metrics_dict[category].keys())
    
    for _ in metrics:
        performance_table += "c"
    
    performance_table += "}\n\\toprule\nModel & "
    performance_table += " & ".join(metrics) + " \\\\\n\\midrule\n"
    
    for model, metrics_dict in results.items():
        row = [model]
        for metric in metrics:
            found = False
            for category in metrics_dict:
                if metric in metrics_dict[category]:
                    row.append(f"{metrics_dict[category][metric]:.3f}")
                    found = True
                    break
            if not found:
                row.append("-")
        performance_table += " & ".join(row) + " \\\\\n"
    
    performance_table += "\\bottomrule\n\\end{tabular}\n\\caption{Model Performance Comparison}\n\\label{tab:model_comparison}\n\\end{table}"
    
    latex_tables["performance"] = performance_table
    
    return latex_tables