# visualization/performance_plots.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path

class PerformancePlotter:
    """Plotter for model performance visualizations."""
    
    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize performance plotter.
        
        Args:
            theme: Plotly theme to use
        """
        self.theme = theme
    
    def create_accuracy_plot(self,
                           results: Dict[str, Dict],
                           output_path: Optional[Path] = None) -> go.Figure:
        """Create interactive accuracy comparison plot."""
        # Prepare data
        models = list(results.keys())
        accuracies = [results[model]["performance"]["accuracy"] 
                     for model in models]
        
        # Create figure
        fig = go.Figure(data=[
            go.Bar(x=models, y=accuracies,
                  text=np.round(accuracies, 3),
                  textposition='auto')
        ])
        
        # Update layout
        fig.update_layout(
            title="Model Accuracy Comparison",
            xaxis_title="Model",
            yaxis_title="Accuracy",
            template=self.theme,
            showlegend=False
        )
        
        if output_path:
            fig.write_html(output_path / "accuracy_comparison.html")
        
        return fig
    
    def create_training_curves(self,
                             training_history: Dict[str, List],
                             output_path: Optional[Path] = None) -> go.Figure:
        """Create interactive training curves plot."""
        fig = go.Figure()
        
        # Add training loss
        fig.add_trace(go.Scatter(
            x=list(range(len(training_history["train_loss"]))),
            y=training_history["train_loss"],
            name="Training Loss",
            mode="lines"
        ))
        
        # Add validation loss
        if "val_loss" in training_history:
            fig.add_trace(go.Scatter(
                x=list(range(len(training_history["val_loss"]))),
                y=training_history["val_loss"],
                name="Validation Loss",
                mode="lines"
            ))
        
        # Update layout
        fig.update_layout(
            title="Training Progress",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template=self.theme
        )
        
        if output_path:
            fig.write_html(output_path / "training_curves.html")
        
        return fig
    
    def create_metrics_heatmap(self,
                             results: Dict[str, Dict],
                             output_path: Optional[Path] = None) -> go.Figure:
        """Create interactive metrics heatmap."""
        # Prepare data
        models = list(results.keys())
        metrics = []
        values = []
        
        for model in models:
            for category in results[model]:
                for metric, value in results[model][category].items():
                    metrics.append(f"{category}.{metric}")
                    values.append(value)
        
        metrics = list(set(metrics))
        data = np.zeros((len(models), len(metrics)))
        
        for i, model in enumerate(models):
            for j, metric in enumerate(metrics):
                category, metric_name = metric.split(".")
                if category in results[model] and metric_name in results[model][category]:
                    data[i, j] = results[model][category][metric_name]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=metrics,
            y=models,
            colorscale="RdYlBu"
        ))
        
        # Update layout
        fig.update_layout(
            title="Model Metrics Comparison",
            xaxis_title="Metric",
            yaxis_title="Model",
            template=self.theme
        )
        
        if output_path:
            fig.write_html(output_path / "metrics_heatmap.html")
        
        return fig
