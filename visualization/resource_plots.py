# visualization/resource_plots.py
from typing import Dict, List, Optional, Union
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots


class ResourcePlotter:
    """Plotter for resource utilization visualizations."""
    
    def __init__(self, theme: str = "plotly_white"):
        self.theme = theme
    
    def create_memory_plot(self,
                          results: Dict[str, Dict],
                          output_path: Optional[Path] = None) -> go.Figure:
        """Create interactive memory usage plot."""
        # Prepare data
        models = list(results.keys())
        memory_usage = [results[model]["resource"]["memory_usage_mb"]
                       for model in models]
        
        # Create figure
        fig = go.Figure(data=[
            go.Bar(x=models, y=memory_usage,
                  text=np.round(memory_usage, 2),
                  textposition='auto')
        ])
        
        # Update layout
        fig.update_layout(
            title="Model Memory Usage",
            xaxis_title="Model",
            yaxis_title="Memory Usage (MB)",
            template=self.theme,
            showlegend=False
        )
        
        if output_path:
            fig.write_html(output_path / "memory_usage.html")
        
        return fig
    
    def create_inference_time_plot(self,
                                 results: Dict[str, Dict],
                                 output_path: Optional[Path] = None) -> go.Figure:
        """Create interactive inference time plot."""
        # Prepare data
        models = list(results.keys())
        inference_times = [results[model]["resource"]["inference_time_ms"]
                         for model in models]
        
        # Create figure
        fig = go.Figure(data=[
            go.Bar(x=models, y=inference_times,
                  text=np.round(inference_times, 2),
                  textposition='auto')
        ])
        
        # Update layout
        fig.update_layout(
            title="Model Inference Time",
            xaxis_title="Model",
            yaxis_title="Inference Time (ms)",
            template=self.theme,
            showlegend=False
        )
        
        if output_path:
            fig.write_html(output_path / "inference_time.html")
        
        return fig
    
    def create_resource_dashboard(self,
                                results: Dict[str, Dict],
                                output_path: Optional[Path] = None) -> go.Figure:
        """Create interactive resource utilization dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Memory Usage",
                "Inference Time",
                "GPU Utilization",
                "Power Consumption"
            )
        )
        
        # Memory usage
        models = list(results.keys())
        memory_usage = [results[model]["resource"]["memory_usage_mb"]
                       for model in models]
        fig.add_trace(
            go.Bar(x=models, y=memory_usage, name="Memory"),
            row=1, col=1
        )
        
        # Inference time
        inference_times = [results[model]["resource"]["inference_time_ms"]
                         for model in models]
        fig.add_trace(
            go.Bar(x=models, y=inference_times, name="Inference"),
            row=1, col=2
        )
        
        # GPU utilization (if available)
        if all("gpu_utilization" in results[model]["resource"] for model in models):
            gpu_util = [results[model]["resource"]["gpu_utilization"]
                       for model in models]
            fig.add_trace(
                go.Bar(x=models, y=gpu_util, name="GPU"),
                row=2, col=1
            )
        
        # Power consumption (if available)
        if all("power_consumption" in results[model]["resource"] for model in models):
            power = [results[model]["resource"]["power_consumption"]
                    for model in models]
            fig.add_trace(
                go.Bar(x=models, y=power, name="Power"),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="Resource Utilization Dashboard",
            template=self.theme,
            showlegend=True,
            height=800
        )
        
        if output_path:
            fig.write_html(output_path / "resource_dashboard.html")
        
        return fig
