# visualization/utils/viz_utils.py
from typing import Dict, List, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path

from visualization.performance_plots import PerformancePlotter

def create_interactive_scatter(x: List[float],
                             y: List[float],
                             labels: List[str],
                             title: str,
                             xlabel: str,
                             ylabel: str) -> go.Figure:
    """Create interactive scatter plot."""
    fig = go.Figure(data=[
        go.Scatter(x=x, y=y, text=labels, mode="markers+text")
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel
    )
    
    return fig

def create_radar_chart(results: Dict[str, Dict],
                      metrics: List[str],
                      output_path: Optional[Path] = None) -> go.Figure:
    """Create interactive radar chart."""
    fig = go.Figure()
    
    for model, metrics_dict in results.items():
        values = []
        for metric in metrics:
            category, metric_name = metric.split(".")
            if category in metrics_dict and metric_name in metrics_dict[category]:
                values.append(metrics_dict[category][metric_name])
            else:
                values.append(0)
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            name=model,
            fill='toself'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True
    )
    
    if output_path:
        fig.write_html(output_path / "radar_chart.html")
    
    return fig

def create_parallel_coordinates(results: Dict[str, Dict],
                              metrics: List[str],
                              output_path: Optional[Path] = None) -> go.Figure:
    """Create parallel coordinates plot."""
    # Prepare data
    data = []
    for model, metrics_dict in results.items():
        row = {"Model": model}
        for metric in metrics:
            category, metric_name = metric.split(".")
            if category in metrics_dict and metric_name in metrics_dict[category]:
                row[metric] = metrics_dict[category][metric_name]
        data.append(row)
    
    df = pd.DataFrame(data)
    
    fig = px.parallel_coordinates(
        df,
        color="Model",
        dimensions=["Model"] + metrics
    )
    
    if output_path:
        fig.write_html(output_path / "parallel_coordinates.html")
    
    return fig

def export_plots_to_pdf(figures: List[go.Figure],
                       output_path: Path,
                       filename: str = "visualizations.pdf"):
    """Export plots to PDF format."""
    from plotly.subplots import make_subplots
    
    # Create PDF with all figures
    pdf_path = output_path / filename
    
    # Create subplot figure
    subplot_fig = make_subplots(
        rows=len(figures),
        cols=1,
        subplot_titles=[fig.layout.title.text for fig in figures]
    )
    
    # Add all traces from individual figures
    for i, fig in enumerate(figures, 1):
        for trace in fig.data:
            subplot_fig.add_trace(trace, row=i, col=1)
    
    # Update layout
    subplot_fig.update_layout(height=400*len(figures))
    
    # Save as PDF
    subplot_fig.write_image(str(pdf_path))

def create_plotly_dashboard(results: Dict[str, Dict],
                          output_path: Path) -> None:
    """Create comprehensive Plotly dashboard."""
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output
    
    
    
    app = dash.Dash(__name__)
    
    # Create layout
    app.layout = html.Div([
        html.H1("Model Comparison Dashboard"),
        
        dcc.Tabs([
            dcc.Tab(label="Performance", children=[
                dcc.Graph(id="accuracy-plot"),
                dcc.Graph(id="metrics-heatmap")
            ]),
            dcc.Tab(label="Resources", children=[
                dcc.Graph(id="memory-plot"),
                dcc.Graph(id="inference-plot")
            ]),
            dcc.Tab(label="Analysis", children=[
                dcc.Graph(id="radar-chart"),
                dcc.Graph(id="parallel-coords")
            ])
        ])
    ])
    
    # Add callbacks for interactivity
    @app.callback(
        Output("accuracy-plot", "figure"),
        Input("accuracy-plot", "id")
    )
    def update_accuracy_plot(_):
        plotter = PerformancePlotter()
        return plotter.create_accuracy_plot(results)
    
    # Save dashboard
    app.write_html(output_path / "dashboard.html")