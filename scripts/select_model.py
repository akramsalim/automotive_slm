# scripts/select_model.py
import torch
import yaml
import json
from pathlib import Path
import logging
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from comparison.model_evaluator import AutomotiveModelEvaluator
from data.command_generator import AutomotiveCommandGenerator
from safety.safety_checker import AutomotiveSafetyChecker
#from models.automotive_adapter import AutomotiveSafetyChecker
from data.automotive_dataset import AutomotiveDataset

class ModelSelector:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.logger = self._setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.command_generator = AutomotiveCommandGenerator()
        self.safety_checker = AutomotiveSafetyChecker()
        self.evaluator = AutomotiveModelEvaluator(device=str(self.device))
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / f"model_selection_{timestamp}.log")
        console_handler = logging.StreamHandler()
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def prepare_evaluation_data(self) -> AutomotiveDataset:
        """Prepare dataset for model evaluation."""
        self.logger.info("Generating evaluation dataset...")
        
        eval_data = self.command_generator.generate_synthetic_dataset(
            num_samples=self.config["evaluation"]["num_samples"],
            distribution=self.config["data"]["command_distribution"]
        )
        
        return AutomotiveDataset(eval_data, self.config["data"]["command_hierarchy"])
    
    def evaluate_models(self) -> Dict:
        """Evaluate all models and collect performance metrics."""
        self.logger.info("Starting model evaluation...")
        
        eval_dataset = self.prepare_evaluation_data()
        results = self.evaluator.compare_models(eval_dataset, self.safety_checker)
        
        self._save_results(results)
        return results
    
    def _save_results(self, results: Dict):
        """Save evaluation results."""
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        with open(output_dir / f"model_comparison_{timestamp}.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Generate and save visualizations
        self._generate_visualizations(results, output_dir, timestamp)
    
    def _generate_visualizations(self, results: Dict, output_dir: Path, timestamp: str):
        """Generate comparison visualizations."""
        # Convert results to DataFrame
        df = self._results_to_dataframe(results)
        
        # Create visualizations
        self._plot_performance_comparison(df, output_dir, timestamp)
        self._plot_resource_usage(df, output_dir, timestamp)
        self._plot_automotive_metrics(df, output_dir, timestamp)
    
    def _results_to_dataframe(self, results: Dict) -> pd.DataFrame:
        """Convert results to pandas DataFrame for visualization."""
        data = []
        for model_name, performance in results.items():
            data.append({
                "model": model_name,
                **vars(performance)
            })
        return pd.DataFrame(data)
    
    def _plot_performance_comparison(self, df: pd.DataFrame, 
                                   output_dir: Path, timestamp: str):
        """Plot performance metrics comparison."""
        plt.figure(figsize=(12, 6))
        metrics = ["accuracy", "safety_score", "automotive_specific_score"]
        
        df_melted = df.melt(
            id_vars=["model"],
            value_vars=metrics,
            var_name="Metric",
            value_name="Score"
        )
        
        sns.barplot(data=df_melted, x="model", y="Score", hue="Metric")
        plt.title("Model Performance Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_dir / f"performance_comparison_{timestamp}.png")
        plt.close()
    
    def _plot_resource_usage(self, df: pd.DataFrame, 
                            output_dir: Path, timestamp: str):
        """Plot resource usage comparison."""
        plt.figure(figsize=(12, 6))
        metrics = ["latency", "memory_usage"]
        
        df_melted = df.melt(
            id_vars=["model"],
            value_vars=metrics,
            var_name="Metric",
            value_name="Value"
        )
        
        sns.barplot(data=df_melted, x="model", y="Value", hue="Metric")
        plt.title("Resource Usage Comparison")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_dir / f"resource_usage_{timestamp}.png")
        plt.close()
    
    def _plot_automotive_metrics(self, df: pd.DataFrame, 
                               output_dir: Path, timestamp: str):
        """Plot automotive-specific metrics."""
        plt.figure(figsize=(10, 6))
        
        sns.scatterplot(
            data=df,
            x="throughput",
            y="automotive_specific_score",
            size="accuracy",
            hue="model",
            sizes=(100, 400)
        )
        
        plt.title("Automotive Performance vs Throughput")
        plt.xlabel("Throughput (commands/second)")
        plt.ylabel("Automotive-Specific Score")
        plt.tight_layout()
        
        plt.savefig(output_dir / f"automotive_metrics_{timestamp}.png")
        plt.close()
    
    def select_best_model(self, results: Dict) -> str:
        """Select the best model based on weighted metrics."""
        weights = self.config["selection_weights"]
        scores = {}
        
        for model_name, performance in results.items():
            # Calculate weighted score
            score = (
                weights["accuracy"] * performance.accuracy +
                weights["safety"] * performance.safety_score +
                weights["automotive"] * performance.automotive_specific_score -
                weights["latency"] * (performance.latency / 100) -  # Normalize latency
                weights["memory"] * (performance.memory_usage / 1000)  # Normalize memory
            )
            scores[model_name] = score
        
        best_model = max(scores.items(), key=lambda x: x[1])[0]
        
        self.logger.info(f"Selected best model: {best_model}")
        self.logger.info(f"Model scores: {scores}")
        
        return best_model

def main():
    # Parse command line arguments if needed
    config_path = "config/model_selection_config.yaml"
    
    # Initialize selector
    selector = ModelSelector(config_path)
    
    try:
        # Run evaluation
        results = selector.evaluate_models()
        
        # Select best model
        best_model = selector.select_best_model(results)
        
        # Save selection results
        output_dir = Path("results")
        with open(output_dir / "model_selection_result.json", "w") as f:
            json.dump({
                "best_model": best_model,
                "results": results
            }, f, indent=2)
            
    except Exception as e:
        selector.logger.error(f"Model selection failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()