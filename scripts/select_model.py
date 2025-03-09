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
        
        # Clear existing handlers to avoid duplicates
        if logger.handlers:
            logger.handlers.clear()
        
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
        
        # Get the number of samples from config or use default
        num_samples = self.config["evaluation"].get("num_samples", 1000)
        
        # Get distribution or use default even distribution
        distribution = self.config.get("data", {}).get("command_distribution", {})
        if not distribution:
            # Create even distribution across all command types
            command_types = ["climate", "navigation", "vehicle_control", "media", "system"]
            distribution = {cmd_type: 1.0/len(command_types) for cmd_type in command_types}
            
        self.logger.info(f"Generating {num_samples} samples with distribution: {distribution}")
        
        # Generate data
        eval_data = self.command_generator.generate_synthetic_dataset(
            num_samples=num_samples,
            output_path="./eval_data",
            distribution=distribution
        )
        
        # Load command hierarchy from config
        command_hierarchy_path = Path(self.config.get("data", {}).get("hierarchy_path", "config/command_hierarchy.json"))
        try:
            with open(command_hierarchy_path) as f:
                command_hierarchy = json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load command hierarchy: {e}")
            command_hierarchy = {}
        
        self.logger.info("Dataset generation complete")
        return AutomotiveDataset(eval_data, self.tokenizer, command_labels=list(command_hierarchy.keys()))
    
    def _get_tokenizer(self):
        """Get tokenizer for the dataset."""
        from transformers import AutoTokenizer
        
        # Use a base model tokenizer - could be improved to use the actual model's tokenizer
        tokenizer_name = "distilbert-base-uncased"
        return AutoTokenizer.from_pretrained(tokenizer_name)
    
    def evaluate_models(self) -> Dict:
        """Evaluate all models and collect performance metrics."""
        self.logger.info("Starting model evaluation...")
        
        # Initialize tokenizer
        self.tokenizer = self._get_tokenizer()
        
        # Prepare evaluation dataset
        try:
            eval_dataset = self.prepare_evaluation_data()
            self.logger.info(f"Prepared evaluation dataset with {len(eval_dataset)} samples")
        except Exception as e:
            self.logger.error(f"Error preparing evaluation dataset: {e}")
            raise
        
        # Run evaluation
        try:
            results = self.evaluator.compare_models(eval_dataset, self.safety_checker)
            self.logger.info(f"Evaluation complete for {len(results)} models")
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {e}")
            raise
        
        # Save and visualize results
        self._save_results(results)
        return results
    
    def _save_results(self, results: Dict):
        """Save evaluation results."""
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        with open(output_dir / f"model_comparison_{timestamp}.json", "w") as f:
            # Convert model performance objects to dictionaries
            serializable_results = {}
            for model, perf in results.items():
                serializable_results[model] = vars(perf)
            
            json.dump(serializable_results, f, indent=2)
        
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
        
        # Create scatter plot
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
        # Get selection weights from config or use defaults
        weights = self.config.get("selection_weights", {
            "accuracy": 0.3,
            "safety": 0.3,
            "automotive": 0.2,
            "latency": 0.1,
            "memory": 0.1
        })
        
        scores = {}
        
        for model_name, performance in results.items():
            # Calculate weighted score
            perf_attrs = vars(performance)
            score = (
                weights["accuracy"] * perf_attrs["accuracy"] +
                weights["safety"] * perf_attrs["safety_score"] +
                weights["automotive"] * perf_attrs["automotive_specific_score"] -
                weights["latency"] * (perf_attrs["latency"] / 100) -  # Normalize latency
                weights["memory"] * (perf_attrs["memory_usage"] / 1000)  # Normalize memory
            )
            scores[model_name] = score
        
        # Find model with highest score
        best_model = max(scores.items(), key=lambda x: x[1])[0]
        
        self.logger.info(f"Selected best model: {best_model}")
        self.logger.info(f"Model scores: {scores}")
        
        return best_model

def main():
    # Parse command line arguments using argparse
    import argparse
    
    parser = argparse.ArgumentParser(description="Select the best model for automotive applications")
    parser.add_argument("--config", type=str, default="config/model_selection_config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="results",
                      help="Directory to save results")
    
    args = parser.parse_args()
    
    # Initialize selector
    selector = ModelSelector(args.config)
    
    try:
        # Run evaluation
        results = selector.evaluate_models()
        
        # Select best model
        best_model = selector.select_best_model(results)
        
        # Save selection results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "model_selection_result.json", "w") as f:
            # Convert to serializable format
            serializable_results = {}
            for model, perf in results.items():
                serializable_results[model] = vars(perf)
                
            json.dump({
                "best_model": best_model,
                "results": serializable_results
            }, f, indent=2)
            
        print(f"Best model selected: {best_model}")
        print(f"Results saved to {output_dir / 'model_selection_result.json'}")
            
    except Exception as e:
        selector.logger.error(f"Model selection failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()