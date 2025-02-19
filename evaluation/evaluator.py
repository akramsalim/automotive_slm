# evaluation/evaluator.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Union
import logging
import json
from pathlib import Path
from datetime import datetime
from .metrics import MetricsCalculator
import numpy as np
from tqdm import tqdm

class ModelEvaluator:
    """Evaluator for automotive models."""
    
    def __init__(self,
                 model: nn.Module,
                 tokenizer,
                 config: Dict,
                 device: str = "cuda"):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            config: Evaluation configuration
            device: Device to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.metrics = MetricsCalculator()
        self.logger = logging.getLogger(__name__)
    
    def evaluate(self,
                eval_dataloader: DataLoader,
                output_dir: Optional[Union[str, Path]] = None) -> Dict:
        """
        Evaluate model on dataset.
        
        Args:
            eval_dataloader: Evaluation data loader
            output_dir: Optional directory to save results
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        all_metrics = {
            "general": {},
            "safety": {},
            "automotive": {},
            "resource": {}
        }
        
        self.logger.info("Starting evaluation...")
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # Model inference
                outputs = self.model(**batch)
                
                # Calculate metrics
                batch_metrics = self._calculate_batch_metrics(outputs, batch)
                
                # Accumulate metrics
                for category in all_metrics:
                    for metric, value in batch_metrics[category].items():
                        if metric not in all_metrics[category]:
                            all_metrics[category][metric] = []
                        all_metrics[category][metric].append(value)
        
        # Average metrics
        averaged_metrics = self._average_metrics(all_metrics)
        
        # Save results if output directory provided
        if output_dir:
            self._save_results(averaged_metrics, output_dir)
        
        return averaged_metrics
    
    def _calculate_batch_metrics(self,
                               outputs: Dict,
                               batch: Dict) -> Dict:
        """Calculate metrics for a batch."""
        metrics = {
            "general": self.metrics.calculate_general_metrics(
                outputs["logits"], batch["labels"]
            ),
            "safety": self.metrics.calculate_safety_metrics(
                outputs["logits"], batch.get("safety_labels")
            ),
            "automotive": self.metrics.calculate_automotive_metrics(
                outputs["logits"], batch
            ),
            "resource": self.metrics.calculate_resource_metrics(
                self.model, outputs["logits"]
            )
        }
        
        return metrics
    
    def _average_metrics(self, metrics: Dict) -> Dict:
        """Average accumulated metrics."""
        averaged = {}
        
        for category, category_metrics in metrics.items():
            averaged[category] = {}
            for metric_name, values in category_metrics.items():
                averaged[category][metric_name] = np.mean(values)
        
        return averaged
    
    def _save_results(self,
                     metrics: Dict,
                     output_dir: Union[str, Path]):
        """Save evaluation results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"eval_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Saved evaluation results to {output_file}")
