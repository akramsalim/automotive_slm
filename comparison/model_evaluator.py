# comparison/model_evaluator.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel
)
import numpy as np
from pathlib import Path
import json
import time
import psutil
import logging
from tqdm import tqdm

@dataclass
class ModelSpecs:
    name: str
    size: int  # Parameters in millions
    architecture: str
    input_size: int
    output_size: int
    memory_requirements: int  # MB

@dataclass
class ModelPerformance:
    accuracy: float
    latency: float  # ms
    memory_usage: float  # MB
    throughput: float  # commands/second
    safety_score: float
    automotive_specific_score: float

class ModelProfiler:
    """Profiles model performance and resource usage."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def measure_inference_time(self, model: PreTrainedModel, 
                             sample_input: torch.Tensor, 
                             num_runs: int = 100) -> float:
        """Measure average inference time."""
        times = []
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(**sample_input)
            
            # Actual measurement
            for _ in range(num_runs):
                start_time = time.perf_counter()
                _ = model(**sample_input)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        return np.mean(times)
    
    def measure_memory_usage(self, model: PreTrainedModel) -> float:
        """Measure model memory usage in MB."""
        memory_usage = 0
        for param in model.parameters():
            memory_usage += param.nelement() * param.element_size()
        return memory_usage / (1024 * 1024)  # Convert to MB
    
    def measure_throughput(self, model: PreTrainedModel, 
                          dataloader: torch.utils.data.DataLoader) -> float:
        """Measure commands processed per second."""
        model.eval()
        start_time = time.perf_counter()
        num_commands = 0
        
        with torch.no_grad():
            for batch in dataloader:
                _ = model(**batch)
                num_commands += batch["input_ids"].size(0)
        
        total_time = time.perf_counter() - start_time
        return num_commands / total_time

class AutomotiveModelEvaluator:
    """Evaluates models for automotive command processing."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.profiler = ModelProfiler()
        self.models = self._initialize_models()
        self.logger = logging.getLogger(__name__)
        
    def _initialize_models(self) -> Dict[str, Dict]:
        """Initialize all models for comparison."""
        return {
            "phi-2": {
                "name": "microsoft/phi-2",
                "class": AutoModelForCausalLM,
                "type": "causal_lm",
                "size": 2700
            },
            "bert-small": {
                "name": "prajjwal1/bert-small",
                "class": AutoModelForSequenceClassification,
                "type": "sequence_classification",
                "size": 14
            },
            "distilbert": {
                "name": "distilbert-base-uncased",
                "class": AutoModelForSequenceClassification,
                "type": "sequence_classification",
                "size": 66
            },
            "tinybert": {
                "name": "huawei-noah/TinyBERT_General_4L_312D",
                "class": AutoModelForSequenceClassification,
                "type": "sequence_classification",
                "size": 14.5
            },
            "albert-base": {
                "name": "albert-base-v2",
                "class": AutoModelForSequenceClassification,
                "type": "sequence_classification",
                "size": 12
            }
        }
    
    def evaluate_model(self, 
                      model_key: str, 
                      eval_dataset: torch.utils.data.Dataset,
                      safety_checker) -> ModelPerformance:
        """Evaluate a specific model's performance."""
        model_info = self.models[model_key]
        model = self._load_model(model_key)
        tokenizer = AutoTokenizer.from_pretrained(model_info["name"])
        
        # Prepare evaluation dataloader
        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=16,
            shuffle=False
        )
        
        # Measure performance metrics
        latency = self.profiler.measure_inference_time(
            model,
            next(iter(dataloader))
        )
        
        memory_usage = self.profiler.measure_memory_usage(model)
        throughput = self.profiler.measure_throughput(model, dataloader)
        
        # Evaluate accuracy and safety
        accuracy, safety_score = self._evaluate_accuracy_and_safety(
            model,
            dataloader,
            safety_checker
        )
        
        # Evaluate automotive-specific metrics
        automotive_score = self._evaluate_automotive_metrics(
            model,
            dataloader
        )
        
        return ModelPerformance(
            accuracy=accuracy,
            latency=latency,
            memory_usage=memory_usage,
            throughput=throughput,
            safety_score=safety_score,
            automotive_specific_score=automotive_score
        )
    
    def compare_models(self, 
                      eval_dataset: torch.utils.data.Dataset,
                      safety_checker) -> Dict[str, ModelPerformance]:
        """Compare all models and return their performance metrics."""
        results = {}
        
        for model_key in self.models:
            self.logger.info(f"Evaluating model: {model_key}")
            try:
                performance = self.evaluate_model(
                    model_key,
                    eval_dataset,
                    safety_checker
                )
                results[model_key] = performance
            except Exception as e:
                self.logger.error(f"Error evaluating {model_key}: {str(e)}")
                continue
        
        return results
    
    def _load_model(self, model_key: str) -> PreTrainedModel:
        """Load model with appropriate configuration."""
        model_info = self.models[model_key]
        model = model_info["class"].from_pretrained(
            model_info["name"],
            num_labels=len(self._get_command_labels())
            if model_info["type"] == "sequence_classification"
            else None
        )
        return model.to(self.device)
    
    def _get_command_labels(self) -> List[str]:
        """Get list of possible command labels."""
        # Implement based on your command hierarchy
        return [
            "climate_control",
            "navigation",
            "vehicle_control",
            "media_control",
            "system_settings"
        ]
    
    def _evaluate_accuracy_and_safety(self,
                                    model: PreTrainedModel,
                                    dataloader: torch.utils.data.DataLoader,
                                    safety_checker) -> Tuple[float, float]:
        """Evaluate model accuracy and safety compliance."""
        model.eval()
        correct = 0
        total = 0
        safety_violations = 0
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                # Accuracy
                correct += (predictions == batch["labels"]).sum().item()
                total += batch["labels"].size(0)
                
                # Safety
                for pred in predictions:
                    command = self._convert_prediction_to_command(pred)
                    if not safety_checker.check_command_safety(command):
                        safety_violations += 1
        
        accuracy = correct / total
        safety_score = 1 - (safety_violations / total)
        
        return accuracy, safety_score
    
    def _evaluate_automotive_metrics(self,
                                   model: PreTrainedModel,
                                   dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate automotive-specific performance metrics."""
        # Implement automotive-specific evaluation logic
        # This could include:
        # - Command understanding accuracy
        # - Context sensitivity
        # - Parameter handling
        # - Emergency command handling
        return 0.0  # Placeholder
    
    def _convert_prediction_to_command(self, prediction: torch.Tensor) -> str:
        """Convert model prediction to command string."""
        # Get the prediction as a Python value
        if isinstance(prediction, torch.Tensor):
            pred_idx = prediction.item()
        else:
            pred_idx = prediction
            
        # Map prediction index to command category
        command_categories = self._get_command_labels()
        category_idx = pred_idx % len(command_categories)
        category = command_categories[category_idx]
        
        # Define command templates for each category
        templates = {
            "climate_control": [
                "set temperature to {temp} degrees",
                "adjust fan speed to {speed}",
                "turn {state} air conditioning",
                "set {location} temperature to {temp} degrees"
            ],
            "navigation": [
                "navigate to {destination}",
                "find route to {destination} via {preference}",
                "avoid {obstacle} on route",
                "set destination to {destination}"
            ],
            "vehicle_control": [
                "set cruise control to {speed} km/h",
                "change to {mode} driving mode",
                "{action} lane keeping assist",
                "adjust {system} sensitivity to {level}"
            ],
            "media_control": [
                "play {content}",
                "set volume to {level}",
                "switch to {source}",
                "{action} {content} playback"
            ],
            "system_settings": [
                "update {setting}",
                "check {component} status",
                "configure {option}",
                "run {diagnostic_type} diagnostics"
            ]
        }
        
        # Select a template for the category
        category_templates = templates.get(category, ["perform {action}"])
        template_idx = pred_idx % len(category_templates)
        template = category_templates[template_idx]
        
        # Fill in template parameters with appropriate values
        filled_template = template
        
        # Temperature values (16-30)
        if "{temp}" in template:
            filled_template = filled_template.replace("{temp}", str(16 + (pred_idx % 15)))
        
        # Fan speeds
        if "{speed}" in template:
            speeds = ["low", "medium", "high", "auto"]
            filled_template = filled_template.replace("{speed}", speeds[pred_idx % len(speeds)])
        
        # On/off states
        if "{state}" in template:
            states = ["on", "off"]
            filled_template = filled_template.replace("{state}", states[pred_idx % len(states)])
        
        # Location zones
        if "{location}" in template:
            locations = ["driver", "passenger", "rear", "all"]
            filled_template = filled_template.replace("{location}", locations[pred_idx % len(locations)])
        
        # Destinations
        if "{destination}" in template:
            destinations = ["home", "work", "restaurant", "gas station", "airport"]
            filled_template = filled_template.replace("{destination}", destinations[pred_idx % len(destinations)])
        
        # Route preferences
        if "{preference}" in template:
            preferences = ["fastest", "eco", "scenic", "toll-free"]
            filled_template = filled_template.replace("{preference}", preferences[pred_idx % len(preferences)])
        
        # Obstacles to avoid
        if "{obstacle}" in template:
            obstacles = ["tolls", "highways", "traffic", "ferries"]
            filled_template = filled_template.replace("{obstacle}", obstacles[pred_idx % len(obstacles)])
        
        # Drive modes
        if "{mode}" in template:
            modes = ["eco", "comfort", "sport", "auto"]
            filled_template = filled_template.replace("{mode}", modes[pred_idx % len(modes)])
        
        # Actions
        if "{action}" in template:
            actions = ["activate", "deactivate", "enable", "disable", "start", "stop"]
            filled_template = filled_template.replace("{action}", actions[pred_idx % len(actions)])
        
        # Vehicle systems
        if "{system}" in template:
            systems = ["parking assist", "lane keeping", "cruise control", "auto-pilot"]
            filled_template = filled_template.replace("{system}", systems[pred_idx % len(systems)])
        
        # Media content
        if "{content}" in template:
            content = ["music", "radio", "podcast", "audiobook", "news"]
            filled_template = filled_template.replace("{content}", content[pred_idx % len(content)])
        
        # Volume/sensitivity levels
        if "{level}" in template:
            levels = ["low", "medium", "high", "maximum"]
            filled_template = filled_template.replace("{level}", levels[pred_idx % len(levels)])
        
        # Media sources
        if "{source}" in template:
            sources = ["radio", "bluetooth", "usb", "streaming"]
            filled_template = filled_template.replace("{source}", sources[pred_idx % len(sources)])
        
        # System settings
        if "{setting}" in template:
            settings = ["display", "sound", "connectivity", "notifications"]
            filled_template = filled_template.replace("{setting}", settings[pred_idx % len(settings)])
        
        # System components
        if "{component}" in template:
            components = ["battery", "sensors", "system", "connectivity"]
            filled_template = filled_template.replace("{component}", components[pred_idx % len(components)])
        
        # System options
        if "{option}" in template:
            options = ["brightness", "language", "theme", "notifications"]
            filled_template = filled_template.replace("{option}", options[pred_idx % len(options)])
        
        # Diagnostic types
        if "{diagnostic_type}" in template:
            diagnostic_types = ["quick", "full", "targeted"]
            filled_template = filled_template.replace("{diagnostic_type}", diagnostic_types[pred_idx % len(diagnostic_types)])
        
        return filled_template    

    def generate_comparison_report(self, 
                                 results: Dict[str, ModelPerformance],
                                 output_path: str):
        """Generate detailed comparison report."""
        report = {
            "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "model_comparisons": {}
        }
        
        for model_key, performance in results.items():
            report["model_comparisons"][model_key] = {
                "specs": vars(self.models[model_key]),
                "performance": vars(performance)
            }
        
        output_file = Path(output_path) / "model_comparison_report.json"
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Comparison report saved to {output_file}")