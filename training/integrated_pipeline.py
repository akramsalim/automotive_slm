# training/integrated_pipeline.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler, AutoModelForCausalLM
import wandb
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
from tqdm import tqdm
import os
import numpy as np
from datetime import datetime

from data.automotive_dataset import AutomotiveDataset
from data.command_generator import AutomotiveCommandGenerator
from evaluation.metrics import AutomotiveEvaluator, EvaluationConfig
from losses.automotive_losses import AutomotiveLossFunction
from peft import get_peft_model, LoraConfig, TaskType
from safety.safety_checker import AutomotiveSafetyChecker, SafetyContext

class IntegratedAutomotiveTrainer:
    def __init__(
        self,
        model_name: str,
        train_dataset: AutomotiveDataset,
        val_dataset: AutomotiveDataset,
        command_generator: AutomotiveCommandGenerator,
        safety_checker: AutomotiveSafetyChecker,
        config: Dict,
    ):
        self.model_name = model_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.command_generator = command_generator
        self.safety_checker = safety_checker
        self.config = config
        
        # Initialize components
        self._init_model()
        self._init_tokenizer()
        self._init_dataloaders()
        self._init_loss_functions()
        self._init_optimizer()
        self._init_evaluator()
        self.logger = self._init_logger()
        
        # Initialize training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_eval_metric = float('inf')
        
    def _init_model(self):
        """Initialize model with LoRA configuration."""
        # Base model initialization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config["lora"]["r"],  # Use r instead of rank
            lora_alpha=self.config["lora"]["alpha"],
            target_modules=self.config["lora"]["target_modules"],
            lora_dropout=self.config["lora"]["dropout"],
            bias=self.config["lora"]["bias"],
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Define model as a module dictionary for organization
        self.model = nn.ModuleDict({
            'base_model': self.model,
            'safety_checker': self.safety_checker
        })
        
    def _init_tokenizer(self):
        """Initialize tokenizer with automotive-specific tokens."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add special tokens for automotive domain
        new_tokens = [
            "<VEHICLE>", "<COMMAND>", "<SAFETY>", "<ERROR>",
            "<NAVIGATION>", "<CLIMATE>", "<MEDIA>"
        ]
        
        # Check if tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        num_added = self.tokenizer.add_tokens(new_tokens)
        self.model.base_model.resize_token_embeddings(len(self.tokenizer))
        self.logger.info(f"Added {num_added} new tokens to the tokenizer")
        
    def _init_dataloaders(self):
        """Initialize training and validation dataloaders."""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["training"].get("num_workers", 4)
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["training"].get("num_workers", 4)
        )
        
    def _init_loss_functions(self):
        """Initialize combined loss functions."""
        command_hierarchy = self._load_command_hierarchy()
        self.loss_fn = AutomotiveLossFunction(
            command_hierarchy=command_hierarchy,
            safety_weight=self.config["loss"]["safety_weight"],
            context_weight=self.config["loss"]["context_weight"]
        )
        
    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        # Only optimize LoRA parameters for efficiency
        trainable_params = [p for p in self.model.base_model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )
        
        # Learning rate scheduler
        total_steps = len(self.train_loader) * self.config["training"]["num_epochs"]
        warmup_steps = int(total_steps * self.config["training"].get("warmup_ratio", 0.1))
        
        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
    def _init_evaluator(self):
        """Initialize evaluation framework."""
        self.evaluator = AutomotiveEvaluator(EvaluationConfig(
            latency_threshold=self.config["evaluation"]["latency_threshold"],
            memory_threshold=self.config["evaluation"]["memory_threshold"],
            accuracy_threshold=self.config["evaluation"]["accuracy_threshold"],
            safety_threshold=self.config["evaluation"]["safety_threshold"]
        ))
        
    def _init_logger(self):
        """Initialize logging and wandb."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Add console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(console_handler)
            
            # Add file handler
            log_dir = Path(self.config["logging"].get("save_dir", "logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(file_handler)
        
        # Initialize wandb if enabled
        if self.config["logging"].get("use_wandb", False):
            wandb.init(
                project=self.config["logging"]["project_name"],
                config=self.config
            )
            
        return logger
        
    def _load_command_hierarchy(self) -> Dict[str, List[str]]:
        """Load command hierarchy from config."""
        if "data" in self.config and "hierarchy_path" in self.config["data"]:
            hierarchy_path = Path(self.config["data"]["hierarchy_path"])
            try:
                with open(hierarchy_path) as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load command hierarchy: {e}")
                
        # Fallback to default hierarchy
        return {
            "climate_control": ["set_temperature", "adjust_fan", "toggle_ac"],
            "navigation": ["set_destination", "find_route", "show_traffic"],
            "vehicle_control": ["cruise_control", "drive_mode", "parking_assist"],
            "media_control": ["play_media", "adjust_volume", "change_source"],
            "system_control": ["adjust_display", "update_settings", "pair_device"]
        }
            
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config["training"]["num_epochs"]):
            self.current_epoch = epoch
            self.logger.info(f"Starting epoch {epoch + 1}")
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = self._validate()
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics)
            
            # Save checkpoint
            if val_metrics["eval_loss"] < self.best_eval_metric:
                self.best_eval_metric = val_metrics["eval_loss"]
                self._save_checkpoint("best_model")
                
            # Regular checkpoint
            if (epoch + 1) % self.config["training"]["save_every"] == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch+1}")
                
    def _train_epoch(self) -> Dict[str, float]:
        """Training logic for one epoch."""
        self.model.train()
        total_loss = 0
        metrics = {
            "safety_violations": 0,
            "accuracy": 0,
            "latency": []
        }
        
        progress_bar = tqdm(self.train_loader, desc=f"Training epoch {self.current_epoch + 1}")
        
        for batch in progress_bar:
            # Move batch to device
            device = next(self.model.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in batch.items()}
            
            # Generate synthetic context features
            context_features = self._generate_context_features(batch)
            
            # Forward pass
            outputs = self.model.base_model(**batch)
            
            # Safety check
            safety_scores = self._compute_safety_scores(outputs.logits, batch)
            
            # Create category labels if not in batch
            if "category_labels" not in batch:
                batch["category_labels"] = self._derive_category_labels(batch)
            
            # Compute loss
            losses = self.loss_fn(
                logits=outputs.logits,
                labels=batch["labels"],
                safety_scores=safety_scores,
                context_features=context_features,
                category_labels=batch["category_labels"]
            )
            
            # Backward pass
            losses["total_loss"].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config["training"]["max_grad_norm"]
            )
            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += losses["total_loss"].item()
            metrics["safety_violations"] += (safety_scores < 0.5).sum().item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": losses["total_loss"].item(),
                "safety_violations": metrics["safety_violations"]
            })
            
        return {
            "train_loss": total_loss / len(self.train_loader),
            "safety_violations": metrics["safety_violations"],
            "learning_rate": self.lr_scheduler.get_last_lr()[0]
        }
        
    def _validate(self) -> Dict[str, float]:
        """Validation logic."""
        self.model.eval()
        total_loss = 0
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move batch to device
                device = next(self.model.parameters()).device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                       for k, v in batch.items()}
                
                # Generate context features
                context_features = self._generate_context_features(batch)
                
                # Forward pass
                outputs = self.model.base_model(**batch)
                
                # Safety check
                safety_scores = self._compute_safety_scores(outputs.logits, batch)
                
                # Create category labels if not in batch
                if "category_labels" not in batch:
                    batch["category_labels"] = self._derive_category_labels(batch)
                
                # Compute metrics
                batch_context = self._create_safety_context()
                metrics = self.evaluator.evaluate_model(
                    self.model.base_model,
                    batch,
                    lambda: batch_context
                )
                all_metrics.append(metrics)
                
                # Compute loss
                losses = self.loss_fn(
                    logits=outputs.logits,
                    labels=batch["labels"],
                    safety_scores=safety_scores,
                    context_features=context_features,
                    category_labels=batch["category_labels"]
                )
                total_loss += losses["total_loss"].item()
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        aggregated_metrics["eval_loss"] = total_loss / len(self.val_loader)
        
        return aggregated_metrics
    
    def _generate_context_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate context features for the batch."""
        batch_size = batch["input_ids"].shape[0]
        device = batch["input_ids"].device
        context_dim = 768  # Common embedding dimension for transformer models
        
        # Create context features tensor with appropriate dimensions
        context_features = torch.zeros((batch_size, context_dim), device=device)
        
        # For each item in the batch, extract context information
        for i in range(batch_size):
            if "category_ids" in batch:
                category_id = batch["category_ids"][i].item()
                # Set specific features based on category
                start_idx = category_id * 100  # Use different segments for each category
                end_idx = min(start_idx + 100, context_dim)
                context_features[i, start_idx:end_idx] = 1.0
            
            # Incorporate safety-related features if available
            if "safety_score" in batch:
                safety_score = batch["safety_score"][i].item()
                # Use last segment for safety information
                start_idx = max(0, context_dim - 100)
                context_features[i, start_idx:] = safety_score
            
            # Additional context information (vehicle state, etc.)
            # Could be incorporated here from batch data
        
        return context_features
    
    def _derive_category_labels(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Derive category labels from command labels if not provided."""
        if "labels" not in batch:
            return torch.zeros(batch["input_ids"].shape[0], dtype=torch.long, device=batch["input_ids"].device)
            
        # Map from command ID to category ID based on hierarchy
        # For simplicity, we'll use label // 5 as the category (5 commands per category)
        category_labels = batch["labels"] // 5
        return category_labels
    
    def _compute_safety_scores(self, 
                         logits: torch.Tensor, 
                         batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute safety scores for predictions."""
        batch_size = logits.shape[0]
        device = logits.device
        predictions = torch.argmax(logits, dim=-1)
        
        # Create safety context
        safety_context = self._create_safety_context()
        
        # Update context with batch-specific information if available
        if "vehicle_speed" in batch:
            safety_context.speed = batch["vehicle_speed"].mean().item()
        
        if "time_of_day" in batch:
            # Map time of day index to string value
            time_mapping = {0: "day", 1: "night"}
            time_idx = batch["time_of_day"].mean().round().int().item()
            safety_context.time_of_day = time_mapping.get(time_idx, "day")
        
        if "weather_condition" in batch:
            # Map weather index to string value
            weather_mapping = {0: "clear", 1: "rain", 2: "snow", 3: "fog"}
            weather_idx = batch["weather_condition"].mean().round().int().item()
            safety_context.weather = weather_mapping.get(weather_idx, "clear")
        
        # Compute safety scores for each prediction
        safety_scores = torch.ones(batch_size, device=device)
        
        for i in range(batch_size):
            # Convert prediction to command string
            command = self._convert_prediction_to_command(predictions[i])
            
            # Check command safety
            is_safe, violation = self.safety_checker.check_command_safety(command, safety_context)
            
            # Assign safety score
            safety_scores[i] = 1.0 if is_safe else 0.0
            
            # If there's a recoverable violation, assign partial score
            if not is_safe and violation and hasattr(violation, 'severity'):
                if violation.severity == "warning":
                    safety_scores[i] = 0.5  # Partial score for recoverable issues
        
        return safety_scores

    def _create_safety_context(self) -> SafetyContext:
        """Create a safety context with appropriate values."""
        context = SafetyContext()
        
        # Set default values
        context.speed = 50.0  # 50 km/h
        context.location = {"latitude": 37.7749, "longitude": -122.4194}  # Example location
        context.weather = "clear"
        context.time_of_day = "day"
        context.road_type = "normal"
        
        # Set vehicle state
        context.vehicle_state = {
            "engine": "on",
            "doors": "closed",
            "safety_systems": "active"
        }
        
        return context
    
    def _convert_prediction_to_command(self, prediction: torch.Tensor) -> str:
        """Convert model prediction to command string."""
        # Get the class index from the prediction
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.item()
        
        # Command templates for conversion
        command_types = [
            "set_temperature to 22 degrees",
            "navigate_to Central Park",
            "activate_cruise_control at 80 km/h",
            "play_media Jazz Playlist",
            "adjust_volume to 60 percent",
            "set_drive_mode to eco"
        ]
        
        # Map prediction index to command type
        command_idx = prediction % len(command_types)
        command = command_types[command_idx]
        
        return command
    
    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics from multiple validation steps."""
        if not metrics_list:
            return {}
            
        aggregated = {}
        
        # Identify all metric keys from first entry
        for category in metrics_list[0]:
            if isinstance(metrics_list[0][category], dict):
                aggregated[category] = {}
                for metric_name, value in metrics_list[0][category].items():
                    if isinstance(value, (int, float)):
                        # Average numeric metrics
                        values = [m[category].get(metric_name, 0) for m in metrics_list]
                        aggregated[category][metric_name] = sum(values) / len(values)
        
        return aggregated
    
    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics to console and wandb."""
        # Flatten nested metrics for easier logging
        flat_train_metrics = {"train_" + k: v for k, v in train_metrics.items()}
        
        flat_val_metrics = {}
        for category, metrics in val_metrics.items():
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    flat_val_metrics[f"val_{category}_{k}"] = v
            else:
                flat_val_metrics[f"val_{category}"] = metrics
        
        # Combine all metrics
        all_metrics = {**flat_train_metrics, **flat_val_metrics}
        
        # Console logging
        self.logger.info(f"Epoch {self.current_epoch + 1} metrics:")
        for name, value in all_metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"{name}: {value:.4f}")
            
        # WandB logging
        if self.config["logging"].get("use_wandb", False):
            wandb.log(all_metrics, step=self.global_step)
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        # Create checkpoint directory
        checkpoint_dir = Path(self.config["training"].get("checkpoint_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = checkpoint_dir / f"{name}_model"
        self.model.base_model.save_pretrained(model_path)
        
        # Save tokenizer
        tokenizer_path = checkpoint_dir / f"{name}_tokenizer"
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # Save training state
        state_path = checkpoint_dir / f"{name}_training_state.pt"
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_eval_metric': self.best_eval_metric,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict()
        }, state_path)
        
        self.logger.info(f"Saved checkpoint: {name}")

# Example default config for reference
default_training_config = {
    "model_name": "microsoft/phi-2",
    "lora": {
        "r": 8,
        "alpha": 16,
        "target_modules": ["query_key_value", "dense"],
        "dropout": 0.1,
        "bias": "none"
    },
    "training": {
        "batch_size": 16,
        "num_workers": 4,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "num_epochs": 10,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        "save_every": 1,
        "checkpoint_dir": "checkpoints/"
    },
    "loss": {
        "safety_weight": 1.0,
        "context_weight": 0.5
    },
    "evaluation": {
        "latency_threshold": 0.1,
        "memory_threshold": 512,
        "accuracy_threshold": 0.95,
        "safety_threshold": 0.99
    },
    "logging": {
        "use_wandb": True,
        "project_name": "automotive-slm",
        "save_dir": "logs"
    },
    "data": {
        "hierarchy_path": "config/command_hierarchy.json"
    }
}