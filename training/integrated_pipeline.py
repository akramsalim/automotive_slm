# training/integrated_pipeline.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler, AutoModelForCausalLM
from data.automotive_dataset import AutomotiveDataset
from data.command_generator import AutomotiveCommandGenerator
from evaluation.metrics import AutomotiveEvaluator, EvaluationConfig
from losses.automotive_losses import AutomotiveLossFunction
from peft import get_peft_model, LoraConfig, TaskType
from safety.safety_checker import AutomotiveSafetyChecker, SafetyContext
import wandb
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
from tqdm import tqdm

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
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config["lora"]["rank"],
            lora_alpha=self.config["lora"]["alpha"],
            target_modules=self.config["lora"]["target_modules"],
            lora_dropout=self.config["lora"]["dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Add safety checker
        self.model = nn.ModuleDict({
            'base_model': self.model,
            'safety_checker': self.safety_checker
        })
        
    def _init_tokenizer(self):
        """Initialize tokenizer with automotive-specific tokens."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        new_tokens = [
            "<VEHICLE>", "<COMMAND>", "<SAFETY>", "<ERROR>",
            "<NAVIGATION>", "<CLIMATE>", "<MEDIA>"
        ]
        self.tokenizer.add_tokens(new_tokens)
        self.model.base_model.resize_token_embeddings(len(self.tokenizer))
        
    def _init_dataloaders(self):
        """Initialize training and validation dataloaders."""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["training"]["num_workers"]
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["training"]["num_workers"]
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
        # Only optimize LoRA parameters
        trainable_params = self.model.base_model.parameters()
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )
        
        # Learning rate scheduler
        num_training_steps = len(self.train_loader) * self.config["training"]["num_epochs"]
        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=self.config["training"]["warmup_steps"],
            num_training_steps=num_training_steps
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
        
        if self.config["logging"]["use_wandb"]:
            wandb.init(
                project=self.config["logging"]["project_name"],
                config=self.config
            )
            
        return logger
        
    def _load_command_hierarchy(self) -> Dict[str, List[str]]:
        """Load command hierarchy from config."""
        hierarchy_path = Path(self.config["data"]["hierarchy_path"])
        with open(hierarchy_path) as f:
            return json.load(f)
            
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
            # Generate synthetic context
            context_features = self._generate_context_features(batch)
            
            # Forward pass
            outputs = self.model.base_model(**batch)
            
            # Safety check
            safety_scores = self._compute_safety_scores(outputs.logits, batch)
            
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
                # Similar to training forward pass
                context_features = self._generate_context_features(batch)
                outputs = self.model.base_model(**batch)
                safety_scores = self._compute_safety_scores(outputs.logits, batch)
                
                # Compute metrics
                metrics = self.evaluator.evaluate_model(
                    self.model,
                    batch,
                    lambda: self._generate_context_features(batch)
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
        # Implement context feature generation logic
        raise NotImplementedError
    
    def _compute_safety_scores(self, 
                             logits: torch.Tensor, 
                             batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute safety scores for predictions."""
        predictions = torch.argmax(logits, dim=-1)
        safety_context = SafetyContext()  # Initialize with appropriate values
        
        safety_scores = []
        for pred in predictions:
            command = self.tokenizer.decode(pred)
            is_safe, _ = self.safety_checker.check_command_safety(command, safety_context)
            safety_scores.append(float(is_safe))
            
        return torch.tensor(safety_scores, device=logits.device)
    
    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics from multiple validation steps."""
        aggregated = {}
        for key in metrics_list[0].keys():
            if isinstance(metrics_list[0][key], (int, float)):
                aggregated[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
        return aggregated
    
    def _log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics to console and wandb."""
        metrics = {
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()}
        }
        
        # Console logging
        self.logger.info(f"Epoch {self.current_epoch + 1} metrics:")
        for name, value in metrics.items():
            self.logger.info(f"{name}: {value:.4f}")
            
        # WandB logging
        if self.config["logging"]["use_wandb"]:
            wandb.log(metrics, step=self.global_step)
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config["training"]["checkpoint_dir"])
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

# Example configuration
training_config = {
    "model_name": "microsoft/phi-2",
    "lora": {
        "rank": 8,
        "alpha": 16,
        "target_modules": ["query_key_value", "dense"],
        "dropout": 0.1
    },
    "training": {
        "batch_size": 16,
        "num_workers": 4,
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "num_epochs": 10,
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        "save_every": 1
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
        "project_name": "automotive-slm"
    },
    "data": {
        "hierarchy_path": "config/command_hierarchy.json"
    }
}