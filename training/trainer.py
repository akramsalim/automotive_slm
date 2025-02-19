import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_scheduler
import wandb
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from tqdm import tqdm
import time
from datetime import datetime

class AutomotiveTrainer:
    """Trainer class for automotive models."""
    
    def __init__(self,
                 model: nn.Module,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 config: Dict[str, Any],
                 device: str = "cuda"):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optimizer
            config: Training configuration
            device: Device to use
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # Initialize components
        self.lr_scheduler = self._init_scheduler()
        self.logger = self._init_logger()
        self._init_wandb()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def _init_scheduler(self):
        """Initialize learning rate scheduler."""
        return get_scheduler(
            name=self.config["training"]["scheduler_type"],
            optimizer=self.optimizer,
            num_warmup_steps=self.config["training"]["warmup_steps"],
            num_training_steps=len(self.train_dataloader) * self.config["training"]["num_epochs"]
        )
    
    def _init_logger(self):
        """Initialize logger."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        if self.config["logging"]["use_wandb"]:
            wandb.init(
                project=self.config["logging"]["project_name"],
                config=self.config,
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config["training"]["num_epochs"]):
            self.epoch = epoch
            self.logger.info(f"Starting epoch {epoch + 1}")
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = self._validate()
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics)
            
            # Save checkpoint
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self._save_checkpoint("best_model")
            
            if (epoch + 1) % self.config["training"]["save_every"] == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch+1}")
    
    def _train_epoch(self) -> Dict[str, float]:
        """Training logic for one epoch."""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Training epoch {self.epoch + 1}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config["training"]["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["max_grad_norm"]
                )
            
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
            
            self.global_step += 1
        
        return {
            "train_loss": total_loss / len(self.train_dataloader),
            "learning_rate": self.lr_scheduler.get_last_lr()[0]
        }
    
    def _validate(self) -> Dict[str, float]:
        """Validation logic."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                
                total_loss += loss.item()
        
        return {
            "val_loss": total_loss / len(self.val_dataloader)
        }
    
    def _log_metrics(self,
                    train_metrics: Dict[str, float],
                    val_metrics: Dict[str, float]):
        """Log metrics to console and wandb."""
        metrics = {
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()}
        }
        
        # Console logging
        self.logger.info(f"Epoch {self.epoch + 1} metrics:")
        for name, value in metrics.items():
            self.logger.info(f"{name}: {value:.4f}")
        
        # WandB logging
        if self.config["logging"]["use_wandb"]:
            wandb.log(metrics, step=self.global_step)
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config["training"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config
        }
        
        checkpoint_path = checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        self.logger.info(f"Loaded checkpoint from epoch {self.epoch}")

