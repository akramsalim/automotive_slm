# losses/automotive_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import math

class SafetyAwareLoss(nn.Module):
    def __init__(self, safety_weight: float = 1.0):
        super().__init__()
        self.safety_weight = safety_weight
        self.base_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, 
                logits: torch.Tensor, 
                labels: torch.Tensor, 
                safety_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with safety penalty.
        
        Args:
            logits: Model predictions (batch_size, vocab_size)
            labels: Ground truth labels (batch_size)
            safety_scores: Safety scores for each prediction (batch_size)
        """
        # Compute base loss
        base_loss = self.base_loss(logits, labels)
        
        # Apply safety weighting
        safety_penalty = (1 - safety_scores) * self.safety_weight
        
        return (base_loss * (1 + safety_penalty)).mean()

class CommandContextLoss(nn.Module):
    def __init__(self, context_weight: float = 0.5):
        super().__init__()
        self.context_weight = context_weight
        self.base_loss = nn.CrossEntropyLoss()
    
    def forward(self, 
                logits: torch.Tensor, 
                labels: torch.Tensor, 
                context_features: torch.Tensor) -> torch.Tensor:
        """
        Compute loss considering command context.
        
        Args:
            logits: Model predictions
            labels: Ground truth labels
            context_features: Encoded context information
        """
        # Base command loss
        command_loss = self.base_loss(logits, labels)
        
        # Context consistency loss
        context_loss = self._compute_context_consistency(logits, context_features)
        
        return command_loss + self.context_weight * context_loss
    
    def _compute_context_consistency(self, 
                                   logits: torch.Tensor, 
                                   context_features: torch.Tensor) -> torch.Tensor:
        """Compute consistency between predictions and context."""
        # Project logits and context to same space
        logits_proj = self._project_to_context_space(logits)
        
        # Compute consistency loss (e.g., cosine similarity)
        consistency = 1 - F.cosine_similarity(logits_proj, context_features).mean()
        
        return consistency
    
    def _project_to_context_space(self, logits: torch.Tensor) -> torch.Tensor:
        """Project logits to context feature space."""
        # Implement projection logic
        raise NotImplementedError

class HierarchicalCommandLoss(nn.Module):
    def __init__(self, command_hierarchy: Dict[str, List[str]], hierarchy_weights: Optional[Dict[str, float]] = None):
        """
        Initialize hierarchical command loss.
        
        Args:
            command_hierarchy: Dictionary mapping command categories to lists of commands
            hierarchy_weights: Optional weights for different hierarchy levels
        """
        super().__init__()
        self.command_hierarchy = command_hierarchy
        self.hierarchy_weights = hierarchy_weights or {
            "category": 0.4,
            "command": 0.6
        }
        self.base_loss = nn.CrossEntropyLoss()
        
    def forward(self, 
                logits: torch.Tensor, 
                labels: torch.Tensor, 
                category_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute hierarchical loss for commands.
        
        Args:
            logits: Model predictions
            labels: Ground truth command labels
            category_labels: Ground truth category labels
        """
        # Command-level loss
        command_loss = self.base_loss(logits, labels)
        
        # Category-level loss
        category_logits = self._aggregate_category_logits(logits)
        category_loss = self.base_loss(category_logits, category_labels)
        
        # Combine losses
        total_loss = (self.hierarchy_weights["command"] * command_loss + 
                     self.hierarchy_weights["category"] * category_loss)
        
        return total_loss
    
    def _aggregate_category_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Aggregate command logits to category level."""
        category_logits = []
        for category, commands in self.command_hierarchy.items():
            # Sum logits for all commands in category
            category_score = torch.sum(logits[:, commands], dim=1)
            category_logits.append(category_score)
        
        return torch.stack(category_logits, dim=1)

class AutomotiveLossFunction(nn.Module):
    def __init__(self, 
                 command_hierarchy: Dict[str, List[str]],
                 safety_weight: float = 1.0,
                 context_weight: float = 0.5):
        """
        Combined loss function for automotive command generation.
        
        Args:
            command_hierarchy: Dictionary mapping command categories to commands
            safety_weight: Weight for safety-aware loss
            context_weight: Weight for context-aware loss
        """
        super().__init__()
        self.safety_loss = SafetyAwareLoss(safety_weight)
        self.context_loss = CommandContextLoss(context_weight)
        self.hierarchical_loss = HierarchicalCommandLoss(command_hierarchy)
        
    def forward(self,
                logits: torch.Tensor,
                labels: torch.Tensor,
                safety_scores: torch.Tensor,
                context_features: torch.Tensor,
                category_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss for automotive command generation.
        
        Args:
            logits: Model predictions
            labels: Ground truth labels
            safety_scores: Safety scores for predictions
            context_features: Encoded context information
            category_labels: Ground truth category labels
            
        Returns:
            Dictionary containing total loss and individual loss components
        """
        # Compute individual losses
        safety_loss = self.safety_loss(logits, labels, safety_scores)
        context_loss = self.context_loss(logits, labels, context_features)
        hierarchical_loss = self.hierarchical_loss(logits, labels, category_labels)
        
        # Combine losses
        total_loss = safety_loss + context_loss + hierarchical_loss
        
        return {
            "total_loss": total_loss,
            "safety_loss": safety_loss,
            "context_loss": context_loss,
            "hierarchical_loss": hierarchical_loss
        }

class LateralityAwareLoss(nn.Module):
    """Loss function for commands with lateral movement considerations."""
    
    def __init__(self, laterality_weight: float = 0.3):
        super().__init__()
        self.laterality_weight = laterality_weight
        self.base_loss = nn.CrossEntropyLoss()
        
    def forward(self, 
                logits: torch.Tensor, 
                labels: torch.Tensor,
                current_lateral_state: torch.Tensor) -> torch.Tensor:
        """
        Compute loss considering lateral movement safety.
        
        Args:
            logits: Model predictions
            labels: Ground truth labels
            current_lateral_state: Current lateral movement state
        """
        # Base command loss
        base_loss = self.base_loss(logits, labels)
        
        # Compute laterality penalty
        laterality_penalty = self._compute_laterality_penalty(
            logits, current_lateral_state
        )
        
        return base_loss + self.laterality_weight * laterality_penalty
    
    def _compute_laterality_penalty(self, 
                                  logits: torch.Tensor,
                                  current_lateral_state: torch.Tensor) -> torch.Tensor:
        """Compute penalty for unsafe lateral movements."""
        # Convert logits to movement predictions
        movement_probs = F.softmax(logits, dim=-1)
        
        # Calculate penalty based on current state and predicted movements
        penalty = torch.abs(movement_probs - current_lateral_state).mean()
        
        return penalty

# Example usage of the combined loss function
def train_step(model: nn.Module,
               batch: Dict[str, torch.Tensor],
               loss_fn: AutomotiveLossFunction,
               optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    """
    Single training step with combined loss function.
    
    Args:
        model: The model to train
        batch: Batch of training data
        loss_fn: Combined loss function
        optimizer: Optimizer
    
    Returns:
        Dictionary of loss values
    """
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(batch["input_ids"], batch["attention_mask"])
    
    # Compute losses
    losses = loss_fn(
        logits=logits,
        labels=batch["labels"],
        safety_scores=batch["safety_scores"],
        context_features=batch["context_features"],
        category_labels=batch["category_labels"]
    )
    
    # Backward pass
    losses["total_loss"].backward()
    optimizer.step()
    
    return {k: v.item() for k, v in losses.items()}