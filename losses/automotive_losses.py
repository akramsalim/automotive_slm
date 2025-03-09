# losses/automotive_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Any
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
        # Initialize projection layer
        self.projection = nn.Linear(768, 768)  # Common embedding dimension
    
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
        # Project logits to context space
        logits_proj = self._project_to_context_space(logits)
        
        # Compute consistency loss (cosine similarity)
        if logits_proj.shape[0] != context_features.shape[0]:
            # If batch sizes don't match, resize context features
            context_features = context_features[:logits_proj.shape[0]]
        
        if logits_proj.shape[1] != context_features.shape[1]:
            # If feature dimensions don't match, pad or truncate
            if logits_proj.shape[1] < context_features.shape[1]:
                # Pad logits_proj
                padding = torch.zeros(
                    logits_proj.shape[0], 
                    context_features.shape[1] - logits_proj.shape[1], 
                    device=logits_proj.device
                )
                logits_proj = torch.cat([logits_proj, padding], dim=1)
            else:
                # Truncate logits_proj
                logits_proj = logits_proj[:, :context_features.shape[1]]
        
        # Normalize for cosine similarity
        logits_norm = F.normalize(logits_proj, p=2, dim=1)
        context_norm = F.normalize(context_features, p=2, dim=1)
        
        # Compute consistency (1 - similarity)
        consistency = 1 - F.cosine_similarity(logits_norm, context_norm).mean()
        
        return consistency
    
    def _project_to_context_space(self, logits: torch.Tensor) -> torch.Tensor:
        """Project logits to context feature space."""
        batch_size = logits.shape[0]
        
        # Take mean over vocabulary dimension to get a representation for each item
        logits_avg = logits.mean(dim=1)
        
        # Project to context space using the projection layer
        if not hasattr(self, 'projection') or self.projection.in_features != logits_avg.shape[1]:
            self.projection = nn.Linear(
                logits_avg.shape[1], 768, 
                device=logits.device
            )
        
        return self.projection(logits_avg)

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
        
        # Map commands to categories for efficient lookup
        self.command_to_category = {}
        for category, commands in command_hierarchy.items():
            for command in commands:
                self.command_to_category[command] = category
        
        # Create mapping between category names and indices
        self.category_to_idx = {category: idx for idx, category in enumerate(command_hierarchy.keys())}
        
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
        batch_size = logits.shape[0]
        num_categories = len(self.command_hierarchy)
        device = logits.device
        
        # Initialize tensor for category logits
        category_logits = torch.zeros(batch_size, num_categories, device=device)
        
        # Handle empty or malformed command hierarchy
        if not self.command_hierarchy:
            return category_logits
            
        # Map command indices to category indices
        # For each predicted command, increase the score for its category
        _, command_indices = torch.topk(logits, k=min(5, logits.shape[1]), dim=1)
        
        for batch_idx in range(batch_size):
            for command_idx in command_indices[batch_idx]:
                # Convert tensor to Python scalar
                cmd_idx = command_idx.item()
                
                # Map command index to category, handling out-of-bounds
                cmd_idx_mod = cmd_idx % sum(len(cmds) for cmds in self.command_hierarchy.values())
                
                # Find which category this command belongs to
                running_idx = 0
                category_idx = 0
                
                for cat_idx, (cat, commands) in enumerate(self.command_hierarchy.items()):
                    if running_idx <= cmd_idx_mod < running_idx + len(commands):
                        category_idx = cat_idx
                        break
                    running_idx += len(commands)
                
                # Add score to this category
                category_logits[batch_idx, category_idx] += logits[batch_idx, cmd_idx]
        
        return category_logits

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