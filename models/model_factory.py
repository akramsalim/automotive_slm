# models/model_factory.py
from typing import Optional, Dict, Any
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoConfig,
    PreTrainedModel
)
from peft import get_peft_model, LoraConfig, TaskType
import logging
from dataclasses import dataclass

@dataclass
class ModelSpecs:
    """Specifications for model initialization."""
    name: str
    model_type: str
    max_length: int
    tokenizer_name: Optional[str] = None
    num_labels: Optional[int] = None
    special_tokens: Optional[Dict[str, str]] = None

class ModelFactory:
    """Factory class for creating and configuring models."""
    
    SUPPORTED_MODELS = {
        "phi-2": {
            "class": AutoModelForCausalLM,
            "specs": ModelSpecs(
                name="microsoft/phi-2",
                model_type="causal_lm",
                max_length=512,
                special_tokens={
                    "pad_token": "<PAD>",
                    "eos_token": "<EOS>",
                    "bos_token": "<BOS>"
                }
            )
        },
        "bert-small": {
            "class": AutoModelForSequenceClassification,
            "specs": ModelSpecs(
                name="prajjwal1/bert-small",
                model_type="sequence_classification",
                max_length=128,
                num_labels=5
            )
        },
        "distilbert": {
            "class": AutoModelForSequenceClassification,
            "specs": ModelSpecs(
                name="distilbert-base-uncased",
                model_type="sequence_classification",
                max_length=128,
                num_labels=5
            )
        },
        "tinybert": {
            "class": AutoModelForSequenceClassification,
            "specs": ModelSpecs(
                name="huawei-noah/TinyBERT_General_4L_312D",
                model_type="sequence_classification",
                max_length=128,
                num_labels=5
            )
        },
        "albert-base": {
            "class": AutoModelForSequenceClassification,
            "specs": ModelSpecs(
                name="albert-base-v2",
                model_type="sequence_classification",
                max_length=128,
                num_labels=5
            )
        }
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @classmethod
    def create_model(cls,
                    model_key: str,
                    config: Optional[Dict[str, Any]] = None,
                    use_lora: bool = True,
                    device: str = "cuda") -> PreTrainedModel:
        """
        Create and configure a model.
        
        Args:
            model_key: Key identifying the model type
            config: Additional configuration parameters
            use_lora: Whether to apply LoRA adaptation
            device: Device to place the model on
        
        Returns:
            Configured model
        """
        if model_key not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_key}")

        model_info = cls.SUPPORTED_MODELS[model_key]
        model_config = cls._prepare_config(model_info, config)
        
        # Create base model
        model = cls._create_base_model(model_info, model_config)
        
        # Apply LoRA if requested
        if use_lora:
            model = cls._apply_lora(model, model_info["specs"].model_type)
        
        return model.to(device)

    @classmethod
    def _prepare_config(cls,
                       model_info: Dict,
                       custom_config: Optional[Dict] = None) -> AutoConfig:
        """Prepare model configuration."""
        specs = model_info["specs"]
        base_config = {
            "max_length": specs.max_length,
            "num_labels": specs.num_labels
        }
        
        if custom_config:
            base_config.update(custom_config)
            
        return AutoConfig.from_pretrained(
            specs.name,
            **base_config
        )

    @classmethod
    def _create_base_model(cls,
                          model_info: Dict,
                          config: AutoConfig) -> PreTrainedModel:
        """Create base model with configuration."""
        model_class = model_info["class"]
        specs = model_info["specs"]
        
        model = model_class.from_pretrained(
            specs.name,
            config=config
        )
        
        # Add special tokens if needed
        if specs.special_tokens:
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(
                    len(specs.special_tokens) + model.config.vocab_size
                )
        
        return model

    @classmethod
    def _apply_lora(cls,
                    model: PreTrainedModel,
                    model_type: str) -> PreTrainedModel:
        """Apply LoRA adaptation to model."""
        if model_type == "causal_lm":
            target_modules = ["query_key_value", "dense"]
        else:
            target_modules = ["query", "key", "value", "dense"]
        
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM if model_type == "causal_lm"
            else TaskType.SEQ_CLS
        )
        
        return get_peft_model(model, lora_config)
