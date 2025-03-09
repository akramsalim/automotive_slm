# main.py
import torch
import yaml
import json
from pathlib import Path
import logging
import argparse
from datetime import datetime
import os

from data.command_generator import AutomotiveCommandGenerator
from data.automotive_dataset import AutomotiveDataset
from data.data_processor import DataProcessor
from safety.safety_checker import AutomotiveSafetyChecker
from models.model_factory import ModelFactory
from training.integrated_pipeline import IntegratedAutomotiveTrainer
from transformers import AutoTokenizer

def setup_logging(config):
    """Set up logging configuration."""
    log_dir = Path(config["logging"]["save_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, config["logging"]["log_level"]),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_configs(config_path):
    """Load configuration files."""
    with open(config_path) as f:
        training_config = yaml.safe_load(f)
    
    # Load command hierarchy
    hierarchy_path = Path(training_config.get("data", {}).get("hierarchy_path", "config/command_hierarchy.json"))
    
    if hierarchy_path.exists():
        with open(hierarchy_path) as f:
            command_hierarchy = json.load(f)
    else:
        logging.warning(f"Command hierarchy file not found at {hierarchy_path}. Using empty hierarchy.")
        command_hierarchy = {}
    
    # Ensure required structure exists
    if "model" not in training_config:
        training_config["model"] = {}
    
    if "model_name" not in training_config:
        # Set a default model name based on available models
        if "phi-2" in training_config.get("model", {}):
            training_config["model_name"] = "microsoft/phi-2"
        else:
            training_config["model_name"] = "distilbert-base-uncased"
    
    return training_config, command_hierarchy


def generate_synthetic_data(config, command_generator):
    """Generate synthetic training and validation datasets."""
    logging.info("Generating synthetic data...")
    
    # Get parameters from config
    num_train_samples = int(config["data"]["max_samples"] * config["data"]["train_split"])
    num_val_samples = int(config["data"]["max_samples"] * config["data"]["val_split"])
    distribution = config["data"].get("command_distribution", {})
    
    # If distribution is not specified, create an even distribution
    if not distribution:
        command_types = ["climate", "navigation", "vehicle_control", "media", "system"]
        distribution = {cmd_type: 1.0/len(command_types) for cmd_type in command_types}
    
    # Generate data
    train_data = command_generator.generate_synthetic_dataset(
        num_samples=num_train_samples,
        output_path="./data",
        distribution=distribution
    )
    
    val_data = command_generator.generate_synthetic_dataset(
        num_samples=num_val_samples,
        output_path="./data",
        distribution=distribution
    )
    
    return train_data, val_data

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Automotive SLM")
    parser.add_argument("--config", type=str, default="config/training_config.yaml", 
                      help="Path to configuration file")
    parser.add_argument("--model-key", type=str, default="phi-2",
                      help="Model key to use (phi-2, bert-small, etc.)")
    parser.add_argument("--output-dir", type=str, default="output",
                      help="Output directory for models and results")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Load configurations
    training_config, command_hierarchy = load_configs(args.config)
    
    # Setup logging
    setup_logging(training_config)
    logger = logging.getLogger(__name__)
    
    # Override config for debug mode
    if args.debug:
        logger.info("Running in debug mode")
        training_config["data"]["max_samples"] = 1000
        training_config["training"]["num_epochs"] = 1
        training_config["logging"]["use_wandb"] = False
    
    # Set output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_config["training"]["checkpoint_dir"] = str(output_dir / "checkpoints")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Initialize command generator
        logger.info("Initializing command generator...")
        command_generator = AutomotiveCommandGenerator()
        
        # Generate synthetic data
        logger.info("Generating datasets...")
        train_data, val_data = generate_synthetic_data(training_config, command_generator)
        
        # Get tokenizer based on selected model
        model_name = training_config["model"][args.model_key]["name"]
        logger.info(f"Using model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create datasets
        max_length = training_config["model"][args.model_key]["max_length"]
        train_dataset = AutomotiveDataset(train_data, tokenizer, max_length=max_length)
        val_dataset = AutomotiveDataset(val_data, tokenizer, max_length=max_length)
        
        logger.info(f"Created datasets: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
        
        # Initialize safety checker
        logger.info("Initializing safety checker...")
        safety_checker = AutomotiveSafetyChecker()
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = IntegratedAutomotiveTrainer(
            model_name=model_name,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            command_generator=command_generator,
            safety_checker=safety_checker,
            config=training_config
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()