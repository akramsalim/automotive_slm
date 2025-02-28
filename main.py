# main.py
import torch
import yaml
import json
from pathlib import Path
import logging
import argparse
from datetime import datetime

from data.command_generator import AutomotiveCommandGenerator
from models.automotive_adapter import AutomotiveSafetyChecker
from training.integrated_pipeline import IntegratedAutomotiveTrainer
from data.automotive_dataset import AutomotiveDataset

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

def load_configs():
    """Load configuration files."""
    with open("config/training_config.yaml") as f:
        training_config = yaml.safe_load(f)
    
    with open("config/command_hierarchy.json") as f:
        command_hierarchy = json.load(f)
    
    # Ensure required structure exists
    if "model" not in training_config:
        training_config["model"] = {}
    
    if "base_model_name" not in training_config["model"]:
        # Set a default model name based on available models
        if "phi-2" in training_config.get("model", {}):
            training_config["model"]["base_model_name"] = "microsoft/phi-2"
        else:
            training_config["model"]["base_model_name"] = "distilbert-base-uncased"
    
    return training_config, command_hierarchy


def generate_datasets(config, command_hierarchy):
    """Generate training and validation datasets."""
    # Initialize command generator
    command_generator = AutomotiveCommandGenerator()
    
    # Generate synthetic data
    train_data = command_generator.generate_synthetic_dataset(
        num_samples=int(config["data"]["max_samples"] * config["data"]["train_split"]),
        distribution=config["data"]["command_distribution"]
    )
    
    val_data = command_generator.generate_synthetic_dataset(
        num_samples=int(config["data"]["max_samples"] * config["data"]["val_split"]),
        distribution=config["data"]["command_distribution"]
    )
    
    # Create datasets
    train_dataset = AutomotiveDataset(train_data, command_hierarchy)
    val_dataset = AutomotiveDataset(val_data, command_hierarchy)
    
    return train_dataset, val_dataset

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Automotive SLM")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Load configurations
    training_config, command_hierarchy = load_configs()
    
    # Setup logging
    setup_logging(training_config)
    logger = logging.getLogger(__name__)
    
    if args.debug:
        logger.info("Running in debug mode")
        training_config["data"]["max_samples"] = 1000
        training_config["training"]["num_epochs"] = 1
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Generate datasets
        logger.info("Generating datasets...")
        train_dataset, val_dataset = generate_datasets(training_config, command_hierarchy)
        
        # Initialize safety checker
        safety_checker = AutomotiveSafetyChecker()
        
        # Initialize trainer
        trainer = IntegratedAutomotiveTrainer(
            model_name=training_config["model"]["base_model_name"],
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            command_generator=AutomotiveCommandGenerator(),
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