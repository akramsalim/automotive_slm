# scripts/train.py
import argparse
import logging
from pathlib import Path
import yaml
import torch
from training.trainer import AutomotiveTrainer
from models.model_factory import ModelFactory
from data.data_processor import DataProcessor
from safety.safety_checker import SafetyChecker

def parse_args():
    parser = argparse.ArgumentParser(description="Train automotive SLM")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to config file")
    parser.add_argument("--model-key", type=str, required=True,
                      help="Model key to use (phi-2, bert-small, etc.)")
    parser.add_argument("--output-dir", type=str, required=True,
                      help="Output directory")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug mode")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO if not args.debug else logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "train.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        logger.info("Initializing components...")
        model_factory = ModelFactory()
        data_processor = DataProcessor(config["data"])
        safety_checker = SafetyChecker(config["safety"])
        
        # Create model
        logger.info(f"Creating model: {args.model_key}")
        model = model_factory.create_model(
            model_key=args.model_key,
            config=config["model"]
        )
        
        # Prepare data
        logger.info("Preparing data...")
        train_dataloader, val_dataloader = data_processor.prepare_data()
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = AutomotiveTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            safety_checker=safety_checker,
            config=config,
            output_dir=args.output_dir
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