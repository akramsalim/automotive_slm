import sys
import os

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


import argparse
import logging
from pathlib import Path
import yaml
from data.command_generator import AutomotiveCommandGenerator
from data.data_processor import DataProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to config file")
    parser.add_argument("--output-dir", type=str, required=True,
                      help="Output directory for generated data")
    parser.add_argument("--num-samples", type=int, default=10000,
                      help="Number of samples to generate")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup output directory and logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "generate_data.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        logger.info("Initializing command generator...")
        generator = AutomotiveCommandGenerator()
        
        # Generate commands
        logger.info(f"Generating {args.num_samples} commands...")
        default_distribution = {
            "climate": 0.2,
            "navigation": 0.3,
            "vehicle_control": 0.2,
            "media": 0.2,
            "system": 0.1
        }

        # Try to get distribution from config, use default if not found
        distribution = None
        try:
            distribution = config["data"]["command_distribution"]
        except (KeyError, TypeError):
            logger.info("No command distribution found in config, using default distribution")
            distribution = default_distribution

        commands = generator.generate_commands(
            num_samples=args.num_samples,
            distribution=distribution)          
        # Process and save data
        logger.info("Processing and saving data...")
        data_processor = DataProcessor(config["data"])
        
        train_data, val_data, test_data = data_processor.split_data(
            commands,
            train_ratio=config["data"]["train_split"],
            val_ratio=config["data"]["val_split"],
            test_ratio=config["data"]["test_split"]
        )
        
        # Save splits
        data_processor.save_splits(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            output_dir=output_dir
        )
        
        logger.info(f"Data generation completed. Files saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Data generation failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()