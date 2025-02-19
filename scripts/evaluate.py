import argparse
import logging
from pathlib import Path
import json
import yaml
import torch
from evaluation.evaluator import ModelEvaluator
from models.model_factory import ModelFactory
from data.data_processor import DataProcessor
from visualization.performance_plots import PerformancePlotter

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate automotive SLM")
    parser.add_argument("--model-path", type=str, required=True,
                      help="Path to trained model")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to config file")
    parser.add_argument("--output-dir", type=str, required=True,
                      help="Output directory for results")
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
            logging.FileHandler(output_dir / "evaluate.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Load model
        logger.info("Loading model...")
        model_factory = ModelFactory()
        model = model_factory.load_model(args.model_path)
        
        # Prepare evaluation data
        logger.info("Preparing evaluation data...")
        data_processor = DataProcessor(config["data"])
        eval_dataloader = data_processor.prepare_eval_data()
        
        # Initialize evaluator
        logger.info("Initializing evaluator...")
        evaluator = ModelEvaluator(config["evaluation"])
        
        # Run evaluation
        logger.info("Running evaluation...")
        results = evaluator.evaluate(
            model=model,
            dataloader=eval_dataloader
        )
        
        # Create visualizations
        logger.info("Creating visualizations...")
        plotter = PerformancePlotter()
        plotter.plot_results(
            results=results,
            output_dir=output_dir
        )
        
        # Save results
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()