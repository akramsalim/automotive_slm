from typing import Dict, List, Tuple, Optional
import torch
from pathlib import Path
import json
import logging
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, config: Dict):
        """
        Initialize data processor.
        
        Args:
            config: Data processing configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def process_data(self, 
                    data: List[Dict],
                    train_ratio: float = 0.8,
                    val_ratio: float = 0.1,
                    test_ratio: float = 0.1,
                    shuffle: bool = True) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Process and split data into train, validation, and test sets.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, \
            "Split ratios must sum to 1"

        # First split: train and temp (val + test)
        train_data, temp_data = train_test_split(
            data,
            train_size=train_ratio,
            shuffle=shuffle,
            random_state=self.config.get("seed", 42)
        )

        # Second split: val and test from temp
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data,
            train_size=val_ratio_adjusted,
            shuffle=shuffle,
            random_state=self.config.get("seed", 42)
        )

        self.logger.info(f"Data split: {len(train_data)} train, "
                        f"{len(val_data)} validation, {len(test_data)} test")

        return train_data, val_data, test_data

    def save_splits(self,
                   train_data: List[Dict],
                   val_data: List[Dict],
                   test_data: List[Dict],
                   output_dir: Path):
        """Save data splits to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each split
        splits = {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }

        for split_name, split_data in splits.items():
            output_path = output_dir / f"{split_name}.json"
            with open(output_path, 'w') as f:
                json.dump(split_data, f, indent=2)

            self.logger.info(f"Saved {split_name} split to {output_path}")