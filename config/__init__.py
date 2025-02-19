from pathlib import Path
from typing import Dict, Any
import yaml
import json

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_hierarchy(hierarchy_path: str) -> Dict[str, Any]:
    """Load command hierarchy JSON file."""
    with open(hierarchy_path, 'r') as f:
        return json.load(f)