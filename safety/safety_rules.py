# safety/safety_rules.py
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import json
from pathlib import Path

from safety.safety_checker import SafetyContext

@dataclass
class SafetyRule:
    """Definition of a safety rule."""
    name: str
    description: str
    severity: str
    conditions: List[str]
    parameters: Optional[Dict] = None
    
    def check(self, parameters: Dict, context: 'SafetyContext') -> bool:
        """Check if parameters and context satisfy the rule."""
        try:
            for condition in self.conditions:
                if not eval(condition, {"__builtins__": None},
                          {"parameters": parameters, "context": context}):
                    return False
            return True
        except Exception as e:
            logging.error(f"Error evaluating rule {self.name}: {str(e)}")
            return False
    
    def get_violation_description(self,
                                parameters: Dict,
                                context: 'SafetyContext') -> str:
        """Get description of rule violation."""
        return f"Violation of {self.name}: {self.description}"

class SafetyRules:
    """Container for safety rules."""
    
    def __init__(self, config: Union[str, Dict]):
        """
        Initialize safety rules.
        
        Args:
            config: Rules configuration file path or dictionary
        """
        self.rules = self._load_rules(config)
        self.command_rules_map = self._create_command_rules_map()
    
    def _load_rules(self, config: Union[str, Dict]) -> List[SafetyRule]:
        """Load safety rules from configuration."""
        if isinstance(config, str):
            with open(config) as f:
                config = json.load(f)
        
        rules = []
        for rule_config in config["rules"]:
            rules.append(SafetyRule(**rule_config))
        
        return rules
    
    def _create_command_rules_map(self) -> Dict[str, List[SafetyRule]]:
        """Create mapping from command types to applicable rules."""
        command_rules = {}
        for rule in self.rules:
            if "command_types" in rule.parameters:
                for command_type in rule.parameters["command_types"]:
                    if command_type not in command_rules:
                        command_rules[command_type] = []
                    command_rules[command_type].append(rule)
        return command_rules
    
    def get_rules_for_command(self, command_type: str) -> List[SafetyRule]:
        """Get rules applicable to a command type."""
        return self.command_rules_map.get(command_type, [])