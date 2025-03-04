# safety/safety_checker.py
import time
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from dataclasses import dataclass
import re
import json
from pathlib import Path

@dataclass
class SafetyRule:
    name: str
    conditions: List[str]
    priority: int
    recovery_action: Optional[str] = None

class SafetyViolation:
    def __init__(self, rule: str, severity: str, context: Dict):
        self.rule = rule
        self.severity = severity
        self.context = context
        self.timestamp = time.time()

class SafetyContext:
    def __init__(self):
        self.speed: float = 0.0
        self.location: Dict = {"latitude": 0.0, "longitude": 0.0}
        self.weather: str = "clear"
        self.time_of_day: str = "day"
        self.road_type: str = "normal"
        self.vehicle_state: Dict = {
            "engine": "on",
            "doors": "closed",
            "safety_systems": "active"
        }

class AutomotiveSafetyChecker(nn.Module):
    def __init__(self, rules_path: Optional[str] = None):
        super().__init__()
        self.rules = self._load_safety_rules(rules_path)
        self.violations_log = []
        self.command_patterns = self._compile_command_patterns()
        
    def _load_safety_rules(self, rules_path: Optional[str]) -> List[SafetyRule]:
        """Load safety rules from file or use defaults."""
        default_rules = [
            SafetyRule(
                name="speed_limit",
                conditions=["context.speed <= 130"],
                priority=1,
                recovery_action="reduce_speed"
            ),
            SafetyRule(
                name="autopilot_engagement",
                conditions=[
                    "context.road_type == 'highway'",
                    "context.weather in ['clear', 'cloudy']",
                    "context.time_of_day == 'day'",
                    "context.vehicle_state['safety_systems'] == 'active'"
                ],
                priority=1,
                recovery_action="disengage_autopilot"
            ),
            # Add more default rules
        ]
        
        if rules_path:
            try:
                with open(rules_path) as f:
                    custom_rules = json.load(f)
                return [SafetyRule(**rule) for rule in custom_rules]
            except Exception as e:
                print(f"Error loading custom rules: {e}. Using defaults.")
                return default_rules
        return default_rules
    
    def _compile_command_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for command recognition."""
        return {
            "speed_control": re.compile(r"(?i)(set|adjust|change)\s+speed\s+to\s+(\d+)"),
            "autopilot": re.compile(r"(?i)(enable|activate|turn\s+on)\s+autopilot"),
            "safety_override": re.compile(r"(?i)(disable|deactivate|override)\s+safety"),
            # Add more patterns
        }
    
    def check_command_safety(self, 
                           command: str, 
                           context: SafetyContext) -> Tuple[bool, Optional[SafetyViolation]]:
        """Check if a command is safe given the current context."""
        # Check for explicit safety overrides
        if self._is_safety_override(command):
            violation = SafetyViolation(
                "safety_override_attempted",
                "critical",
                vars(context)
            )
            self.violations_log.append(violation)
            return False, violation
        
        # Parse command intent
        command_type = self._identify_command_type(command)
        
        # Apply relevant safety rules
        for rule in sorted(self.rules, key=lambda x: x.priority):
            if self._rule_applies_to_command(rule, command_type):
                if not self._check_rule_conditions(rule, context):
                    violation = SafetyViolation(
                        rule.name,
                        "warning" if rule.recovery_action else "critical",
                        vars(context)
                    )
                    self.violations_log.append(violation)
                    return False, violation
        
        return True, None
    
    def _is_safety_override(self, command: str) -> bool:
        """Check if command attempts to override safety systems."""
        return bool(self.command_patterns["safety_override"].search(command))
    
    def _identify_command_type(self, command: str) -> str:
        """Identify the type of command using regex patterns."""
        for cmd_type, pattern in self.command_patterns.items():
            if pattern.search(command):
                return cmd_type
        return "unknown"
    
    def _rule_applies_to_command(self, rule: SafetyRule, command_type: str) -> bool:
        """Determine if a safety rule applies to a command type."""
        # Command-rule type matching logic
        command_rule_mapping = {
            "speed_control": ["speed_limit"],
            "autopilot": ["autopilot_engagement"],
            # Add more mappings
        }
        return command_type in command_rule_mapping and rule.name in command_rule_mapping[command_type]
    
    def _check_rule_conditions(self, rule: SafetyRule, context: SafetyContext) -> bool:
        """Check if all conditions of a rule are satisfied."""
        context_dict = vars(context)
        try:
            return all(eval(condition, {"__builtins__": None}, {"context": context}) 
                      for condition in rule.conditions)
        except Exception as e:
            print(f"Error evaluating rule conditions: {e}")
            return False
    
    def get_safety_report(self) -> Dict:
        """Generate a report of safety violations."""
        return {
            "total_violations": len(self.violations_log),
            "violations_by_type": self._categorize_violations(),
            "critical_violations": len([v for v in self.violations_log 
                                     if v.severity == "critical"]),
            "violations_detail": [vars(v) for v in self.violations_log]
        }
    
    def _categorize_violations(self) -> Dict:
        """Categorize violations by rule type."""
        categories = {}
        for violation in self.violations_log:
            categories[violation.rule] = categories.get(violation.rule, 0) + 1
        return categories

class SafetyFilter(nn.Module):
    def __init__(self, safety_checker: AutomotiveSafetyChecker):
        super().__init__()
        self.safety_checker = safety_checker
        
    def forward(self, 
                logits: torch.Tensor, 
                input_ids: torch.Tensor, 
                context: SafetyContext) -> torch.Tensor:
        """Filter model outputs based on safety rules."""
        batch_size = logits.shape[0]
        filtered_logits = logits.clone()
        
        for i in range(batch_size):
            # Convert logits to command
            command = self._logits_to_command(logits[i])
            
            # Check command safety
            is_safe, violation = self.safety_checker.check_command_safety(command, context)
            
            if not is_safe:
                # Mask out unsafe predictions
                filtered_logits[i] = torch.full_like(filtered_logits[i], float('-inf'))
                
                # If recovery action exists, modify logits to suggest safe alternative
                if violation and violation.rule in [r.name for r in self.safety_checker.rules]:
                    rule = next(r for r in self.safety_checker.rules if r.name == violation.rule)
                    if rule.recovery_action:
                        safe_command_ids = self._get_safe_alternative_ids(rule.recovery_action)
                        filtered_logits[i, safe_command_ids] = logits[i, safe_command_ids]
        
        return filtered_logits
    
    def _logits_to_command(self, logits: torch.Tensor) -> str:
        """Convert logits to command string."""
        # Get the most likely token IDs
        token_ids = torch.argmax(logits, dim=-1).unsqueeze(0)
        
        # Define automotive command types for mapping
        command_types = ["set_temperature", "navigate_to", "adjust_climate", 
                       "activate_cruise_control", "play_media"]
        
        # Use the first token to determine command type (simplified approach)
        if isinstance(token_ids, torch.Tensor) and token_ids.numel() > 0:
            # Use modulo to map to available command types
            cmd_idx = token_ids[0].item() % len(command_types)
            command = f"{command_types[cmd_idx]} param=value"
        else:
            command = "unknown command"
            
        return command
    
    def _get_safe_alternative_ids(self, recovery_action: str) -> torch.Tensor:
        """Get token IDs for safe alternative commands."""
        # Map recovery actions to token patterns that represent safe commands
        safe_alternatives = {
            "reduce_speed": [101, 345, 678],  # Example token IDs
            "disengage_autopilot": [202, 456, 789],
            "pull_over": [303, 567, 890],
        }
        
        # Get token IDs for the specified recovery action
        tokens = safe_alternatives.get(recovery_action, [404, 505, 606])
        
        # Create tensor on the same device as the model
        device = next(self.parameters()).device
        return torch.tensor(tokens, device=device)