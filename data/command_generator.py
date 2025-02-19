# data/command_generator.py
from dataclasses import dataclass
from typing import List, Dict, Optional
import random
import json
import itertools
from pathlib import Path
import numpy as np

@dataclass
class CommandTemplate:
    intent: str
    templates: List[str]
    parameters: Dict[str, List[str]]
    constraints: Optional[Dict[str, List[str]]] = None

class AutomotiveCommandGenerator:
    def __init__(self):
        # Define core command categories
        self.command_types = {
            "climate": self._init_climate_commands(),
            "navigation": self._init_navigation_commands(),
            "vehicle_control": self._init_vehicle_control_commands(),
            "media": self._init_media_commands(),
            "system": self._init_system_commands()
        }
        
        # Initialize safety parameters
        self.safety_constraints = self._init_safety_constraints()
        
    def _init_climate_commands(self) -> CommandTemplate:
        return CommandTemplate(
            intent="climate_control",
            templates=[
                "Set {location} temperature to {temperature} degrees",
                "Adjust {location} climate to {temperature}",
                "{action} the {location} temperature",
                "Make it {temperature} degrees in the {location}"
            ],
            parameters={
                "location": ["front", "rear", "driver side", "passenger side", "all zones"],
                "temperature": [str(t) for t in range(16, 31)],
                "action": ["increase", "decrease", "maintain"]
            }
        )
    
    def _init_navigation_commands(self) -> CommandTemplate:
        return CommandTemplate(
            intent="navigation",
            templates=[
                "Navigate to {destination}",
                "Find route to {destination} via {preference}",
                "Take me to {destination} avoiding {avoid_type}",
                "Set destination to {destination}"
            ],
            parameters={
                "destination": ["<LOCATION_PLACEHOLDER>"],  # To be filled with real addresses
                "preference": ["highways", "toll-free", "fastest route", "eco route"],
                "avoid_type": ["tolls", "highways", "traffic", "construction"]
            }
        )
    
    def _init_vehicle_control_commands(self) -> CommandTemplate:
        return CommandTemplate(
            intent="vehicle_control",
            templates=[
                "{action} the {system}",
                "Set {system} to {state}",
                "Adjust {system} {parameter} to {value}"
            ],
            parameters={
                "action": ["activate", "deactivate", "enable", "disable"],
                "system": ["parking assist", "lane keeping", "cruise control", "auto-pilot"],
                "state": ["on", "off", "auto", "standby"],
                "parameter": ["speed", "distance", "sensitivity"],
                "value": ["low", "medium", "high"]
            },
            constraints={
                "system": {
                    "parking assist": ["sensitivity", "auto"],
                    "cruise control": ["speed", "distance"]
                }
            }
        )
    
    def _init_media_commands(self) -> CommandTemplate:
        return CommandTemplate(
            intent="media_control",
            templates=[
                "{action} {media_type}",
                "Switch to {media_type}",
                "Set {parameter} to {value}",
                "Play {content} on {media_type}"
            ],
            parameters={
                "action": ["play", "pause", "stop", "skip", "previous"],
                "media_type": ["radio", "bluetooth", "usb", "streaming"],
                "parameter": ["volume", "bass", "treble", "balance"],
                "value": ["up", "down", "max", "min"],
                "content": ["<MEDIA_PLACEHOLDER>"]  # To be filled with actual content
            }
        )
    
    def _init_system_commands(self) -> CommandTemplate:
        return CommandTemplate(
            intent="system_control",
            templates=[
                "Update {system} settings",
                "Check {system} status",
                "Run {system} diagnostic",
                "Configure {system} {parameter}"
            ],
            parameters={
                "system": ["display", "connectivity", "sensors", "software"],
                "parameter": ["brightness", "sensitivity", "mode", "language"]
            }
        )
    
    def _init_safety_constraints(self) -> Dict:
        return {
            "speed_limit": 130,  # km/h
            "temp_range": (16, 30),  # Celsius
            "restricted_commands": ["disable_safety_systems", "override_speed_limit"],
            "required_conditions": {
                "parking_assist": ["speed < 20", "daylight"],
                "autopilot": ["highway", "good_weather", "speed > 30"]
            }
        }

    def generate_command(self, command_type: str, num_variations: int = 1) -> List[Dict]:
        """Generate automotive command variations based on templates."""
        template = self.command_types.get(command_type)
        if not template:
            raise ValueError(f"Unknown command type: {command_type}")
        
        commands = []
        for _ in range(num_variations):
            # Select random template
            template_str = random.choice(template.templates)
            
            # Fill parameters
            params = {}
            for param in template.parameters:
                if template.constraints and param in template.constraints:
                    valid_values = template.constraints[param]
                    params[param] = random.choice(valid_values)
                else:
                    params[param] = random.choice(template.parameters[param])
            
            # Generate command
            command = template_str.format(**params)
            
            # Add metadata
            commands.append({
                "command": command,
                "intent": template.intent,
                "parameters": params,
                "safety_level": self._assess_safety(command, params)
            })
        
        return commands
    
    def _assess_safety(self, command: str, parameters: Dict) -> str:
        """Assess safety level of generated command."""
        # Check against safety constraints
        if any(restricted in command.lower() for restricted in self.safety_constraints["restricted_commands"]):
            return "unsafe"
        
        # Check parameter ranges
        if "temperature" in parameters:
            temp = int(parameters["temperature"])
            if temp < self.safety_constraints["temp_range"][0] or temp > self.safety_constraints["temp_range"][1]:
                return "warning"
        
        # Check system-specific constraints
        if "system" in parameters:
            system = parameters["system"]
            if system in self.safety_constraints["required_conditions"]:
                # In real implementation, would check actual conditions
                return "conditional"
        
        return "safe"

    def generate_synthetic_dataset(self, 
                                 num_samples: int,
                                 output_path: str,
                                 distribution: Optional[Dict[str, float]] = None) -> None:
        """Generate a full synthetic dataset with specified distribution of command types."""
        if distribution is None:
            distribution = {cmd_type: 1/len(self.command_types) for cmd_type in self.command_types}
        
        all_commands = []
        for cmd_type, prob in distribution.items():
            num_cmd = int(num_samples * prob)
            commands = self.generate_command(cmd_type, num_cmd)
            all_commands.extend(commands)
        
        # Shuffle commands
        random.shuffle(all_commands)
        
        # Save to file
        output_file = Path(output_path) / "synthetic_commands.json"
        with open(output_file, 'w') as f:
            json.dump(all_commands, f, indent=2)
        
        print(f"Generated {len(all_commands)} commands saved to {output_file}")

# Example usage
if __name__ == "__main__":
    generator = AutomotiveCommandGenerator()
    
    # Generate dataset with custom distribution
    distribution = {
        "climate": 0.2,
        "navigation": 0.3,
        "vehicle_control": 0.2,
        "media": 0.2,
        "system": 0.1
    }
    
    generator.generate_synthetic_dataset(
        num_samples=1000,
        output_path="./data",
        distribution=distribution
    )