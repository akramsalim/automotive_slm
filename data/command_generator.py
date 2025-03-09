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
                "destination": ["Central Park", "Home", "Airport", "Downtown", "Shopping Mall", "Office"],
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
                "content": ["Jazz", "News", "Podcast", "Rock", "Classical"]
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
            param_choices = {}
            
            for param in template.parameters:
                if template.constraints and param in template.constraints:
                    # Handle parameter constraints
                    system_choice = param_choices.get("system")
                    if system_choice and system_choice in template.constraints[param]:
                        valid_values = template.constraints[param][system_choice]
                        param_values = [v for v in template.parameters[param] if v in valid_values]
                        params[param] = random.choice(param_values)
                    else:
                        params[param] = random.choice(template.parameters[param])
                else:
                    params[param] = random.choice(template.parameters[param])
                
                # Store choice for constraint checking
                param_choices[param] = params[param]
            
            # Format template with parameters
            try:
                command_text = template_str.format(**params)
            except KeyError:
                # If formatting fails (e.g., missing parameter), use a simple fallback
                command_text = f"{template.intent} command"
            
            # Create vehicle state for context
            vehicle_state = self._generate_vehicle_state()
            
            # Add metadata
            commands.append({
                "command": command_text,
                "intent": template.intent,
                "parameters": params,
                "safety_level": self._assess_safety(command_text, params),
                "category": template.intent,
                "vehicle_state": vehicle_state,
                "safety_rules": self._get_safety_rules(template.intent, params)
            })
        
        return commands
    
    def _generate_vehicle_state(self) -> Dict:
        """Generate random vehicle state for context."""
        return {
            "speed": random.randint(0, 140),
            "time_of_day": random.choice(["day", "night"]),
            "weather": random.choice(["clear", "rain", "snow", "fog"]),
            "road_type": random.choice(["city", "highway", "rural"]),
            "engine_status": "on",
            "lights": random.choice(["on", "off", "auto"])
        }
    
    def _get_safety_rules(self, intent: str, parameters: Dict) -> Dict:
        """Get relevant safety rules for the command."""
        if intent == "climate_control":
            return {
                "temperature": {
                    "min": self.safety_constraints["temp_range"][0],
                    "max": self.safety_constraints["temp_range"][1]
                }
            }
        elif intent == "vehicle_control":
            rules = {
                "speed_limit": self.safety_constraints["speed_limit"]
            }
            
            # Add specific rules based on system
            if "system" in parameters:
                system = parameters["system"]
                if system in self.safety_constraints["required_conditions"]:
                    rules[system] = self.safety_constraints["required_conditions"][system]
            
            return rules
        elif intent == "navigation":
            return {
                "input_while_moving": False
            }
        
        # Default empty rules for other intents
        return {}
    
    def _assess_safety(self, command: str, parameters: Dict) -> str:
        """Assess safety level of generated command."""
        # Check against safety constraints
        if any(restricted in command.lower() for restricted in self.safety_constraints["restricted_commands"]):
            return "unsafe"
        
        # Check parameter ranges
        if "temperature" in parameters:
            try:
                temp = int(parameters["temperature"])
                if temp < self.safety_constraints["temp_range"][0] or temp > self.safety_constraints["temp_range"][1]:
                    return "warning"
            except (ValueError, TypeError):
                pass
        
        # Check system-specific constraints
        if "system" in parameters:
            system = parameters["system"]
            if system in self.safety_constraints["required_conditions"]:
                # In a real implementation, we would check actual conditions
                # Here we randomly determine if conditions are met
                if random.random() < 0.2:  # 20% chance of not meeting conditions
                    return "conditional"
        
        return "safe"

    def generate_synthetic_dataset(self, 
                                 num_samples: int,
                                 output_path: str,
                                 distribution: Optional[Dict[str, float]] = None) -> List[Dict]:
        """Generate a full synthetic dataset with specified distribution of command types."""
        if distribution is None:
            distribution = {cmd_type: 1/len(self.command_types) for cmd_type in self.command_types}
        
        # Normalize distribution if needed
        total = sum(distribution.values())
        if total != 1.0:
            distribution = {k: v/total for k, v in distribution.items()}
        
        all_commands = []
        for cmd_type, prob in distribution.items():
            num_cmd = max(1, int(num_samples * prob))
            commands = self.generate_command(cmd_type, num_cmd)
            all_commands.extend(commands)
        
        # Shuffle commands
        random.shuffle(all_commands)
        
        # Trim to exactly num_samples if needed
        all_commands = all_commands[:num_samples]
        
        # Save to file
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "synthetic_commands.json"
        with open(output_file, 'w') as f:
            json.dump(all_commands, f, indent=2)
        
        print(f"Generated {len(all_commands)} commands saved to {output_file}")
        return all_commands

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