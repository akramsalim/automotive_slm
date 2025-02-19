# safety/utils/safety_utils.py
from typing import Dict, List, Optional
import json
from pathlib import Path
import logging
import numpy as np
from datetime import datetime

def load_safety_config(config_path: Path) -> Dict:
    """Load safety configuration from file."""
    with open(config_path) as f:
        return json.load(f)

def validate_safety_config(config: Dict) -> bool:
    """Validate safety configuration structure."""
    required_fields = ["rules", "thresholds", "command_types"]
    return all(field in config for field in required_fields)

def create_safety_report(violations: List[Dict],
                        output_dir: Path,
                        include_plots: bool = True):
    """Create safety violation report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate statistics
    stats = {
        "total_violations": len(violations),
        "violations_by_type": _count_violations_by_type(violations),
        "violations_by_severity": _count_violations_by_severity(violations),
        "timeline": _create_violation_timeline(violations)
    }
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"safety_report_{timestamp}.json"
    
    with open(report_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    if include_plots:
        create_violation_plots(stats, output_dir)

def _count_violations_by_type(violations: List[Dict]) -> Dict[str, int]:
    """Count violations by type."""
    counts = {}
    for violation in violations:
        counts[violation["rule_name"]] = counts.get(violation["rule_name"], 0) + 1
    return counts

def _count_violations_by_severity(violations: List[Dict]) -> Dict[str, int]:
    """Count violations by severity."""
    counts = {}
    for violation in violations:
        counts[violation["severity"]] = counts.get(violation["severity"], 0) + 1
    return counts

def _create_violation_timeline(violations: List[Dict]) -> List[Dict]:
    """Create timeline of violations."""
    timeline = []
    for violation in sorted(violations, key=lambda x: x["timestamp"]):
        timeline.append({
            "timestamp": violation["timestamp"],
            "rule_name": violation["rule_name"],
            "severity": violation["severity"]
        })
    return timeline

def create_violation_plots(stats: Dict, output_dir: Path):
    """Create visualization plots for safety violations."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Violations by type
    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=list(stats["violations_by_type"].keys()),
        y=list(stats["violations_by_type"].values())
    )
    plt.title("Safety Violations by Type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "violations_by_type.png")
    plt.close()
    
    # Violations by severity
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x=list(stats["violations_by_severity"].keys()),
        y=list(stats["violations_by_severity"].values())
    )
    plt.title("Safety Violations by Severity")
    plt.tight_layout()
    plt.savefig(output_dir / "violations_by_severity.png")
    plt.close()