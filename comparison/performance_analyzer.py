# comparison/performance_analyzer.py
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from scipy import stats

class PerformanceAnalyzer:
    """Analyzer for model performance metrics."""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize performance analyzer.
        
        Args:
            significance_level: Statistical significance level
        """
        self.significance_level = significance_level
    
    def analyze_performance_differences(self,
                                     results: Dict[str, List[float]]) -> Dict:
        """
        Analyze statistical differences in model performances.
        
        Args:
            results: Dictionary of model results
            
        Returns:
            Analysis results
        """
        analysis = {
            "descriptive_stats": self._calculate_descriptive_stats(results),
            "statistical_tests": self._perform_statistical_tests(results),
            "effect_sizes": self._calculate_effect_sizes(results)
        }
        
        return analysis
    
    def _calculate_descriptive_stats(self,
                                   results: Dict[str, List[float]]) -> Dict:
        """Calculate descriptive statistics."""
        stats_dict = {}
        
        for model_name, values in results.items():
            stats_dict[model_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "median": np.median(values),
                "min": np.min(values),
                "max": np.max(values)
            }
        
        return stats_dict
    
    def _perform_statistical_tests(self,
                                 results: Dict[str, List[float]]) -> Dict:
        """Perform statistical tests between models."""
        model_names = list(results.keys())
        tests = {}
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(
                    results[model1],
                    results[model2]
                )
                
                tests[f"{model1}_vs_{model2}"] = {
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < self.significance_level
                }
        
        return tests
    
    def _calculate_effect_sizes(self,
                              results: Dict[str, List[float]]) -> Dict:
        """Calculate effect sizes between models."""
        model_names = list(results.keys())
        effect_sizes = {}
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                
                # Calculate Cohen's d
                d = self._cohens_d(
                    results[model1],
                    results[model2]
                )
                
                effect_sizes[f"{model1}_vs_{model2}"] = {
                    "cohens_d": d,
                    "effect_magnitude": self._interpret_cohens_d(d)
                }
        
        return effect_sizes
    
    def _cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        return (np.mean(group1) - np.mean(group2)) / pooled_se
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
