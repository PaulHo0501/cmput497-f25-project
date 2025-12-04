"""
Configuration Generator for UED Analysis
Creates configuration files needed to run the EmotionDynamics UED analysis.
"""

import json
import os
from pathlib import Path

class UEDConfigGenerator:
    """
    Generates configuration files for running UED analysis on emotion diary data.
    """
    
    def __init__(self, base_dir="EmotionDynamics"):
        """
        Initialize the configuration generator.
        
        Args:
            base_dir: Path to the EmotionDynamics repository
        """
        self.base_dir = Path(base_dir)
        self.config_dir = Path("ued_configs")
        self.config_dir.mkdir(exist_ok=True)
    
    def create_basic_config(self, 
                           input_dir,
                           output_dir,
                           lexicon_path=None,
                           dimension='valence',
                           window_size=50,
                           step_size=1,
                           min_tokens=10):
        """
        Create a basic configuration file for UED analysis.
        
        Args:
            input_dir: Directory containing preprocessed user data
            output_dir: Where to save UED analysis results
            lexicon_path: Path to emotion lexicon (e.g., valence.csv)
            dimension: Emotion dimension to analyze ('valence', 'arousal', or 'dominance')
            window_size: Number of words in the rolling window (default: 50)
            step_size: How many words to move forward each step (default: 1)
            min_tokens: Minimum number of tokens required for analysis
        
        Returns:
            dict: Configuration dictionary
        """
        
        config = {
            # Input/Output paths
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            
            # Emotion lexicon settings
            "lexicon": {
                "path": str(lexicon_path) if lexicon_path else None,
                "dimension": dimension,
                "use_existing_scores": lexicon_path is None  # If no lexicon, use existing scores
            },
            
            # Window settings for rolling emotion calculation
            "window": {
                "size": window_size,
                "step": step_size
            },
            
            # Filtering criteria
            "filters": {
                "min_tokens": min_tokens,
                "min_entries_per_user": 3  # Minimum entries needed to calculate dynamics
            },
            
            # UED metrics to calculate
            "metrics": {
                "home_base": True,      # Most frequent emotional state
                "variability": True,     # Standard deviation of emotions
                "density": True,         # Concentration of emotional states
                "displacement": True,    # Total movement in emotion space
                "rise_rate": True,       # Rate of emotion escalation
                "recovery": True,        # Return to baseline
                "mean": True,           # Average emotional state
                "median": True,         # Median emotional state
                "entropy": True         # Unpredictability of emotions
            },
            
            # Output format settings
            "output": {
                "save_trajectories": True,  # Save emotion trajectories
                "save_summary": True,        # Save summary statistics
                "create_plots": True         # Generate visualization plots
            }
        }
        
        return config
    
    def save_config(self, config, filename="ued_config.json"):
        """
        Save configuration to a JSON file.
        
        Args:
            config: Configuration dictionary
            filename: Name of the config file
        """
        filepath = self.config_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to: {filepath}")
        return filepath
    
    def create_multi_dimension_configs(self, input_dir, output_base_dir):
        """
        Create separate configs for valence, arousal, and dominance analysis.
        
        Args:
            input_dir: Directory with preprocessed data
            output_base_dir: Base directory for outputs
        
        Returns:
            dict: Paths to created config files
        """
        dimensions = ['valence', 'arousal', 'dominance']
        config_paths = {}
        
        for dim in dimensions:
            output_dir = Path(output_base_dir) / f"{dim}_analysis"
            
            # Determine lexicon path if using EmotionDynamics lexicons
            lexicon_path = self.base_dir / "lexicons" / f"{dim}.csv"
            if not lexicon_path.exists():
                lexicon_path = None  # Will use existing scores instead
            
            config = self.create_basic_config(
                input_dir=input_dir,
                output_dir=str(output_dir),
                lexicon_path=str(lexicon_path) if lexicon_path else None,
                dimension=dim
            )
            
            config_file = f"ued_config_{dim}.json"
            config_paths[dim] = self.save_config(config, config_file)
        
        return config_paths
    
    def create_custom_config(self, **kwargs):
        """
        Create a custom configuration with user-specified parameters.
        
        Accepts any parameters that create_basic_config takes.
        """
        config = self.create_basic_config(**kwargs)
        return config


def generate_sample_configs():
    """
    Generate sample configuration files for different analysis scenarios.
    """
    generator = UEDConfigGenerator()
    
    # Scenario 1: Use existing valence/arousal scores from your data
    print("\n=== Creating Config for Existing Scores ===")
    config1 = generator.create_basic_config(
        input_dir="processed_emotion_data/users",
        output_dir="ued_results/existing_scores",
        lexicon_path=None,  # Use existing scores
        dimension='valence',
        window_size=50
    )
    generator.save_config(config1, "config_existing_scores.json")
    
    # Scenario 2: Recalculate from text using NRC VAD lexicon
    print("\n=== Creating Config for Lexicon-Based Analysis ===")
    config2 = generator.create_basic_config(
        input_dir="processed_emotion_data/text_corpus",
        output_dir="ued_results/lexicon_based",
        lexicon_path="EmotionDynamics/lexicons/valence.csv",
        dimension='valence',
        window_size=50
    )
    generator.save_config(config2, "config_lexicon_based.json")
    
    # Scenario 3: Multi-dimensional analysis
    print("\n=== Creating Multi-Dimension Configs ===")
    generator.create_multi_dimension_configs(
        input_dir="processed_emotion_data/users",
        output_base_dir="ued_results/multi_dimension"
    )
    
    # Scenario 4: Short window for rapid emotion changes
    print("\n=== Creating Config for Rapid Changes ===")
    config4 = generator.create_basic_config(
        input_dir="processed_emotion_data/users",
        output_dir="ued_results/rapid_changes",
        lexicon_path=None,
        dimension='arousal',
        window_size=20,  # Smaller window for finer-grained analysis
        step_size=1
    )
    generator.save_config(config4, "config_rapid_changes.json")
    
    print("\n✓ All configuration files generated!")
    print(f"  Location: {generator.config_dir}")


if __name__ == "__main__":
    generate_sample_configs()
