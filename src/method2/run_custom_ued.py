"""
UED Analysis Runner for Custom Emotion Diary Data
Wrapper script that adapts the EmotionDynamics UED code for your dataset.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class CustomUEDAnalyzer:
    """
    Wrapper class to run UED analysis on emotion diary data.
    Adapts the EmotionDynamics code for your specific dataset format.
    """
    
    def __init__(self, config_path):
        """
        Initialize the analyzer with a configuration file.
        
        Args:
            config_path: Path to the JSON configuration file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.input_dir = Path(self.config['input_dir'])
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.lexicon = None
        if self.config['lexicon']['path']:
            self.load_lexicon()
    
    def load_lexicon(self):
        """Load emotion lexicon if specified."""
        lexicon_path = self.config['lexicon']['path']
        if os.path.exists(lexicon_path):
            self.lexicon = pd.read_csv(lexicon_path)
            print(f"Loaded lexicon: {lexicon_path}")
            print(f"  Lexicon size: {len(self.lexicon)} words")
        else:
            print(f"Warning: Lexicon not found at {lexicon_path}")
            print("  Will use existing emotion scores instead")
    
    def load_user_data(self, user_file):
        """
        Load data for a single user.
        
        Args:
            user_file: Path to user data file (JSON or CSV)
        
        Returns:
            DataFrame with user's entries
        """
        if user_file.suffix == '.json':
            with open(user_file, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data['entries'])
        elif user_file.suffix == '.csv':
            return pd.read_csv(user_file)
        else:
            raise ValueError(f"Unsupported file format: {user_file.suffix}")
    
    def tokenize_text(self, text):
        """
        Simple tokenization of text.
        
        Args:
            text: Text string to tokenize
        
        Returns:
            List of tokens (words)
        """
        # Remove punctuation and convert to lowercase
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return tokens
    
    def calculate_emotion_scores(self, tokens):
        """
        Calculate emotion scores for a sequence of tokens using lexicon.
        
        Args:
            tokens: List of word tokens
        
        Returns:
            List of emotion scores (one per token)
        """
        if self.lexicon is None:
            return None
        
        scores = []
        lexicon_dict = dict(zip(self.lexicon['word'], self.lexicon['val']))
        
        for token in tokens:
            if token in lexicon_dict:
                scores.append(lexicon_dict[token])
        
        return scores
    
    def create_emotion_trajectory(self, user_df):
        """
        Create emotion trajectory using rolling window.
        
        Args:
            user_df: DataFrame with user's entries
        
        Returns:
            dict: Emotion trajectory data
        """
        window_size = self.config['window']['size']
        step_size = self.config['window']['step']
        dimension = self.config['lexicon']['dimension']
        use_existing = self.config['lexicon'].get('use_existing_scores', False)
        
        trajectory = {
            'time_points': [],
            'emotion_values': [],
            'entry_ids': []
        }
        
        # Concatenate all text with entry boundaries
        all_tokens = []
        token_to_entry = []  # Maps token index to entry ID
        
        for idx, row in user_df.iterrows():
            tokens = self.tokenize_text(row['text'])
            entry_id = row.get('text_id', idx)
            
            all_tokens.extend(tokens)
            token_to_entry.extend([entry_id] * len(tokens))
        
        # Check if we have enough tokens
        if len(all_tokens) < self.config['filters']['min_tokens']:
            return None
        
        # Calculate emotion scores for rolling windows
        if use_existing and dimension in user_df.columns:
            # Use existing scores from the dataset
            # Map them to the appropriate windows
            for i, row in user_df.iterrows():
                if pd.notna(row[dimension]):
                    trajectory['time_points'].append(row.get('timestamp', i))
                    trajectory['emotion_values'].append(float(row[dimension]))
                    trajectory['entry_ids'].append(row.get('text_id', i))
        
        elif self.lexicon is not None:
            # Calculate from text using lexicon
            for i in range(0, len(all_tokens) - window_size + 1, step_size):
                window_tokens = all_tokens[i:i + window_size]
                scores = self.calculate_emotion_scores(window_tokens)
                
                if scores and len(scores) > 0:
                    avg_score = np.mean(scores)
                    trajectory['time_points'].append(i)
                    trajectory['emotion_values'].append(avg_score)
                    trajectory['entry_ids'].append(token_to_entry[i])
        
        else:
            print("Warning: No emotion scores available (no lexicon and no existing scores)")
            return None
        
        return trajectory if len(trajectory['emotion_values']) > 0 else None
    
    def calculate_ued_metrics(self, trajectory):
        """
        Calculate UED metrics from emotion trajectory.
        
        Args:
            trajectory: Emotion trajectory dictionary
        
        Returns:
            dict: UED metrics
        """
        values = np.array(trajectory['emotion_values'])
        
        if len(values) < 3:
            return None
        
        metrics = {}
        config_metrics = self.config['metrics']
        
        # Basic statistics
        if config_metrics.get('mean', True):
            metrics['mean'] = float(np.mean(values))
        
        if config_metrics.get('median', True):
            metrics['median'] = float(np.median(values))
        
        # Variability
        if config_metrics.get('variability', True):
            metrics['variability'] = float(np.std(values))
            metrics['variance'] = float(np.var(values))
        
        # Home base (most common emotional state)
        if config_metrics.get('home_base', True):
            # Discretize into bins for home base calculation
            bins = np.linspace(values.min(), values.max(), 10)
            hist, _ = np.histogram(values, bins=bins)
            home_base_bin = np.argmax(hist)
            metrics['home_base'] = float((bins[home_base_bin] + bins[home_base_bin + 1]) / 2)
        
        # Density (inverse of spread)
        if config_metrics.get('density', True):
            metrics['density'] = 1.0 / (np.std(values) + 1e-6)
        
        # Displacement (total distance traveled)
        if config_metrics.get('displacement', True):
            displacement = np.sum(np.abs(np.diff(values)))
            metrics['displacement'] = float(displacement)
            metrics['avg_displacement'] = float(displacement / len(values))
        
        # Rise rate and recovery
        if config_metrics.get('rise_rate', True) or config_metrics.get('recovery', True):
            diffs = np.diff(values)
            
            if config_metrics.get('rise_rate', True):
                rises = diffs[diffs > 0]
                metrics['rise_rate'] = float(np.mean(rises)) if len(rises) > 0 else 0.0
                metrics['max_rise'] = float(np.max(rises)) if len(rises) > 0 else 0.0
            
            if config_metrics.get('recovery', True):
                falls = diffs[diffs < 0]
                metrics['recovery_rate'] = float(np.mean(np.abs(falls))) if len(falls) > 0 else 0.0
                metrics['max_fall'] = float(np.min(falls)) if len(falls) > 0 else 0.0
        
        # Entropy (unpredictability)
        if config_metrics.get('entropy', True):
            # Discretize and calculate entropy
            bins = np.linspace(values.min(), values.max(), 10)
            hist, _ = np.histogram(values, bins=bins)
            probs = hist / hist.sum()
            probs = probs[probs > 0]  # Remove zero probabilities
            entropy = -np.sum(probs * np.log2(probs))
            metrics['entropy'] = float(entropy)
        
        # Range
        metrics['range'] = float(values.max() - values.min())
        metrics['min_value'] = float(values.min())
        metrics['max_value'] = float(values.max())
        
        return metrics
    
    def analyze_user(self, user_file):
        """
        Run complete UED analysis for a single user.
        
        Args:
            user_file: Path to user data file
        
        Returns:
            dict: Analysis results
        """
        user_id = user_file.stem.replace('user_', '')
        print(f"\nAnalyzing user {user_id}...")
        
        # Load user data
        user_df = self.load_user_data(user_file)
        
        # Check minimum entries requirement
        min_entries = self.config['filters']['min_entries_per_user']
        if len(user_df) < min_entries:
            print(f"  Skipping: Only {len(user_df)} entries (minimum: {min_entries})")
            return None
        
        # Create emotion trajectory
        trajectory = self.create_emotion_trajectory(user_df)
        if trajectory is None:
            print(f"  Skipping: Could not create emotion trajectory")
            return None
        
        print(f"  Created trajectory with {len(trajectory['emotion_values'])} points")
        
        # Calculate UED metrics
        metrics = self.calculate_ued_metrics(trajectory)
        if metrics is None:
            print(f"  Skipping: Could not calculate metrics")
            return None
        
        print(f"  Calculated {len(metrics)} UED metrics")
        
        results = {
            'user_id': user_id,
            'num_entries': len(user_df),
            'trajectory': trajectory,
            'metrics': metrics
        }
        
        # Save individual user results if configured
        if self.config['output'].get('save_trajectories', False):
            self.save_user_results(results)
        
        return results
    
    def save_user_results(self, results):
        """Save individual user analysis results."""
        user_output_dir = self.output_dir / 'individual_users'
        user_output_dir.mkdir(exist_ok=True)
        
        user_id = results['user_id']
        
        # Save trajectory
        trajectory_df = pd.DataFrame({
            'time_point': results['trajectory']['time_points'],
            'emotion_value': results['trajectory']['emotion_values'],
            'entry_id': results['trajectory']['entry_ids']
        })
        trajectory_df.to_csv(
            user_output_dir / f'user_{user_id}_trajectory.csv',
            index=False
        )
        
        # Save metrics
        metrics_df = pd.DataFrame([results['metrics']])
        metrics_df['user_id'] = user_id
        metrics_df.to_csv(
            user_output_dir / f'user_{user_id}_metrics.csv',
            index=False
        )
    
    def run_analysis(self):
        """
        Run UED analysis on all users in the input directory.
        
        Returns:
            DataFrame: Summary of results for all users
        """
        print(f"\n{'='*60}")
        print(f"Running UED Analysis")
        print(f"{'='*60}")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Dimension: {self.config['lexicon']['dimension']}")
        
        # Find all user files
        user_files = list(self.input_dir.glob('user_*.json')) + \
                    list(self.input_dir.glob('user_*.csv'))
        
        if not user_files:
            print(f"\nError: No user files found in {self.input_dir}")
            print("  Expected files like: user_3.json, user_7.csv, etc.")
            return None
        
        print(f"\nFound {len(user_files)} user files")
        
        # Analyze each user
        all_results = []
        for user_file in user_files:
            result = self.analyze_user(user_file)
            if result:
                all_results.append(result)
        
        if not all_results:
            print("\nNo users were successfully analyzed")
            return None
        
        print(f"\n{'='*60}")
        print(f"Successfully analyzed {len(all_results)} users")
        print(f"{'='*60}")
        
        # Create summary DataFrame
        summary_data = []
        for result in all_results:
            row = {'user_id': result['user_id'], 'num_entries': result['num_entries']}
            row.update(result['metrics'])
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        if self.config['output'].get('save_summary', True):
            summary_file = self.output_dir / 'ued_summary.csv'
            summary_df.to_csv(summary_file, index=False)
            print(f"\nSaved summary: {summary_file}")
        
        # Print summary statistics
        print("\n=== Summary Statistics Across All Users ===")
        print(summary_df.describe().round(3))
        
        return summary_df


def main():
    """
    Main function to run UED analysis.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python run_custom_ued.py <config_file>")
        print("\nExample:")
        print("  python run_custom_ued.py ued_configs/config_existing_scores.json")
        return
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return
    
    # Run analysis
    analyzer = CustomUEDAnalyzer(config_path)
    summary = analyzer.run_analysis()
    
    if summary is not None:
        print("\n✓ Analysis complete!")
        print(f"  Results saved to: {analyzer.output_dir}")
    else:
        print("\n✗ Analysis failed")


if __name__ == "__main__":
    main()
