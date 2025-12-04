"""
Visualization Tools for UED Analysis Results
Creates plots and visualizations for emotion dynamics analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class UEDVisualizer:
    """
    Creates visualizations for UED analysis results.
    """
    
    def __init__(self, results_dir):
        """
        Initialize visualizer with results directory.
        
        Args:
            results_dir: Path to UED analysis results
        """
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / 'visualizations'
        self.output_dir.mkdir(exist_ok=True)
        
        # Load summary if available
        summary_file = self.results_dir / 'ued_summary.csv'
        if summary_file.exists():
            self.summary = pd.read_csv(summary_file)
            print(f"Loaded summary for {len(self.summary)} users")
        else:
            self.summary = None
            print(f"Warning: Summary file not found at {summary_file}")
    
    def plot_emotion_trajectory(self, user_id, dimension='valence'):
        """
        Plot emotion trajectory for a single user.
        
        Args:
            user_id: User ID to plot
            dimension: Emotion dimension name (for title)
        """
        trajectory_file = self.results_dir / 'individual_users' / f'user_{user_id}_trajectory.csv'
        
        if not trajectory_file.exists():
            print(f"Trajectory file not found: {trajectory_file}")
            return
        
        trajectory = pd.read_csv(trajectory_file)
        
        plt.figure(figsize=(14, 6))
        
        # Main trajectory plot
        plt.subplot(1, 2, 1)
        plt.plot(trajectory['time_point'], 
                trajectory['emotion_value'], 
                marker='o', 
                linewidth=2,
                markersize=6,
                alpha=0.7)
        
        # Add mean line
        mean_val = trajectory['emotion_value'].mean()
        plt.axhline(y=mean_val, color='r', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.2f}')
        
        plt.xlabel('Time Point', fontsize=12)
        plt.ylabel(f'{dimension.capitalize()} Value', fontsize=12)
        plt.title(f'User {user_id}: {dimension.capitalize()} Trajectory', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Distribution plot
        plt.subplot(1, 2, 2)
        plt.hist(trajectory['emotion_value'], bins=15, alpha=0.7, edgecolor='black')
        plt.axvline(x=mean_val, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        plt.xlabel(f'{dimension.capitalize()} Value', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'User {user_id}: {dimension.capitalize()} Distribution', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.output_dir / f'user_{user_id}_trajectory.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    def plot_all_trajectories(self, dimension='valence'):
        """
        Plot all user trajectories on one figure for comparison.
        
        Args:
            dimension: Emotion dimension name
        """
        individual_dir = self.results_dir / 'individual_users'
        trajectory_files = list(individual_dir.glob('*_trajectory.csv'))
        
        if not trajectory_files:
            print("No trajectory files found")
            return
        
        plt.figure(figsize=(14, 8))
        
        for traj_file in trajectory_files:
            user_id = traj_file.stem.replace('user_', '').replace('_trajectory', '')
            trajectory = pd.read_csv(traj_file)
            
            plt.plot(trajectory['time_point'], 
                    trajectory['emotion_value'],
                    marker='o',
                    label=f'User {user_id}',
                    alpha=0.7,
                    linewidth=2)
        
        plt.xlabel('Time Point', fontsize=12)
        plt.ylabel(f'{dimension.capitalize()} Value', fontsize=12)
        plt.title(f'All Users: {dimension.capitalize()} Trajectories Comparison', 
                 fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'all_trajectories_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    def plot_metrics_comparison(self, metrics=None):
        """
        Create bar plots comparing UED metrics across users.
        
        Args:
            metrics: List of metric names to plot. If None, plots key metrics.
        """
        if self.summary is None:
            print("No summary data available")
            return
        
        if metrics is None:
            # Default key metrics
            metrics = ['mean', 'variability', 'displacement', 'rise_rate', 'recovery_rate', 'entropy']
        
        # Filter to available metrics
        available_metrics = [m for m in metrics if m in self.summary.columns]
        
        if not available_metrics:
            print("No requested metrics found in summary")
            return
        
        n_metrics = len(available_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            data = self.summary[['user_id', metric]].copy()
            data['user_id'] = data['user_id'].astype(str)
            
            ax.bar(data['user_id'], data[metric], alpha=0.7, edgecolor='black')
            ax.set_xlabel('User ID', fontsize=11)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for idx in range(len(available_metrics), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'metrics_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    def plot_metrics_correlation(self):
        """
        Create correlation heatmap of UED metrics.
        """
        if self.summary is None:
            print("No summary data available")
            return
        
        # Select numeric columns only (exclude user_id, num_entries)
        numeric_cols = self.summary.select_dtypes(include=[np.number]).columns
        metric_cols = [col for col in numeric_cols if col not in ['user_id', 'num_entries']]
        
        if len(metric_cols) < 2:
            print("Not enough metrics for correlation analysis")
            return
        
        # Calculate correlation
        corr_matrix = self.summary[metric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   linewidths=1,
                   cbar_kws={"shrink": 0.8})
        
        plt.title('UED Metrics Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        output_file = self.output_dir / 'metrics_correlation.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    def plot_metrics_distribution(self, metrics=None):
        """
        Create distribution plots for UED metrics.
        
        Args:
            metrics: List of metrics to plot. If None, plots all.
        """
        if self.summary is None:
            print("No summary data available")
            return
        
        numeric_cols = self.summary.select_dtypes(include=[np.number]).columns
        
        if metrics is None:
            metrics = [col for col in numeric_cols if col not in ['user_id', 'num_entries']]
        else:
            metrics = [m for m in metrics if m in numeric_cols]
        
        if not metrics:
            print("No metrics available for distribution plots")
            return
        
        n_metrics = len(metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            values = self.summary[metric].dropna()
            
            ax.hist(values, bins=min(10, len(values)), alpha=0.7, edgecolor='black')
            ax.axvline(x=values.mean(), color='r', linestyle='--', linewidth=2, 
                      label=f'Mean: {values.mean():.2f}')
            ax.axvline(x=values.median(), color='g', linestyle='--', linewidth=2,
                      label=f'Median: {values.median():.2f}')
            
            ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for idx in range(len(metrics), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'metrics_distributions.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()
    
    def create_summary_report(self, dimension='valence'):
        """
        Create a comprehensive visualization report.
        
        Args:
            dimension: Emotion dimension being analyzed
        """
        print(f"\n{'='*60}")
        print(f"Creating Visualization Report for {dimension.capitalize()}")
        print(f"{'='*60}\n")
        
        # Plot individual trajectories
        individual_dir = self.results_dir / 'individual_users'
        if individual_dir.exists():
            trajectory_files = list(individual_dir.glob('*_trajectory.csv'))
            print(f"Plotting {len(trajectory_files)} individual trajectories...")
            
            for traj_file in trajectory_files:
                user_id = traj_file.stem.replace('user_', '').replace('_trajectory', '')
                self.plot_emotion_trajectory(user_id, dimension)
        
        # Plot all trajectories comparison
        print("\nCreating trajectories comparison...")
        self.plot_all_trajectories(dimension)
        
        if self.summary is not None and len(self.summary) > 0:
            # Plot metrics comparison
            print("Creating metrics comparison...")
            self.plot_metrics_comparison()
            
            # Plot correlation heatmap (if enough data)
            if len(self.summary) > 2:
                print("Creating correlation heatmap...")
                self.plot_metrics_correlation()
            
            # Plot distributions
            print("Creating metrics distributions...")
            self.plot_metrics_distribution()
        
        print(f"\n{'='*60}")
        print(f"✓ Visualization report complete!")
        print(f"  Output directory: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    """
    Main function to generate visualizations.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_ued_results.py <results_directory> [dimension]")
        print("\nExample:")
        print("  python visualize_ued_results.py ued_results/existing_scores valence")
        return
    
    results_dir = sys.argv[1]
    dimension = sys.argv[2] if len(sys.argv) > 2 else 'valence'
    
    if not Path(results_dir).exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    visualizer = UEDVisualizer(results_dir)
    visualizer.create_summary_report(dimension)


if __name__ == "__main__":
    main()
