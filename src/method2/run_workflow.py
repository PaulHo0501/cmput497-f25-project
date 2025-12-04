#!/usr/bin/env python3
"""
Master Script for EmotionDynamics Analysis
Runs the complete workflow: preprocess -> configure -> analyze -> visualize
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse


def run_command(command, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {command}\n")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"\n✗ Error: {description} failed")
        return False
    
    print(f"\n✓ {description} completed successfully")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    
    required_packages = ['pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n✗ Missing packages: {', '.join(missing_packages)}")
        print("\nInstall with:")
        print(f"pip install {' '.join(missing_packages)} --break-system-packages")
        return False
    
    print("✓ All dependencies installed\n")
    return True


def run_full_workflow(input_csv, dimension='valence', use_existing_scores=True):
    """
    Run the complete UED analysis workflow.
    
    Args:
        input_csv: Path to input CSV file
        dimension: Emotion dimension to analyze
        use_existing_scores: Whether to use existing emotion scores
    """
    
    # Step 0: Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies first.")
        return False
    
    # Verify input file exists
    if not Path(input_csv).exists():
        print(f"\n✗ Error: Input file not found: {input_csv}")
        return False
    
    print(f"\n{'#'*60}")
    print(f"# EmotionDynamics Analysis Workflow")
    print(f"#")
    print(f"# Input: {input_csv}")
    print(f"# Dimension: {dimension}")
    print(f"# Using existing scores: {use_existing_scores}")
    print(f"{'#'*60}")
    
    # Step 1: Preprocess
    success = run_command(
        f"python src/method2/preprocess_emotion_data.py",
        "Step 1: Preprocessing Data"
    )
    if not success:
        return False
    
    # Step 2: Generate configs
    success = run_command(
        "python src/method2/generate_ued_configs.py",
        "Step 2: Generating Configurations"
    )
    if not success:
        return False
    
    # Step 3: Run analysis
    if use_existing_scores:
        config_file = "ued_configs/config_existing_scores.json"
    else:
        config_file = "ued_configs/config_lexicon_based.json"
    
    success = run_command(
        f"python src/method2/run_custom_ued.py {config_file}",
        "Step 3: Running UED Analysis"
    )
    if not success:
        return False
    
    # Step 4: Create visualizations
    results_dir = "ued_results/existing_scores" if use_existing_scores else "ued_results/lexicon_based"
    
    success = run_command(
        f"python src/method2/visualize_ued_results.py {results_dir} {dimension}",
        "Step 4: Creating Visualizations"
    )
    if not success:
        return False
    
    print(f"\n{'#'*60}")
    print(f"# ✓ WORKFLOW COMPLETE!")
    print(f"#")
    print(f"# Results location:")
    print(f"#   - Summary: {results_dir}/ued_summary.csv")
    print(f"#   - Individual users: {results_dir}/individual_users/")
    print(f"#   - Visualizations: {results_dir}/visualizations/")
    print(f"{'#'*60}\n")
    
    return True


def run_step(step_name, **kwargs):
    """Run a specific step of the workflow."""
    
    if step_name == 'preprocess':
        return run_command(
            "python src/method2/preprocess_emotion_data.py",
            "Preprocessing Data"
        )
    
    elif step_name == 'configure':
        return run_command(
            "python src/method2/generate_ued_configs.py",
            "Generating Configurations"
        )
    
    elif step_name == 'analyze':
        config = kwargs.get('config', 'ued_configs/config_existing_scores.json')
        return run_command(
            f"python run_custom_ued.py {config}",
            "Running UED Analysis"
        )
    
    elif step_name == 'visualize':
        results_dir = kwargs.get('results_dir', 'ued_results/existing_scores')
        dimension = kwargs.get('dimension', 'valence')
        return run_command(
            f"python src/method2/visualize_ued_results.py {results_dir} {dimension}",
            "Creating Visualizations"
        )
    
    else:
        print(f"✗ Unknown step: {step_name}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='EmotionDynamics Analysis Workflow Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete workflow with existing scores
  python run_workflow.py --input emotion_diary.csv --full
  
  # Run complete workflow with lexicon-based analysis
  python run_workflow.py --input emotion_diary.csv --full --use-lexicon
  
  # Run specific steps
  python run_workflow.py --step preprocess
  python run_workflow.py --step analyze --config ued_configs/config_existing_scores.json
  python run_workflow.py --step visualize --results ued_results/existing_scores
  
  # Analyze multiple dimensions
  python run_workflow.py --input emotion_diary.csv --full --dimension arousal
        """
    )
    
    parser.add_argument('--input', '-i', 
                       help='Input CSV file (required for --full)')
    
    parser.add_argument('--full', '-f', action='store_true',
                       help='Run complete workflow (preprocess -> analyze -> visualize)')
    
    parser.add_argument('--step', '-s',
                       choices=['preprocess', 'configure', 'analyze', 'visualize'],
                       help='Run a specific step only')
    
    parser.add_argument('--dimension', '-d', default='valence',
                       choices=['valence', 'arousal', 'dominance'],
                       help='Emotion dimension to analyze (default: valence)')
    
    parser.add_argument('--use-lexicon', action='store_true',
                       help='Use lexicon-based analysis instead of existing scores')
    
    parser.add_argument('--config', '-c',
                       help='Config file for analyze step')
    
    parser.add_argument('--results', '-r',
                       help='Results directory for visualize step')
    
    parser.add_argument('--check-deps', action='store_true',
                       help='Check if dependencies are installed')
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check_deps:
        check_dependencies()
        return
    
    # Run full workflow
    if args.full:
        if not args.input:
            print("✗ Error: --input is required for full workflow")
            parser.print_help()
            sys.exit(1)
        
        success = run_full_workflow(
            args.input,
            dimension=args.dimension,
            use_existing_scores=not args.use_lexicon
        )
        sys.exit(0 if success else 1)
    
    # Run specific step
    elif args.step:
        kwargs = {
            'config': args.config,
            'results_dir': args.results,
            'dimension': args.dimension
        }
        success = run_step(args.step, **kwargs)
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
