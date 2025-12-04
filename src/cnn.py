"""
COMPLETE CNN TUTORIAL
Step-by-step guide to train CNN, predict, and evaluate on train/val/test sets
"""

# ============================================================================
# STEP 1: UNDERSTANDING YOUR DATA
# ============================================================================

"""
Your SemEval data format:
user_id,text_id,text,timestamp,collection_phase,is_words,valence,arousal
3,251,I've been feeling just fine...,2021-06-08 12:26:16,1,False,1.0,1.0
3,252,I've been feeling pretty good...,2021-06-09 13:41:40,1,False,0.0,1.0

What we need:
- text: Input text (English)
- valence: Target score (continuous value)
- arousal: Target score (continuous value)
- user_id: For splitting data (avoid data leakage)
"""

# ============================================================================
# STEP 2: DATA SPLITTING
# ============================================================================

import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(csv_path, test_size=0.2, val_size=0.1):
    """
    Split data into train/val/test by users (not random).
    This prevents data leakage (same user in train and test).
    
    Args:
        csv_path: Path to your CSV file
        test_size: Proportion for test set (0.2 = 20%)
        val_size: Proportion for validation (from remaining)
    
    Returns:
        train_df, val_df, test_df
    """
    print("="*60)
    print("STEP 1: LOADING AND SPLITTING DATA")
    print("="*60)
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} samples from {df['user_id'].nunique()} users")
    
    # Remove missing values
    df = df.dropna(subset=['text', 'valence', 'arousal'])
    print(f"After removing NaN: {len(df)} samples")
    
    # Get unique users
    users = df['user_id'].unique()
    print(f"Total users: {len(users)}")
    
    # Split users (NOT samples!) to avoid leakage
    # Example: If user 3 is in training, ALL their texts go to training
    train_users, test_users = train_test_split(
        users, 
        test_size=test_size, 
        random_state=42
    )
    
    train_users, val_users = train_test_split(
        train_users, 
        test_size=val_size/(1-test_size),  # Adjust proportion
        random_state=42
    )
    
    # Create dataframes
    train_df = df[df['user_id'].isin(train_users)]
    val_df = df[df['user_id'].isin(val_users)]
    test_df = df[df['user_id'].isin(test_users)]
    
    print(f"\n{'='*60}")
    print(f"SPLIT SUMMARY:")
    print(f"{'='*60}")
    print(f"Train: {len(train_df)} samples from {len(train_users)} users ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val:   {len(val_df)} samples from {len(val_users)} users ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test:  {len(test_df)} samples from {len(test_users)} users ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


# ============================================================================
# STEP 3: TRAINING THE CNN MODEL
# ============================================================================

def train_cnn_model(train_df, val_df, epochs=15, batch_size=32):
    """
    Train CNN model on training data, validate on validation data.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        epochs: Number of training epochs
        batch_size: Batch size
    
    Returns:
        Trained model
    """
    
    from cnn_model import CNNVAPredictor
    
    print("\n" + "="*60)
    print("STEP 2: TRAINING CNN MODEL")
    print("="*60)
    
    # Create predictor
    predictor = CNNVAPredictor(
        vocab_size=10000,      # Use top 10k words
        embedding_dim=300,     # Word embedding dimension
        num_filters=100,       # Number of CNN filters
        filter_sizes=[3,4,5]   # N-gram sizes to capture
    )
    
    # Train
    predictor.train(
        train_df=train_df,
        val_df=val_df,
        epochs=epochs,
        batch_size=batch_size,
        lr=0.001
    )
    
    print("\n✓ Training complete!")
    print("✓ Model saved to: models/cnn_va_model.pt")
    
    return predictor


# ============================================================================
# STEP 4: MAKING PREDICTIONS
# ============================================================================

def predict_with_cnn(predictor, df, set_name="Test"):
    """
    Make predictions on a dataset.
    
    Args:
        predictor: Trained CNN predictor
        df: Dataframe with 'text' column
        set_name: Name for display (Train/Val/Test)
    
    Returns:
        predictions (valences, arousals)
    """
    print(f"\n{'='*60}")
    print(f"STEP 3: PREDICTING ON {set_name.upper()} SET")
    print(f"{'='*60}")
    
    print(f"\nPredicting {len(df)} samples...")
    
    # Get predictions
    valences, arousals = predictor.predict(
        df['text'].tolist(),
        batch_size=32
    )
    
    print(f"✓ Predictions complete for {set_name} set")
    
    return valences, arousals


# ============================================================================
# STEP 5: EVALUATION
# ============================================================================

def evaluate_predictions(df, valences, arousals, set_name="Test"):
    """
    Calculate evaluation metrics.
    
    Args:
        df: Dataframe with ground truth
        valences: Predicted valence scores
        arousals: Predicted arousal scores
        set_name: Name for display
    
    Returns:
        Dictionary with all metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import pearsonr, spearmanr
    import numpy as np
    
    print(f"\n{'='*60}")
    print(f"STEP 4: EVALUATING {set_name.upper()} SET")
    print(f"{'='*60}")
    
    # Ground truth
    true_valences = df['valence'].values
    true_arousals = df['arousal'].values
    
    # VALENCE METRICS
    val_mse = mean_squared_error(true_valences, valences)
    val_rmse = np.sqrt(val_mse)
    val_mae = mean_absolute_error(true_valences, valences)
    val_r2 = r2_score(true_valences, valences)
    val_pearson, val_p = pearsonr(true_valences, valences)
    val_spearman, val_sp = spearmanr(true_valences, valences)
    
    # AROUSAL METRICS
    ar_mse = mean_squared_error(true_arousals, arousals)
    ar_rmse = np.sqrt(ar_mse)
    ar_mae = mean_absolute_error(true_arousals, arousals)
    ar_r2 = r2_score(true_arousals, arousals)
    ar_pearson, ar_p = pearsonr(true_arousals, arousals)
    ar_spearman, ar_sp = spearmanr(true_arousals, arousals)
    
    # Print results
    print(f"\n{'VALENCE METRICS':-^60}")
    print(f"  MSE:             {val_mse:.4f}")
    print(f"  RMSE:            {val_rmse:.4f}  ← Lower is better")
    print(f"  MAE:             {val_mae:.4f}  ← Lower is better")
    print(f"  R² Score:        {val_r2:.4f}  ← Higher is better (max 1.0)")
    print(f"  Pearson r:       {val_pearson:.4f}  ← Higher is better (max 1.0)")
    print(f"  Pearson p-value: {val_p:.4f}")
    print(f"  Spearman ρ:      {val_spearman:.4f}")
    
    print(f"\n{'AROUSAL METRICS':-^60}")
    print(f"  MSE:             {ar_mse:.4f}")
    print(f"  RMSE:            {ar_rmse:.4f}  ← Lower is better")
    print(f"  MAE:             {ar_mae:.4f}  ← Lower is better")
    print(f"  R² Score:        {ar_r2:.4f}  ← Higher is better (max 1.0)")
    print(f"  Pearson r:       {ar_pearson:.4f}  ← Higher is better (max 1.0)")
    print(f"  Pearson p-value: {ar_p:.4f}")
    print(f"  Spearman ρ:      {ar_spearman:.4f}")
    
    # COMBINED
    avg_rmse = (val_rmse + ar_rmse) / 2
    avg_pearson = (val_pearson + ar_pearson) / 2
    
    print(f"\n{'COMBINED METRICS':-^60}")
    print(f"  Average RMSE:    {avg_rmse:.4f}")
    print(f"  Average Pearson: {avg_pearson:.4f}")
    
    # Interpretation
    print(f"\n{'INTERPRETATION':-^60}")
    if avg_rmse < 0.15:
        print(f"  ⭐⭐⭐⭐⭐ EXCELLENT performance!")
    elif avg_rmse < 0.20:
        print(f"  ⭐⭐⭐⭐ VERY GOOD performance!")
    elif avg_rmse < 0.25:
        print(f"  ⭐⭐⭐ GOOD performance!")
    elif avg_rmse < 0.30:
        print(f"  ⭐⭐ ACCEPTABLE performance")
    else:
        print(f"  ⭐ Need improvement")
    
    # Return all metrics
    return {
        'set_name': set_name,
        'valence_mse': val_mse,
        'valence_rmse': val_rmse,
        'valence_mae': val_mae,
        'valence_r2': val_r2,
        'valence_pearson': val_pearson,
        'valence_pearson_p': val_p,
        'valence_spearman': val_spearman,
        'arousal_mse': ar_mse,
        'arousal_rmse': ar_rmse,
        'arousal_mae': ar_mae,
        'arousal_r2': ar_r2,
        'arousal_pearson': ar_pearson,
        'arousal_pearson_p': ar_p,
        'arousal_spearman': ar_spearman,
        'avg_rmse': avg_rmse,
        'avg_pearson': avg_pearson
    }


# ============================================================================
# STEP 6: SAVE PREDICTIONS
# ============================================================================

def save_predictions(df, valences, arousals, output_path, set_name):
    """
    Save predictions to CSV.
    
    Args:
        df: Original dataframe
        valences: Predicted valences
        arousals: Predicted arousals
        output_path: Where to save
        set_name: Train/Val/Test
    """
    result_df = df.copy()
    result_df['predicted_valence'] = valences
    result_df['predicted_arousal'] = arousals
    result_df['set'] = set_name
    
    result_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved predictions to: {output_path}")


# ============================================================================
# STEP 7: VISUALIZE RESULTS
# ============================================================================

def visualize_results(all_metrics):
    """
    Create visualization comparing train/val/test performance.
    
    Args:
        all_metrics: List of metric dictionaries
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    sns.set_style("whitegrid")
    
    print(f"\n{'='*60}")
    print("STEP 5: CREATING VISUALIZATIONS")
    print("="*60)
    
    # Extract data
    set_names = [m['set_name'] for m in all_metrics]
    val_rmse = [m['valence_rmse'] for m in all_metrics]
    ar_rmse = [m['arousal_rmse'] for m in all_metrics]
    val_pearson = [m['valence_pearson'] for m in all_metrics]
    ar_pearson = [m['arousal_pearson'] for m in all_metrics]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # RMSE comparison
    x = range(len(set_names))
    width = 0.35
    
    axes[0, 0].bar([i - width/2 for i in x], val_rmse, width, label='Valence', alpha=0.8)
    axes[0, 0].bar([i + width/2 for i in x], ar_rmse, width, label='Arousal', alpha=0.8)
    axes[0, 0].set_ylabel('RMSE (Lower is Better)', fontsize=11)
    axes[0, 0].set_title('RMSE Comparison', fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(set_names)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Pearson comparison
    axes[0, 1].bar([i - width/2 for i in x], val_pearson, width, label='Valence', alpha=0.8)
    axes[0, 1].bar([i + width/2 for i in x], ar_pearson, width, label='Arousal', alpha=0.8)
    axes[0, 1].set_ylabel('Pearson r (Higher is Better)', fontsize=11)
    axes[0, 1].set_title('Pearson Correlation', fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(set_names)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Overfitting check
    if 'Train' in set_names and 'Val' in set_names:
        train_idx = set_names.index('Train')
        val_idx = set_names.index('Val')
        
        train_rmse = (val_rmse[train_idx] + ar_rmse[train_idx]) / 2
        val_rmse_avg = (val_rmse[val_idx] + ar_rmse[val_idx]) / 2
        
        gap = val_rmse_avg - train_rmse
        
        axes[1, 0].bar(['Train', 'Validation'], [train_rmse, val_rmse_avg], alpha=0.7)
        axes[1, 0].set_ylabel('Average RMSE', fontsize=11)
        axes[1, 0].set_title(f'Overfitting Check (Gap: {gap:.4f})', fontsize=13, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        if gap < 0.05:
            axes[1, 0].text(0.5, 0.5, '✓ No Overfitting', 
                          transform=axes[1, 0].transAxes,
                          ha='center', va='center',
                          bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
                          fontsize=12)
        elif gap < 0.10:
            axes[1, 0].text(0.5, 0.5, '⚠ Slight Overfitting', 
                          transform=axes[1, 0].transAxes,
                          ha='center', va='center',
                          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                          fontsize=12)
        else:
            axes[1, 0].text(0.5, 0.5, '✗ Overfitting!', 
                          transform=axes[1, 0].transAxes,
                          ha='center', va='center',
                          bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
                          fontsize=12)
    
    # Summary table
    axes[1, 1].axis('off')
    table_data = []
    for m in all_metrics:
        table_data.append([
            m['set_name'],
            f"{m['valence_rmse']:.3f}",
            f"{m['arousal_rmse']:.3f}",
            f"{m['avg_pearson']:.3f}"
        ])
    
    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=['Set', 'Val RMSE', 'Ar RMSE', 'Avg Pearson'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0.2, 1, 0.6]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(set_names) + 1):
        table[(i, 0)].set_facecolor('#E8E8E8' if i == 0 else 'white')
        table[(i, 0)].set_text_props(weight='bold' if i == 0 else 'normal')
    
    axes[1, 1].set_title('Summary Table', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('cnn_evaluation_results.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: cnn_evaluation_results.png")
    plt.close()


# ============================================================================
# COMPLETE WORKFLOW
# ============================================================================

def complete_cnn_workflow(csv_path, epochs=15, batch_size=32):
    """
    Complete workflow: Split → Train → Predict → Evaluate on all sets.
    
    Args:
        csv_path: Path to your SemEval CSV file
        epochs: Training epochs
        batch_size: Batch size
    """
    print("\n" + "#"*60)
    print("# COMPLETE CNN WORKFLOW")
    print("# Train, Predict, and Evaluate on Train/Val/Test")
    print("#"*60)
    
    # STEP 1: Split data
    train_df, val_df, test_df = split_data(csv_path)
    
    # STEP 2: Train model
    predictor = train_cnn_model(train_df, val_df, epochs=epochs, batch_size=batch_size)
    
    # STEP 3 & 4: Predict and evaluate on all sets
    all_metrics = []
    
    # Train set
    train_val, train_ar = predict_with_cnn(predictor, train_df, "Train")
    train_metrics = evaluate_predictions(train_df, train_val, train_ar, "Train")
    all_metrics.append(train_metrics)
    save_predictions(train_df, train_val, train_ar, 
                    'predictions_train.csv', 'Train')
    
    # Validation set
    val_val, val_ar = predict_with_cnn(predictor, val_df, "Validation")
    val_metrics = evaluate_predictions(val_df, val_val, val_ar, "Validation")
    all_metrics.append(val_metrics)
    save_predictions(val_df, val_val, val_ar, 
                    'predictions_val.csv', 'Validation')
    
    # Test set
    test_val, test_ar = predict_with_cnn(predictor, test_df, "Test")
    test_metrics = evaluate_predictions(test_df, test_val, test_ar, "Test")
    all_metrics.append(test_metrics)
    save_predictions(test_df, test_val, test_ar, 
                    'predictions_test.csv', 'Test')
    
    # STEP 5: Visualize
    visualize_results(all_metrics)
    
    # Final summary
    print("\n" + "#"*60)
    print("# COMPLETE!")
    print("#"*60)
    print("\nGenerated files:")
    print("  ✓ models/cnn_va_model.pt (trained model)")
    print("  ✓ predictions_train.csv")
    print("  ✓ predictions_val.csv")
    print("  ✓ predictions_test.csv")
    print("  ✓ cnn_evaluation_results.png")
    
    print("\nFinal Performance Summary:")
    print(f"  Train: RMSE={train_metrics['avg_rmse']:.4f}, Pearson={train_metrics['avg_pearson']:.3f}")
    print(f"  Val:   RMSE={val_metrics['avg_rmse']:.4f}, Pearson={val_metrics['avg_pearson']:.3f}")
    print(f"  Test:  RMSE={test_metrics['avg_rmse']:.4f}, Pearson={test_metrics['avg_pearson']:.3f}")
    
    # Check overfitting
    gap = val_metrics['avg_rmse'] - train_metrics['avg_rmse']
    if gap < 0.05:
        print("\n✓ Model is well-generalized (no overfitting)")
    elif gap < 0.10:
        print("\n⚠ Slight overfitting detected")
    else:
        print("\n✗ Significant overfitting! Consider:")
        print("    - More training data")
        print("    - Stronger regularization (increase dropout)")
        print("    - Fewer epochs")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Complete CNN training and evaluation workflow'
    )
    parser.add_argument('--data', required=True, 
                       help='Path to your SemEval CSV file')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of epochs (default: 15)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    
    args = parser.parse_args()
    
    # Run complete workflow
    complete_cnn_workflow(
        csv_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


"""
==============================================================================
HOW TO USE THIS SCRIPT
==============================================================================

1. BASIC USAGE:
   python cnn_complete_tutorial.py --data semeval_data.csv

2. WITH CUSTOM SETTINGS:
   python cnn_complete_tutorial.py --data semeval_data.csv --epochs 20 --batch-size 64

3. WHAT IT DOES:
   - Splits data into train/val/test (by users, no leakage)
   - Trains CNN model
   - Predicts on all three sets
   - Evaluates with MSE, RMSE, MAE, Pearson correlation
   - Saves predictions to CSV
   - Creates visualization

4. OUTPUT FILES:
   - models/cnn_va_model.pt (trained model)
   - predictions_train.csv (train predictions)
   - predictions_val.csv (validation predictions)
   - predictions_test.csv (test predictions)
   - cnn_evaluation_results.png (visualization)

5. UNDERSTANDING METRICS:
   - RMSE: Lower is better (< 0.20 is good)
   - Pearson r: Higher is better (> 0.70 is good)
   - R²: Higher is better (> 0.50 is good)

6. CHECKING OVERFITTING:
   Compare train vs validation RMSE:
   - Gap < 0.05: No overfitting ✓
   - Gap 0.05-0.10: Slight overfitting ⚠
   - Gap > 0.10: Significant overfitting ✗

==============================================================================
"""