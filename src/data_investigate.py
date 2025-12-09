import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        exit(1)
    return pd.read_csv(filepath)

def process_subtask1(input_file):
    """
    Subtask 1: Count valence values in bins [-2, -1, 0, 1, 2]
    and arousal in bins [0, 1, 2].
    """
    df = load_data(input_file)
    print(f"--- Subtask 1 Analysis: {len(df)} rows ---")


    v_bins = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    v_labels = [-2, -1, 0, 1, 2]

    a_bins = [-0.5, 0.5, 1.5, 2.5]
    a_labels = [0, 1, 2]

    df['valence_bin'] = pd.cut(df['valence'], bins=v_bins, labels=v_labels)
    df['arousal_bin'] = pd.cut(df['arousal'], bins=a_bins, labels=a_labels)


    v_counts = df['valence_bin'].value_counts().sort_index()
    a_counts = df['arousal_bin'].value_counts().sort_index()

    print("\nValence Distribution:")
    print(v_counts)
    print("\nArousal Distribution:")
    print(a_counts)

    fig, axes = plt.subplots(2, 1, figsize=(10, 14))

    sns.barplot(x=v_counts.index, y=v_counts.values, ax=axes[0], palette="coolwarm")
    axes[0].set_title("Distribution of Valence Scores")
    axes[0].set_xlabel("Valence Bin")
    axes[0].set_ylabel("Count")

    sns.barplot(x=a_counts.index, y=a_counts.values, ax=axes[1], palette="viridis")
    axes[1].set_title("Distribution of Arousal Scores")
    axes[1].set_xlabel("Arousal Bin")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig("subtask1_distribution.png")
    print("Saved plot to 'subtask1_distribution.png'")
    plt.show()

def process_subtask2a(input_file):
    """
    Subtask 2a: Highlight state change in valence score for 3 users with ~100 texts.
    Normalized x-axis (0, 1, 2...) representing the sequence of texts.
    """
    df = load_data(input_file)
    print(f"--- Subtask 2a Analysis: {len(df)} rows ---")

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        print("Error: 'timestamp' column missing.")
        return
    
    if 'valence' not in df.columns:
        print("Error: 'valence' column missing. Please ensure you are using train_subtask2a.csv or a file with this column.")
        return

    user_counts = df['user_id'].value_counts()
    user_counts_diff = (user_counts - 30).abs()
    target_users = user_counts_diff.nsmallest(3).index.tolist()
    
    print(f"Selected Users (counts): {[(u, user_counts[u]) for u in target_users]}")

    subset = df[df['user_id'].isin(target_users)].copy()
    
    subset.sort_values(by=['user_id', 'timestamp'], inplace=True)


    subset['time_step'] = subset.groupby('user_id').cumcount()

    plt.figure(figsize=(14, 7))
    
    sns.lineplot(
        data=subset, 
        x='time_step', 
        y='valence', 
        hue='user_id', 
        marker='o',
        palette="tab10",
        linewidth=2.5
    )

    plt.title("Valence Trajectory for Users with ~30 Texts (Normalized Time)")
    plt.xlabel("Text Sequence Number (0 = First Post)")
    plt.ylabel("Valence State Change (Next - Current)")
    plt.legend(title="User ID", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("subtask2a_top3_users_change.png")
    print("Saved plot to 'subtask2a_top3_users_change.png'")
    plt.show()

def process_subtask2b(input_file):
    """
    Subtask 2b: Visualize change in mean Valence vs Arousal.
    Assumes input file has 'disposition_change_valence' and 'disposition_change_arousal'.
    """
    df = load_data(input_file)
    print(f"--- Subtask 2b Analysis: {len(df)} rows ---")

    val_col = 'disposition_change_valence'
    aro_col = 'disposition_change_arousal'

    if val_col not in df.columns or aro_col not in df.columns:
        print(f"Error: Columns '{val_col}' or '{aro_col}' not found.")
        print("Please use the file that contains the calculated disposition changes (e.g. train_subtask2b.csv).")
        return


    g = sns.JointGrid(data=df, x=val_col, y=aro_col, space=0, ratio=17)
    
    g.plot_joint(sns.kdeplot, fill=True, cmap="Blues", alpha=0.6, thresh=0.05)
    g.plot_joint(sns.scatterplot, color="black", s=10, alpha=0.4)
    
    g.plot_marginals(sns.histplot, color="#03051A", alpha=1, bins=25)

    g.set_axis_labels("Disposition Change: Valence", "Disposition Change: Arousal")
    
    g.ax_joint.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    g.ax_joint.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)

    plt.suptitle("User Disposition Shift (2nd Half - 1st Half)", y=1.02)
    
    plt.savefig("subtask2b_disposition_change.png", bbox_inches='tight')
    print("Saved plot to 'subtask2b_disposition_change.png'")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="SemEval Data Investigation Tool")
    parser.add_argument("--task", type=str, required=True, choices=['1', '2a', '2b'], help="Task ID")
    parser.add_argument("--input", type=str, required=True, help="Path to CSV file")
    
    args = parser.parse_args()

    if args.task == '1':
        process_subtask1(args.input)
    elif args.task == '2a':
        process_subtask2a(args.input)
    elif args.task == '2b':
        process_subtask2b(args.input)

if __name__ == "__main__":
    main()
