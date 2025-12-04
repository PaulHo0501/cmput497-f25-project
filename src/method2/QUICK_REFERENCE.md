# EmotionDynamics Adapter - Quick Reference

## Installation (One-Time Setup)

```bash
pip install pandas numpy scipy matplotlib seaborn --break-system-packages
```

## Three-Step Workflow

### Step 1: Preprocess
```bash
# Edit line 202 in preprocess_emotion_data.py with your CSV filename
python preprocess_emotion_data.py
```
**Output:** `processed_emotion_data/` directory

### Step 2: Configure
```bash
python generate_ued_configs.py
```
**Output:** `ued_configs/*.json` files

### Step 3: Analyze
```bash
python run_custom_ued.py ued_configs/config_existing_scores.json
```
**Output:** `ued_results/existing_scores/` directory

### Step 4: Visualize (Optional)
```bash
python visualize_ued_results.py ued_results/existing_scores valence
```
**Output:** `ued_results/existing_scores/visualizations/` directory

## File Structure

```
your_project/
├── emotion_diary.csv                    # Your input data
├── preprocess_emotion_data.py           # Run first
├── generate_ued_configs.py              # Run second
├── run_custom_ued.py                    # Run third
├── visualize_ued_results.py             # Run fourth (optional)
│
├── processed_emotion_data/              # After step 1
│   ├── users/                           # Individual user files
│   ├── text_corpus/                     # Text-only files
│   └── all_users_combined.csv          # Combined data
│
├── ued_configs/                         # After step 2
│   ├── config_existing_scores.json     # Use your scores
│   ├── config_lexicon_based.json       # Use NRC VAD
│   └── ...
│
└── ued_results/                         # After step 3
    └── existing_scores/                 # Analysis results
        ├── ued_summary.csv             # All users' metrics
        ├── individual_users/           # Per-user results
        └── visualizations/             # Plots (after step 4)
```

## Key Metrics at a Glance

| Metric | What it Measures | Interpretation |
|--------|-----------------|----------------|
| **mean** | Average emotion | Higher = more positive (valence) or more activated (arousal) |
| **variability** | Emotional stability | Higher = more variable/unstable |
| **home_base** | Default emotion | Most frequent emotional state |
| **displacement** | Total movement | How much emotions change overall |
| **rise_rate** | Escalation speed | How fast emotions increase |
| **recovery_rate** | Return speed | How fast emotions decrease/recover |
| **entropy** | Unpredictability | Higher = less predictable patterns |
| **density** | Concentration | Higher = emotions cluster together |

## Common Config Adjustments

### Change Window Size
```json
{
  "window": {
    "size": 20,    // Smaller = more detail, Larger = smoother
    "step": 1
  }
}
```

### Filter Users
```json
{
  "filters": {
    "min_entries_per_user": 5    // Only analyze users with 5+ entries
  }
}
```

### Switch Dimension
```json
{
  "lexicon": {
    "dimension": "arousal"    // Options: "valence", "arousal", "dominance"
  }
}
```

## Quick Commands

### Analyze All Dimensions
```bash
for dim in valence arousal dominance; do
  python run_custom_ued.py ued_configs/ued_config_${dim}.json
  python visualize_ued_results.py ued_results/multi_dimension/${dim}_analysis ${dim}
done
```

### View Results
```bash
# Summary statistics
cat ued_results/existing_scores/ued_summary.csv

# Individual user metrics
cat ued_results/existing_scores/individual_users/user_3_metrics.csv
```

### Python Quick Analysis
```python
import pandas as pd

# Load summary
summary = pd.read_csv('ued_results/existing_scores/ued_summary.csv')

# View statistics
print(summary.describe())

# Compare users
print(summary[['user_id', 'mean', 'variability', 'entropy']])

# Load trajectory
traj = pd.read_csv('ued_results/existing_scores/individual_users/user_3_trajectory.csv')
print(traj.head())
```

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| "No user files found" | Check `input_dir` in config matches `processed_emotion_data/users/` |
| "Module not found" | Run `pip install pandas numpy scipy matplotlib seaborn --break-system-packages` |
| "Could not create trajectory" | User has too few entries, lower `min_entries_per_user` in config |
| "Lexicon not found" | Set `"lexicon": {"path": null}` in config to use existing scores |

## One-Liner Data Check

```bash
# Check preprocessing output
ls -R processed_emotion_data/

# Check how many users were processed
ls processed_emotion_data/users/ | wc -l

# Preview summary results
head -n 5 ued_results/existing_scores/ued_summary.csv

# Check visualizations
ls ued_results/existing_scores/visualizations/
```

## Example Customization

### Analyze Only High-Activity Users
Edit config:
```json
{
  "filters": {
    "min_entries_per_user": 10    // Only users with 10+ entries
  }
}
```

### Focus on Rapid Changes
Edit config:
```json
{
  "window": {
    "size": 15,     // Smaller window
    "step": 1       // Fine-grained
  }
}
```

### Use Multiple Dimensions
```bash
# Create separate analyses
python run_custom_ued.py ued_configs/ued_config_valence.json
python run_custom_ued.py ued_configs/ued_config_arousal.json

# Compare results
python -c "
import pandas as pd
v = pd.read_csv('ued_results/multi_dimension/valence_analysis/ued_summary.csv')
a = pd.read_csv('ued_results/multi_dimension/arousal_analysis/ued_summary.csv')
print('Valence mean:', v['mean'].mean())
print('Arousal mean:', a['mean'].mean())
"
```

## Documentation Files

- **README.md** - Complete overview and usage guide
- **TUTORIAL.md** - Step-by-step walkthrough with examples
- **ADAPTATION_GUIDE.md** - Conceptual understanding and adaptation details
- **QUICK_REFERENCE.md** - This file (cheat sheet)

## Getting Help

1. Read error messages carefully
2. Check file paths in configs
3. Verify data format matches expected structure
4. Consult TUTORIAL.md for detailed examples
5. Review original repo: https://github.com/Priya22/EmotionDynamics
