# Complete Tutorial: Running Emotion Dynamics on Your Custom Dataset

## Quick Start Guide

### Prerequisites
```bash
# Install required Python packages
pip install pandas numpy scipy matplotlib --break-system-packages
```

### Step 1: Prepare Your Data

Save your emotion diary CSV file (the one you showed me) to your working directory. Let's call it `emotion_diary.csv`.

### Step 2: Preprocess the Data

Edit `preprocess_emotion_data.py` and update the input file path:

```python
# Line 202: Change this line
input_csv = "emotion_diary.csv"  # Update with your actual filename
```

Then run:

```bash
python preprocess_emotion_data.py
```

This will:
- Load your CSV file
- Organize data by user
- Create individual user files in `processed_emotion_data/users/`
- Generate text corpus files
- Create a combined CSV file

**Expected Output:**
```
Loading data from emotion_diary.csv...
Loaded 10 entries from 3 users

=== User Statistics ===
         num_entries first_entry          last_entry  duration_days
user_id                                                            
3               5     2021-06-08 12:26:16  2021-06-13 14:54:41           5
7               2     2021-03-11 12:10:58  2021-03-12 12:43:39           1
10              4     2021-06-09 12:11:03  2021-06-11 17:18:21           2

✓ Preprocessing complete!
  Output directory: processed_emotion_data
  Total users: 3
  Total entries: 10
```

### Step 3: Generate Configuration Files

Run:

```bash
python generate_ued_configs.py
```

This creates several configuration files in `ued_configs/` for different analysis scenarios:
- `config_existing_scores.json` - Uses your existing valence/arousal values
- `config_lexicon_based.json` - Recalculates from text using NRC VAD lexicon
- `config_rapid_changes.json` - Analyzes rapid emotional changes
- Multi-dimensional configs for valence, arousal, and dominance

### Step 4: Run UED Analysis

Choose which config to use based on your needs:

**Option A: Use Your Existing Scores (Recommended)**
```bash
python run_custom_ued.py ued_configs/config_existing_scores.json
```

**Option B: Recalculate from Text Using Lexicon**
```bash
# First, download the NRC VAD lexicon from:
# http://saifmohammad.com/WebPages/nrc-vad.html
# Place it in EmotionDynamics/lexicons/

python run_custom_ued.py ued_configs/config_lexicon_based.json
```

**Expected Output:**
```
============================================================
Running UED Analysis
============================================================
Input directory: processed_emotion_data/users
Output directory: ued_results/existing_scores
Dimension: valence

Found 3 user files

Analyzing user 3...
  Created trajectory with 5 points
  Calculated 15 UED metrics

Analyzing user 7...
  Created trajectory with 2 points
  Skipping: Could not calculate metrics

Analyzing user 10...
  Created trajectory with 4 points
  Calculated 15 UED metrics

============================================================
Successfully analyzed 2 users
============================================================

Saved summary: ued_results/existing_scores/ued_summary.csv

=== Summary Statistics Across All Users ===
       user_id  num_entries    mean  median  variability  ...
count      2.0          2.0   2.000   2.000        2.000  ...
mean       6.5          4.5   1.000   1.000        0.707  ...
std        4.9          0.7   0.707   0.707        0.707  ...
...

✓ Analysis complete!
  Results saved to: ued_results/existing_scores
```

### Step 5: Examine Results

Your results will be in the output directory specified in the config (e.g., `ued_results/existing_scores/`):

```
ued_results/existing_scores/
├── ued_summary.csv                      # Summary metrics for all users
└── individual_users/
    ├── user_3_trajectory.csv            # User 3's emotion trajectory
    ├── user_3_metrics.csv               # User 3's UED metrics
    ├── user_10_trajectory.csv
    └── user_10_metrics.csv
```

## Understanding the Output

### UED Metrics Explained

1. **mean / median**: Average emotional state
   - For valence: Higher = more positive
   - For arousal: Higher = more activated/energetic

2. **variability / variance**: How much emotions fluctuate
   - Higher = more emotional instability
   - Lower = more emotional stability

3. **home_base**: Most frequent emotional state
   - The "default" emotion the person returns to

4. **density**: Concentration of emotional states
   - Higher = emotions cluster in a narrow range
   - Lower = emotions spread widely

5. **displacement / avg_displacement**: Total emotional movement
   - How much someone's emotions change over time

6. **rise_rate / max_rise**: Speed of positive emotional changes
   - How quickly emotions escalate

7. **recovery_rate / max_fall**: Speed of negative emotional changes
   - How quickly emotions drop or recover

8. **entropy**: Unpredictability of emotional states
   - Higher = more unpredictable emotional patterns
   - Lower = more predictable emotional patterns

9. **range / min_value / max_value**: Emotional range
   - Span of emotions experienced

## Advanced Usage

### Analyzing Multiple Dimensions

To analyze valence, arousal, and dominance separately:

```bash
# Generate multi-dimension configs
python generate_ued_configs.py

# Run each dimension
python run_custom_ued.py ued_configs/ued_config_valence.json
python run_custom_ued.py ued_configs/ued_config_arousal.json
python run_custom_ued.py ued_configs/ued_config_dominance.json
```

### Adjusting Window Size

Smaller windows capture rapid changes, larger windows smooth out noise:

Edit the config file:
```json
{
  "window": {
    "size": 20,    // Change from 50 to 20 for finer detail
    "step": 1
  }
}
```

### Filtering Users

Set minimum requirements in the config:
```json
{
  "filters": {
    "min_tokens": 10,
    "min_entries_per_user": 5    // Only analyze users with 5+ entries
  }
}
```

## Comparing with Original EmotionDynamics

If you want to use the original EmotionDynamics code directly:

1. Clone the repository:
```bash
git clone https://github.com/Priya22/EmotionDynamics.git
```

2. Your preprocessed data is already compatible!
   - Use files in `processed_emotion_data/text_corpus/` for lexicon-based analysis
   - Use files in `processed_emotion_data/users/` for direct analysis

3. Follow their README in the `code/` folder

## Troubleshooting

### "No user files found"
- Check that preprocessing completed successfully
- Verify the input_dir in your config matches where files were saved
- Files should be named like `user_3.json` or `user_3.csv`

### "Could not create emotion trajectory"
- User might have too few entries (check min_entries_per_user)
- Text might be too short (check min_tokens)
- If using lexicon: verify lexicon path is correct

### "Lexicon not found"
- Download NRC VAD lexicon from http://saifmohammad.com/WebPages/nrc-vad.html
- Place in `EmotionDynamics/lexicons/` directory
- Or use existing scores by setting lexicon path to `null` in config

## Visualization (Optional)

Create plots of emotion trajectories:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load a user's trajectory
trajectory = pd.read_csv('ued_results/existing_scores/individual_users/user_3_trajectory.csv')

# Plot
plt.figure(figsize=(12, 6))
plt.plot(trajectory['time_point'], trajectory['emotion_value'], marker='o')
plt.xlabel('Time Point')
plt.ylabel('Emotion Value')
plt.title('User 3 Emotion Trajectory')
plt.grid(True)
plt.savefig('user_3_trajectory.png')
plt.show()
```

## Next Steps

1. **Statistical Analysis**: Compare UED metrics across different user groups
2. **Longitudinal Analysis**: Track how metrics change over collection phases
3. **Correlations**: Examine relationships between different UED metrics
4. **Predictive Modeling**: Use UED metrics as features for prediction tasks
5. **Intervention Studies**: Analyze how UED metrics change after interventions

## Citation

If you use this adapted code, please cite the original EmotionDynamics work:

```bibtex
@inproceedings{VM2022-TED,
  title={Tweet Emotion Dynamics: Emotion Word Usage in Tweets from US and Canada},
  author={Krishnapriya Vishnubhotla and Saif M. Mohammad},
  booktitle={Proceedings of LREC 2022},
  year={2022}
}

@article{hipson2021emotion,
  doi={10.1371/journal.pone.0256153},
  author={Hipson, Will E. AND Mohammad, Saif M.},
  journal={PLOS ONE},
  title={Emotion dynamics in movie dialogues},
  year={2021}
}
```
