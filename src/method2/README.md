# EmotionDynamics Adapter for Custom Emotion Diary Data

A complete toolkit for adapting the [EmotionDynamics framework](https://github.com/Priya22/EmotionDynamics) to work with custom emotion diary datasets.

## Overview

This package provides scripts to:
1. **Preprocess** your emotion diary CSV into the format expected by EmotionDynamics
2. **Generate** configuration files for different analysis scenarios
3. **Run** UED (Utterance Emotion Dynamics) analysis on your data
4. **Visualize** results with comprehensive plots and charts

## Your Dataset Format

Your emotion diary should be a CSV with these columns:
- `user_id`: Unique identifier for each participant
- `text_id`: Unique identifier for each entry
- `text`: The diary entry text
- `timestamp`: When the entry was made
- `valence` (optional): Pre-calculated valence score (-2 to 2 or 0 to 1 scale)
- `arousal` (optional): Pre-calculated arousal score (-2 to 2 or 0 to 1 scale)
- `dominance` (optional): Pre-calculated dominance score (-2 to 2 or 0 to 1 scale)

Example:
```csv
user_id,text_id,text,timestamp,valence,arousal
3,251,I've been feeling just fine...,2021-06-08 12:26:16,1.0,1.0
3,252,I've been feeling pretty good...,2021-06-09 13:41:40,0.0,1.0
```

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy scipy matplotlib seaborn --break-system-packages
```

### 2. Preprocess Your Data

```bash
# Edit preprocess_emotion_data.py to set your input CSV path
python preprocess_emotion_data.py
```

This creates:
- `processed_emotion_data/users/` - Individual user files
- `processed_emotion_data/text_corpus/` - Text files for lexicon analysis
- `processed_emotion_data/all_users_combined.csv` - Combined dataset

### 3. Generate Configurations

```bash
python generate_ued_configs.py
```

This creates several configs in `ued_configs/`:
- `config_existing_scores.json` - Use your valence/arousal values
- `config_lexicon_based.json` - Recalculate from text using NRC VAD
- `config_rapid_changes.json` - Analyze rapid emotional changes
- Multi-dimensional configs for valence, arousal, dominance

### 4. Run UED Analysis

```bash
# Using existing scores (recommended)
python run_custom_ued.py ued_configs/config_existing_scores.json

# Or using lexicon-based calculation
python run_custom_ued.py ued_configs/config_lexicon_based.json
```

Results saved to `ued_results/` containing:
- `ued_summary.csv` - Summary metrics for all users
- `individual_users/` - Per-user trajectories and metrics

### 5. Create Visualizations

```bash
python visualize_ued_results.py ued_results/existing_scores valence
```

Generates plots in `ued_results/existing_scores/visualizations/`:
- Individual emotion trajectories
- Comparison across users
- Metrics distributions
- Correlation heatmap

## Files Included

### Core Scripts

1. **`preprocess_emotion_data.py`**
   - Converts your CSV to EmotionDynamics format
   - Creates individual user files
   - Generates text corpus for analysis

2. **`generate_ued_configs.py`**
   - Creates configuration files
   - Supports multiple analysis scenarios
   - Customizable parameters

3. **`run_custom_ued.py`**
   - Runs UED analysis
   - Calculates emotion dynamics metrics
   - Supports both lexicon-based and existing scores

4. **`visualize_ued_results.py`**
   - Creates comprehensive visualizations
   - Generates trajectory plots
   - Produces comparison charts

### Documentation

- **`TUTORIAL.md`** - Step-by-step tutorial with examples
- **`ADAPTATION_GUIDE.md`** - Detailed adaptation guide
- **`README.md`** - This file

## UED Metrics Explained

The analysis calculates these metrics for each user:

### Basic Statistics
- **mean / median**: Average emotional state
- **range**: Span of emotions (max - min)
- **min_value / max_value**: Emotional extremes

### Variability
- **variability**: Standard deviation of emotions (stability/instability)
- **variance**: Squared deviation
- **entropy**: Unpredictability of emotional patterns

### Dynamics
- **home_base**: Most frequent emotional state (emotional default)
- **density**: Concentration of emotions (inverse of spread)
- **displacement**: Total emotional movement over time
- **avg_displacement**: Average movement per time point

### Change Rates
- **rise_rate**: Speed of positive emotional changes
- **max_rise**: Largest positive emotional jump
- **recovery_rate**: Speed of negative emotional changes
- **max_fall**: Largest negative emotional drop

## Advanced Usage

### Custom Window Size

Smaller windows capture rapid changes, larger windows smooth trends:

```python
# Edit config file
{
  "window": {
    "size": 20,    # Default: 50
    "step": 1      # Move 1 word at a time
  }
}
```

### Filter Users

Set minimum requirements:

```python
{
  "filters": {
    "min_tokens": 10,
    "min_entries_per_user": 5    # Only users with 5+ entries
  }
}
```

### Multi-Dimensional Analysis

Analyze all three dimensions:

```bash
python run_custom_ued.py ued_configs/ued_config_valence.json
python run_custom_ued.py ued_configs/ued_config_arousal.json
python run_custom_ued.py ued_configs/ued_config_dominance.json
```

## Comparing Approaches

### Option A: Use Your Existing Scores
**Pros:**
- Faster (no text processing needed)
- Uses your domain-specific emotion coding
- Respects your annotation decisions

**Cons:**
- Limited to entry-level granularity
- Can't capture within-entry dynamics

### Option B: Lexicon-Based Recalculation
**Pros:**
- Captures word-level emotion dynamics
- Uses standardized NRC VAD lexicon
- Enables rolling window analysis

**Cons:**
- Requires downloading lexicon
- May not match your coding scheme
- Generic word-level associations

**Recommendation:** Start with Option A for simplicity, then compare with Option B if you need finer-grained analysis.

## Example Output

### Console Output
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

Successfully analyzed 3 users
============================================================

=== Summary Statistics Across All Users ===
         mean    median  variability  displacement  entropy
count   3.000     3.000        3.000         3.000    3.000
mean    0.667     0.667        0.943         2.667    1.459
std     0.943     0.943        0.236         0.943    0.158
min    -0.200    -0.200        0.707         2.000    1.322
...
```

### Generated Files
```
ued_results/existing_scores/
├── ued_summary.csv                      # All users' metrics
├── individual_users/
│   ├── user_3_trajectory.csv           # User 3's emotion over time
│   ├── user_3_metrics.csv              # User 3's UED metrics
│   └── ...
└── visualizations/
    ├── user_3_trajectory.png           # User 3's trajectory plot
    ├── all_trajectories_comparison.png # All users compared
    ├── metrics_comparison.png          # Bar charts of metrics
    ├── metrics_correlation.png         # Correlation heatmap
    └── metrics_distributions.png       # Distribution histograms
```

## Troubleshooting

### "No user files found"
- Verify preprocessing completed successfully
- Check that `input_dir` in config matches output directory
- Ensure files are named like `user_3.json` or `user_3.csv`

### "Could not create emotion trajectory"
- User may have too few entries (check `min_entries_per_user`)
- Text may be too short (check `min_tokens`)
- Verify emotion scores are present in data

### "Lexicon not found"
- Download NRC VAD lexicon: http://saifmohammad.com/WebPages/nrc-vad.html
- Place in `EmotionDynamics/lexicons/` directory
- Or set `lexicon.path` to `null` in config to use existing scores

### Import Errors
```bash
# Ensure all packages are installed
pip install pandas numpy scipy matplotlib seaborn --break-system-packages
```

## Research Applications

This toolkit enables:

1. **Individual Differences**: Compare emotion dynamics across participants
2. **Longitudinal Analysis**: Track changes over time or interventions
3. **Group Comparisons**: Clinical vs. control, pre vs. post treatment
4. **Predictive Modeling**: Use UED metrics as features
5. **Correlation Studies**: Relate emotion dynamics to outcomes

## Citation

If you use this adaptation or the original EmotionDynamics framework:

```bibtex
@inproceedings{VM2022-TED,
  title={Tweet Emotion Dynamics: Emotion Word Usage in Tweets from US and Canada},
  author={Krishnapriya Vishnubhotla and Saif M. Mohammad},
  booktitle={Proceedings of the Thirteenth International Conference on 
             Language Resources and Evaluation (LREC 2022)},
  address={Marseille, France},
  year={2022}
}

@article{hipson2021emotion,
  doi={10.1371/journal.pone.0256153},
  author={Hipson, Will E. AND Mohammad, Saif M.},
  journal={PLOS ONE},
  title={Emotion dynamics in movie dialogues},
  year={2021},
  volume={16},
  pages={1-19}
}
```

## Support

For questions or issues:
1. Check the `TUTORIAL.md` for detailed walkthroughs
2. Review `ADAPTATION_GUIDE.md` for conceptual understanding
3. Consult the original [EmotionDynamics repository](https://github.com/Priya22/EmotionDynamics)
4. Original authors: Krishnapriya Vishnubhotla (vkpriya@cs.toronto.edu), 
   Saif M. Mohammad (saif.mohammad@nrc-cnrc.gc.ca)

## License

This adaptation follows the original EmotionDynamics repository's terms. 
Please respect the NRC VAD lexicon's terms of use when downloading and using it.

## Contributing

Improvements welcome! Consider:
- Additional visualization types
- Support for other emotion lexicons
- Integration with other emotion dynamics frameworks
- Performance optimizations for large datasets
