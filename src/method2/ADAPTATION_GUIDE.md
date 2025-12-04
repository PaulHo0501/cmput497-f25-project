# Adapting EmotionDynamics for Your Custom Dataset

## Overview
This guide helps you adapt the EmotionDynamics framework to work with your custom emotion diary dataset. Your dataset has the advantage of already containing user IDs, timestamps, and text content - perfect for emotion dynamics analysis!

## Your Dataset Structure
Your CSV contains:
- `user_id`: Identifier for each user
- `text_id`: Unique text entry ID
- `text`: The actual text content
- `timestamp`: When the entry was made
- `collection_phase`: Phase identifier
- `is_words`: Boolean indicating if entry is just words
- `valence`: Pre-calculated valence score
- `arousal`: Pre-calculated arousal score

## Key Differences from Tweet Dataset

### 1. Data Format
- **Original**: Tweet IDs in separate files by location/month
- **Your Data**: Single CSV with all user entries and timestamps
- **Adaptation**: Need to preprocess your CSV into the format expected by the UED library

### 2. Emotion Scores
- **Original**: Calculates emotion scores from text using NRC VAD lexicon
- **Your Data**: Already has valence and arousal scores
- **Adaptation**: You can either:
  - Use your existing scores
  - Recalculate from text using NRC VAD lexicon for consistency
  - Compare both approaches

## Required Steps

### Step 1: Clone the Repository
```bash
git clone https://github.com/Priya22/EmotionDynamics.git
cd EmotionDynamics
```

### Step 2: Install Dependencies
```bash
pip install pandas numpy scipy --break-system-packages
```

### Step 3: Download NRC VAD Lexicon (Optional)
If you want to recalculate emotion scores from text:
1. Visit http://saifmohammad.com/WebPages/nrc-vad.html
2. Download the lexicon files
3. Place them in the `lexicons/` folder

### Step 4: Prepare Your Data
Your data needs to be formatted for the UED library. The expected format is:
- One file per user (or a way to filter by user)
- Text sorted by timestamp
- Emotion scores calculated or provided

### Step 5: Modify the Configuration
The UED library uses a config file. You'll need to create one for your dataset.

### Step 6: Run the Analysis
Execute the UED analysis using your modified configuration.

## What UED Metrics Will You Get?

The framework calculates various metrics including:

1. **Home Base**: The most frequent emotional state
2. **Variability**: How much emotions fluctuate
3. **Density**: How concentrated emotions are in the emotional space
4. **Rise Rate**: How quickly emotions escalate
5. **Recovery**: How quickly emotions return to baseline
6. **Displacement**: Overall movement in emotional space
7. **Mean/Median Values**: Average emotional states
8. **Entropy**: Unpredictability of emotional states

## Next Steps

See the accompanying Python scripts for:
1. Data preprocessing
2. Configuration file generation
3. Running the UED analysis
4. Visualizing results
