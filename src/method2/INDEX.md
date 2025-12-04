# EmotionDynamics Adapter - Getting Started

Welcome! This package helps you run Emotion Dynamics analysis on your custom emotion diary data.

## 📁 What's in This Package?

### 📘 Documentation
- **README.md** - Complete overview, usage guide, and troubleshooting
- **TUTORIAL.md** - Step-by-step walkthrough with detailed examples
- **ADAPTATION_GUIDE.md** - How the adaptation works conceptually
- **QUICK_REFERENCE.md** - Cheat sheet for quick commands
- **INDEX.md** - This file

### 🐍 Python Scripts
- **preprocess_emotion_data.py** - Convert your CSV to analysis format
- **generate_ued_configs.py** - Create configuration files
- **run_custom_ued.py** - Run UED analysis
- **visualize_ued_results.py** - Create visualizations
- **run_workflow.py** - Master script to run everything

### 📊 Sample Data
- **sample_emotion_diary.csv** - Example dataset in the expected format

### 📦 Other Files
- **requirements.txt** - Python package dependencies

## 🚀 Quick Start (3 Minutes)

### 1. Install Dependencies (30 seconds)
```bash
pip install -r requirements.txt
```

### 2. Test with Sample Data (2 minutes)
```bash
# Option A: Run complete workflow automatically
python run_workflow.py --input sample_emotion_diary.csv --full

# Option B: Run step-by-step
python preprocess_emotion_data.py  # Edit line 202 first: input_csv = "sample_emotion_diary.csv"
python generate_ued_configs.py
python run_custom_ued.py ued_configs/config_existing_scores.json
python visualize_ued_results.py ued_results/existing_scores valence
```

### 3. Use Your Own Data
Replace `sample_emotion_diary.csv` with your CSV file and run again!

## 📚 Where to Start?

### If you want to...

**Jump right in:**
→ Start with QUICK_REFERENCE.md

**Understand the concepts first:**
→ Read ADAPTATION_GUIDE.md

**Follow a detailed guide:**
→ Work through TUTORIAL.md

**Get comprehensive information:**
→ Read README.md

**Just run it:**
→ Use run_workflow.py with --full flag

## 🎯 Your Data Format

Your CSV should have:
- **user_id** - Unique identifier for each person
- **text_id** - Unique identifier for each entry
- **text** - The diary entry text
- **timestamp** - When it was written
- **valence** (optional) - Emotion score for valence
- **arousal** (optional) - Emotion score for arousal

Example:
```csv
user_id,text_id,text,timestamp,valence,arousal
3,251,I've been feeling just fine...,2021-06-08 12:26:16,1.0,1.0
3,252,Pretty good day today...,2021-06-09 13:41:40,0.0,1.0
```

## 🔍 What You'll Get

### Results Structure
```
ued_results/existing_scores/
├── ued_summary.csv                  # Metrics for all users
├── individual_users/
│   ├── user_3_trajectory.csv       # Emotion over time
│   ├── user_3_metrics.csv          # UED metrics
│   └── ...
└── visualizations/
    ├── user_3_trajectory.png       # Individual plots
    ├── all_trajectories_comparison.png
    ├── metrics_comparison.png
    └── ...
```

### Key Metrics
- **mean** - Average emotional state
- **variability** - How much emotions fluctuate
- **home_base** - Most common emotional state
- **displacement** - Total emotional movement
- **rise_rate** - Speed of emotional escalations
- **recovery_rate** - Speed of emotional returns
- **entropy** - Unpredictability of emotions

## 🔧 Common Workflows

### Basic Analysis (Using Your Scores)
```bash
python run_workflow.py --input your_data.csv --full
```

### Multi-Dimensional Analysis
```bash
# Analyze all three dimensions
python run_workflow.py --input your_data.csv --full --dimension valence
python run_workflow.py --input your_data.csv --full --dimension arousal
python run_workflow.py --input your_data.csv --full --dimension dominance
```

### Using NRC VAD Lexicon
```bash
# First download lexicon from: http://saifmohammad.com/WebPages/nrc-vad.html
python run_workflow.py --input your_data.csv --full --use-lexicon
```

### Step-by-Step (Manual Control)
```bash
# Step 1: Preprocess
python preprocess_emotion_data.py

# Step 2: Configure
python generate_ued_configs.py

# Step 3: Analyze
python run_custom_ued.py ued_configs/config_existing_scores.json

# Step 4: Visualize
python visualize_ued_results.py ued_results/existing_scores valence
```

## ⚠️ Troubleshooting

### Dependencies Not Installed
```bash
pip install pandas numpy scipy matplotlib seaborn
```

### No User Files Found
- Check that preprocessing ran successfully
- Verify files are in `processed_emotion_data/users/`

### Can't Create Trajectory
- User has too few entries (need at least 3)
- Lower `min_entries_per_user` in config file

### Missing Lexicon
- Download from http://saifmohammad.com/WebPages/nrc-vad.html
- Or use existing scores by setting lexicon path to null

## 📖 Learning Path

1. **Beginner** → QUICK_REFERENCE.md (5 min)
   - Quick commands and file structure
   
2. **Intermediate** → TUTORIAL.md (15 min)
   - Step-by-step with examples
   
3. **Advanced** → README.md (30 min)
   - Complete documentation
   
4. **Theory** → ADAPTATION_GUIDE.md (10 min)
   - Understanding the framework

## 💡 Tips

- Start with sample data to test the workflow
- Use existing scores first (faster, simpler)
- Try lexicon-based analysis for comparison
- Adjust window size in configs for different granularity
- Check visualizations to understand your data
- Use run_workflow.py for automated processing

## 🔗 Resources

- **Original EmotionDynamics**: https://github.com/Priya22/EmotionDynamics
- **NRC VAD Lexicon**: http://saifmohammad.com/WebPages/nrc-vad.html
- **Hipson & Mohammad (2021)**: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0256153

## 📧 Getting Help

1. Check troubleshooting sections in README.md and TUTORIAL.md
2. Review error messages carefully
3. Verify your data format matches examples
4. Contact original authors:
   - Krishnapriya Vishnubhotla: vkpriya@cs.toronto.edu
   - Saif M. Mohammad: saif.mohammad@nrc-cnrc.gc.ca

## ✅ Checklist

Before running analysis:
- [ ] Dependencies installed
- [ ] Input CSV in correct format
- [ ] Preprocessed data created
- [ ] Config files generated
- [ ] Ready to analyze!

## 🎉 Ready to Begin!

Choose your path:
- **Quick start**: Run `python run_workflow.py --input sample_emotion_diary.csv --full`
- **Step-by-step**: Open TUTORIAL.md
- **Reference**: Keep QUICK_REFERENCE.md handy

Good luck with your emotion dynamics analysis! 🚀
