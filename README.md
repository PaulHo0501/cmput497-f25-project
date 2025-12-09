# SemEval Task 2: Predicting Valence-Arousal Score for Longitudinal Texts

## Installation

**Requirement:** Python 3.11

Initialize a virtual environment and install dependencies:
```bash
python3.11 -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

## Running the Code

### Baseline Lexicon

Calculate lexicon averages for all tasks:
```bash
python src/baseline.py
```

### DistilBERT-BiLSTM Model

Train the DistilBERT-BiLSTM baseline model:
```bash
python src/distilBERT_BiLSTM.py --task 1 --target valence --epochs 1
```

### Large Language Models (LLM)

**Basic LLM prompt:**
```bash
python src/prompt_llm.py --user --debug
```

**Task 2a - Indirect approach:**
```bash
python src/2a_prompt_llm_avg.py --debug
```

**Task 2a - Direct approach:**
```bash
python src/2a_prompt_llm_change.py --debug
```

**Task 2b - Indirect approach:**
```bash
python src/2b_prompt_llm_avg.py --debug
```

**Task 2b - Direct approach:**
```bash
python src/2b_prompt_llm_change.py --debug
```

**Evaluate all LLM baseline models:**
```bash
python src/eval_llm.py
```

### Temporal Fusion Transformer (TFT)

Train and evaluate TFT models for Task 2a:
```bash
python src/tft_subtask2a.py --target both --epoch 1
```

### SentiSynset

Train and evaluate SentiSynset models for Task 2a:
```bash
python src/sentisynset.py --task 2a --input data/train_subtask2a.csv --xml data/sentisynset_lexicon.xml --output output/sentisynset_preds.csv
```