# SemEval Task 2: Predicting Valence-Arousal Score for Longitudinal Texts

## Installation

Requirement: **Python 3.11**

Run the following scripts to initialize a virtual environment and install the packages

```[lang=Bash]
python3.11 -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

Then change the file named `.env.example` to `.env` to populate the environment variable.

Lastly, modify the environment variable with the API keys you obtain from these
services to use our code.

The list of services to obtain the API from:

1. DeepL
2. Gemini

## Running the code

Run the following command to have the baseline lexicon average:

```[lang=Bash]
python src/baseline.py
```

Run the following command to train the baseline model by DistilBERT-BiLSTM:

```[lang=Bash]
python src/distilBERT_BiLSTM.py -m both -e 10
```
