# SemEval Task 2: Predicting Valence-Arousal Score for Longitudinal Texts

## Installation

Requirement: **Python 3.11**

Run the following scripts to initialize a virtual environment and install the packages

```[lang=Bash]
python3.11 -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```


## Running the code

Run the following command to have the baseline lexicon average:

```[lang=Bash]
python src/baseline.py
```

Run the following command to train the baseline model by DistilBERT-BiLSTM:

```[lang=Bash]
python src/distilBERT_BiLSTM.py -m both -e 10
```
