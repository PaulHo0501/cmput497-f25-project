import math
import re
from pathlib import Path

import polars as pl

LEXICON_PATH = 'data/NRC-VAD-Lexicon-v2.1.txt'
DATASET_PATH = 'data/train_subtask1.csv'
BASELINE_PATH = 'output/baseline_lexicon.csv'
LOWER_BOUND = -0.3
UPPER_BOUND = 0.3

def prepare_lexicon():
    df = pl.read_csv(LEXICON_PATH, separator='\t')
    # Cut the dominance column
    df.drop_in_place('dominance')
    # Significant valence and arousal
    df_scaled = df.with_columns([
        (pl.col("valence") * 2).alias("valence"),
        (pl.col("arousal") * 2).alias("arousal")
    ])
    df_valence = df_scaled.drop('arousal').filter(
            (pl.col('valence') < LOWER_BOUND) | (pl.col('valence') > UPPER_BOUND)
    )
    df_arousal = df_scaled.drop('valence').filter(
            (pl.col('arousal') < LOWER_BOUND) | (pl.col('arousal') > UPPER_BOUND)
    )
    
    key_valence = df_valence['term'].to_list()
    key_arousal = df_arousal['term'].to_list()
    val_valence = df_valence['valence'].to_list()
    val_arousal = df_arousal['arousal'].to_list()
    valence_dict = dict(zip(key_valence, val_valence))
    arousal_dict = dict(zip(key_arousal, val_arousal))
    return valence_dict, arousal_dict

def normalize_column(col_name: str):
    col = pl.col(col_name)
    return (-1.0 + ((col + 2)*2) / 4).alias(f'{col_name}_normalized')


def prepare_dataset():
    df = pl.read_csv(DATASET_PATH)
    df = df.drop(['timestamp', 'collection_phase', 'is_words'])
    return df

def calculate_average(dataset: pl.DataFrame, lexicon: dict, col_name: str = 'valence'):
    col_baseline = []
    mse = 0.0
    for row in dataset.iter_rows(named=True):
        text = row['text']
        count = 0
        s = 0.0
        lexemes = re.sub(r'[^A-Za-z ]', '', text).split(' ')
        for lexeme in lexemes:
            if lexeme:
                filtered_lexicon_entry = lexicon.get(lexeme, -1000.0)
                if filtered_lexicon_entry == -1000.0:
                    continue
                else:
                    count += 1
                    s += filtered_lexicon_entry
        if (count == 0):
            avg = 0.0
        else:
            avg = s/count
        true_val = row[f"{col_name}"]
        mse += (true_val - avg) * (true_val - avg)
        col_baseline.append(avg)

    mse = mse/dataset.shape[0]
    print(f"MSE {col_name}: {mse}")
    print(f"RMSE {col_name}: {math.sqrt(mse)}")
    dataset = dataset.with_columns(pl.Series(f'{col_name}_baseline', col_baseline, dtype=pl.Float64))
    return dataset



def main():
    df_valence, df_arousal = prepare_lexicon()
    dataset = prepare_dataset()
    dataset = calculate_average(dataset, df_valence, 'valence')
    dataset = calculate_average(dataset, df_arousal, 'arousal')
    full_path = Path(BASELINE_PATH)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_csv(BASELINE_PATH)
    print(dataset)
    print("Done")

if __name__ == "__main__":
    print("Baseline")
    main()
