import polars as pl
import re
import math
from pathlib import Path

LEXICON_PATH = 'data/NRC-VAD-Lexicon-v2.1.txt'
DATASET_PATH = 'data/train_subtask1.csv'
BASELINE_PATH = 'output/baseline_lexicon_broader.csv'
LOWER_BOUND = -0.1
UPPER_BOUND = 0.1

def prepare_lexicon():
    df = pl.read_csv(LEXICON_PATH, separator='\t')
    # Cut the dominance column
    df.drop_in_place('dominance')
    # Significant valence and arousal
    df_valence = df.drop('arousal').filter(
            (pl.col('valence') < LOWER_BOUND) | (pl.col('valence') > UPPER_BOUND)
    )
    df_arousal = df.drop('valence').filter(
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
    if col_name == 'valence':
        # [-2, 2] to [-1, 1]
        return (col / 2.0).alias(f'{col_name}_normalized')
    else:
        # [0, 2] to [-1, 1]
        return (col - 1.0).alias(f'{col_name}_normalized')


def prepare_dataset():
    df = pl.read_csv(DATASET_PATH)
    df = df.drop(['timestamp', 'collection_phase', 'is_words'])
    df_normalized = df.with_columns(
        normalize_column('valence'),
        normalize_column('arousal')
    )
    return df_normalized

def calculate_average(dataset: pl.DataFrame, lexicon: dict, col_name: str = 'valence'):
    col_baseline = []
    mse = 0.0
    for row in dataset.iter_rows(named=True):
        text = row['text']
        text = text.lower()
        count = 0
        s = 0.0
        lexemes = re.sub(r'[^A-Za-z ]', '', text).split(' ')
        for lexeme in lexemes:
            if lexeme:
                filtered_lexicon_entry = lexicon.get(lexeme, None)
                if filtered_lexicon_entry == None:
                    continue
                else:
                    count += 1
                    s += filtered_lexicon_entry
        if (count == 0):
            avg = 0.0
        else:
            # Raw average [-1, 1]
            avg = s/count
        if col_name == 'valence':
            final_avg = avg * 2
        else:
            final_avg = avg + 1.0
        mse += (row[f'{col_name}'] - final_avg) ** 2
        col_baseline.append(final_avg)

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
