import math
import re
from pathlib import Path

import polars as pl

LEXICON_PATH = 'data/NRC-VAD-Lexicon-v2.1.txt'
DATASET_PATH = 'data/train_subtask2a.csv'
OUTPUT_PATH = 'output/baseline_lexicon_subtask2a.csv'
LOWER_BOUND = -0.3
UPPER_BOUND = 0.3

def prepare_lexicon():
    df = pl.read_csv(LEXICON_PATH, separator='\t', ignore_errors=True)
    if 'dominance' in df.columns:
        df.drop_in_place('dominance')
        

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
    val_valence = df_valence['valence'].to_list()
    valence_dict = dict(zip(key_valence, val_valence))
    
    key_arousal = df_arousal['term'].to_list()
    val_arousal = df_arousal['arousal'].to_list()
    arousal_dict = dict(zip(key_arousal, val_arousal))
    
    return valence_dict, arousal_dict

def prepare_dataset():
    df = pl.read_csv(DATASET_PATH)
    return df

def calculate_absolute_scores(dataset: pl.DataFrame, lexicon: dict, col_name: str):
    """
    Calculates the absolute lexicon score for each text row.
    Does not compute MSE yet.
    """
    scores = []
    
    regex = re.compile(r'[^A-Za-z ]')
    
    for row in dataset.iter_rows(named=True):
        text = str(row['text'])
        count = 0
        s = 0.0
        
        clean_text = regex.sub('', text)
        lexemes = clean_text.split(' ')
        
        for lexeme in lexemes:
            if lexeme:
                val = lexicon.get(lexeme, -1000.0)
                if val != -1000.0:
                    count += 1
                    s += val
        
        if count == 0:
            avg = 0.0
        else:
            avg = s / count
            
        scores.append(avg)

    return dataset.with_columns(pl.Series(f'{col_name}_pred_abs', scores, dtype=pl.Float64))

def calculate_state_change_metrics(dataset: pl.DataFrame, col_name: str):
    """
    Calculates predicted state change (Next - Current) and compares with ground truth.
    """
    target_col = f'state_change_{col_name}'
    pred_abs_col = f'{col_name}_pred_abs'
    
    dataset = dataset.sort(['user_id', 'timestamp'])

    dataset = dataset.with_columns(
        pl.col(pred_abs_col).shift(-1).over('user_id').alias(f'{col_name}_next_pred_abs')
    )
    
    dataset = dataset.with_columns(
        (pl.col(f'{col_name}_next_pred_abs') - pl.col(pred_abs_col)).alias(f'{col_name}_pred_change')
    )

    eval_df = dataset.drop_nulls(subset=[target_col, f'{col_name}_pred_change'])
    
    mse = eval_df.select(
        ((pl.col(target_col) - pl.col(f'{col_name}_pred_change')) ** 2).mean()
    ).item()
    
    rmse = math.sqrt(mse)
    
    print(f"=== Results for {col_name} (State Change) ===")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    return dataset

def main():
    print("Preparing Lexicon...")
    valence_dict, arousal_dict = prepare_lexicon()
    
    print(f"Loading Dataset: {DATASET_PATH}")
    dataset = prepare_dataset()
    
    print("Calculating absolute Valence scores...")
    dataset = calculate_absolute_scores(dataset, valence_dict, 'valence')
    
    print("Calculating absolute Arousal scores...")
    dataset = calculate_absolute_scores(dataset, arousal_dict, 'arousal')
    
    dataset = calculate_state_change_metrics(dataset, 'valence')
    dataset = calculate_state_change_metrics(dataset, 'arousal')
    
    full_path = Path(OUTPUT_PATH)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_csv(OUTPUT_PATH)
    print(f"Detailed predictions saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
