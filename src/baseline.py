import math
import re
from pathlib import Path

import polars as pl

LEXICON_PATH = 'data/NRC-VAD-Lexicon-v2.1.txt'

PATH_TRAIN_1 = 'data/train_subtask1.csv'
PATH_TRAIN_2A = 'data/train_subtask2a.csv'
PATH_TRAIN_2B = 'data/train_subtask2b.csv'

OUT_PATH_1 = 'output/baseline_subtask1.csv'
OUT_PATH_2A = 'output/baseline_subtask2a.csv'
OUT_PATH_2B = 'output/baseline_subtask2b.csv'

LOWER_BOUND = -0.1
UPPER_BOUND = 0.1

def prepare_lexicon():
    df = pl.read_csv(LEXICON_PATH, separator='\t')
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
    key_arousal = df_arousal['term'].to_list()
    val_valence = df_valence['valence'].to_list()
    val_arousal = df_arousal['arousal'].to_list()
    
    valence_dict = dict(zip(key_valence, val_valence))
    arousal_dict = dict(zip(key_arousal, val_arousal))
    return valence_dict, arousal_dict

def load_dataset(path: str):
    print(f"Loading {path}...")
    df = pl.read_csv(path)
    if "timestamp" in df.columns:
        df = df.with_columns(pl.col("timestamp").str.to_datetime())
    return df

def apply_lexicon_scoring(dataset: pl.DataFrame, lexicon: dict, col_name: str = 'valence'):
    print(f"Applying lexicon for {col_name}...")
    col_baseline = []
    
    target_col_exists = col_name in dataset.columns
    mse = 0.0
    
    def scale_score(raw_avg, c_name):
        if c_name == 'valence': return raw_avg * 2
        else: return raw_avg + 1.0

    for row in dataset.iter_rows(named=True):
        text = row.get('text', "")
        if text is None: text = ""
        text = text.lower()
        
        lexemes = re.sub(r'[^A-Za-z ]', '', text).split(' ')
        count = 0
        s = 0.0
        for lexeme in lexemes:
            if lexeme:
                score = lexicon.get(lexeme)
                if score is not None:
                    count += 1
                    s += score
        
        if count == 0: avg = 0.0
        else: avg = s / count
            
        final_avg = scale_score(avg, col_name)
        col_baseline.append(final_avg)
        
        if target_col_exists and row[col_name] is not None:
            mse += (row[col_name] - final_avg) ** 2
            
    if target_col_exists and dataset.shape[0] > 0:
        mse = mse / dataset.shape[0]
        print(f"Subtask 1: MSE {col_name}: {mse:.4f} (RMSE: {math.sqrt(mse):.4f})")

    return dataset.with_columns(pl.Series(f'{col_name}_baseline', col_baseline, dtype=pl.Float64))


def process_subtask_2a(df: pl.DataFrame):
    """
    Calculates State Change and MSE if targets exist.
    """
    df = df.sort(["user_id", "timestamp"])
    
    df = df.with_columns([
        (pl.col("valence_baseline").shift(-1).over("user_id") - pl.col("valence_baseline"))
        .alias("state_change_valence_pred"),
        
        (pl.col("arousal_baseline").shift(-1).over("user_id") - pl.col("arousal_baseline"))
        .alias("state_change_arousal_pred")
    ])
    if "state_change_valence" in df.columns:
        valid_rows = df.drop_nulls(subset=["state_change_valence", "state_change_valence_pred"])
        
        if valid_rows.height > 0:
            stats = valid_rows.select([
                (pl.col("state_change_valence") - pl.col("state_change_valence_pred"))
                .pow(2).mean().alias("mse"),
                
                (pl.col("state_change_valence") - pl.col("state_change_valence_pred"))
                .pow(2).mean().sqrt().alias("rmse")
            ])
            
            mse_v = stats["mse"].item()
            rmse_v = stats["rmse"].item()
            
            if mse_v is not None:
                print(f"Subtask 2A Valence - MSE: {mse_v:.4f} | RMSE: {rmse_v:.4f}")
        else:
            print("Subtask 2A Valence: N/A (No valid rows for comparison)")

    if "state_change_arousal" in df.columns:
        valid_rows = df.drop_nulls(subset=["state_change_arousal", "state_change_arousal_pred"])
        
        if valid_rows.height > 0:
            stats = valid_rows.select([
                (pl.col("state_change_arousal") - pl.col("state_change_arousal_pred"))
                .pow(2).mean().alias("mse"),
                
                (pl.col("state_change_arousal") - pl.col("state_change_arousal_pred"))
                .pow(2).mean().sqrt().alias("rmse")
            ])
            
            mse_a = stats["mse"].item()
            rmse_a = stats["rmse"].item()
            
            if mse_a is not None:
                print(f"Subtask 2A Arousal - MSE: {mse_a:.4f} | RMSE: {rmse_a:.4f}")
        else:
            print("Subtask 2A Arousal: N/A (No valid rows for comparison)")

    return df

def process_subtask_2b(df: pl.DataFrame):
    """
    Calculates Disposition Change and MSE if targets exist.
    """
    df = df.sort(["user_id", "timestamp"])

    if "group" not in df.columns:
        df = df.with_columns([
            pl.col("timestamp").rank("ordinal").over("user_id").alias("rank"),
            pl.count("text_id").over("user_id").alias("total_count")
        ])
        df = df.with_columns(
            pl.when(pl.col("rank") <= (pl.col("total_count") / 2))
            .then(1)
            .otherwise(2)
            .alias("group")
        )
    
    aggs = [
        pl.col("valence_baseline").mean().alias("mean_val"),
        pl.col("arousal_baseline").mean().alias("mean_ar")
    ]
    
    has_truth = "disposition_change_valence" in df.columns
    if has_truth:
        aggs.append(pl.col("disposition_change_valence").first().alias("truth_val"))
        aggs.append(pl.col("disposition_change_arousal").first().alias("truth_ar"))

    agg = df.group_by(["user_id", "group"]).agg(aggs)
    
    g1 = agg.filter(pl.col("group") == 1)
    g2 = agg.filter(pl.col("group") == 2)
    
    final = g1.join(g2, on="user_id", suffix="_g2")
    
    final = final.with_columns([
        (pl.col("mean_val_g2") - pl.col("mean_val")).alias("disposition_change_valence_pred"),
        (pl.col("mean_ar_g2") - pl.col("mean_ar")).alias("disposition_change_arousal_pred")
    ])

    if has_truth:
        metrics = final.select([
            (pl.col("truth_val") - pl.col("disposition_change_valence_pred"))
                .pow(2).mean().sqrt().alias("rmse_valence"),
            
            (pl.col("truth_val") - pl.col("disposition_change_valence_pred"))
                .pow(2).mean().alias("mse_valence"),

            (pl.col("truth_ar") - pl.col("disposition_change_arousal_pred"))
                .pow(2).mean().sqrt().alias("rmse_arousal"),
            
            (pl.col("truth_ar") - pl.col("disposition_change_arousal_pred"))
                .pow(2).mean().alias("mse_arousal")
        ])
        
        rmse_v = metrics["rmse_valence"].item()
        mse_v = metrics["mse_valence"].item()
        rmse_a = metrics["rmse_arousal"].item()
        mse_a = metrics["mse_arousal"].item()
        if mse_v != None:
            print(f"Subtask 2B MSE Valence Change: {mse_v:.4f} (RMSE: {rmse_v:.4f})")
        if mse_a != None:
            print(f"Subtask 2B MSE Arousal Change: {mse_a:.4f} (RMSE: {rmse_a:.4f})")

    final = final.select([
        "user_id", 
        "disposition_change_valence_pred", 
        "disposition_change_arousal_pred"
    ])
    
    return final

def main():
    print("Preparing Lexicon")
    val_dict, ar_dict = prepare_lexicon()
    
    print("Processing Subtask 1")
    if Path(PATH_TRAIN_1).exists():
        df_1 = load_dataset(PATH_TRAIN_1)
        df_1 = apply_lexicon_scoring(df_1, val_dict, 'valence')
        df_1 = apply_lexicon_scoring(df_1, ar_dict, 'arousal')
        Path(OUT_PATH_1).parent.mkdir(parents=True, exist_ok=True)
        df_1.write_csv(OUT_PATH_1)

    print("Processing Subtask 2A")
    if Path(PATH_TRAIN_2A).exists():
        df_2a = load_dataset(PATH_TRAIN_2A)
        df_2a = apply_lexicon_scoring(df_2a, val_dict, 'valence')
        df_2a = apply_lexicon_scoring(df_2a, ar_dict, 'arousal')
        df_2a = process_subtask_2a(df_2a)
        
        out_2a = df_2a.select(["user_id", "text_id", "timestamp", "state_change_valence_pred", "state_change_arousal_pred"])
        Path(OUT_PATH_2A).parent.mkdir(parents=True, exist_ok=True)
        out_2a.write_csv(OUT_PATH_2A)

    print("Processing Subtask 2B")
    if Path(PATH_TRAIN_2B).exists():
        df_2b = load_dataset(PATH_TRAIN_2B)
        df_2b = apply_lexicon_scoring(df_2b, val_dict, 'valence')
        df_2b = apply_lexicon_scoring(df_2b, ar_dict, 'arousal')
        out_2b = process_subtask_2b(df_2b)
        
        Path(OUT_PATH_2B).parent.mkdir(parents=True, exist_ok=True)
        out_2b.write_csv(OUT_PATH_2B)
        
    print("All Done.")

if __name__ == "__main__":
    main()

