from pathlib import Path

import polars as pl

DATASET_PATH = 'data/train_subtask1.csv'
STATS_PATH = 'statistics/train_subtask1_group_by_users_len_sorted_asc.csv'

def prepare_df(dataset_path: str):
    df = pl.read_csv(dataset_path, try_parse_dates=True)
    return df

def display_stats(df: pl.DataFrame):
    df_by_users = df.group_by(pl.col("user_id")).len().sort(by=pl.col('len'))
    stats_path = Path(STATS_PATH)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    df_by_users.write_csv(stats_path)
    result = df.select(pl.count('text_id')).item()
    print(result)


def main():
    df = prepare_df(DATASET_PATH)
    display_stats(df)

if __name__ == '__main__':
    main()
