import argparse

import polars as pl
from sklearn.model_selection import GroupShuffleSplit

from prepare_datasets import DATASET_PATH, prepare_df


def parse_args():
    parser = argparse.ArgumentParser(
        prog='DistilBERT-BiLSTM for subtask 1, SemEval 2026 Task 2',
        description='Run the DistilBERT-BiLSTM for subtask 1 to predict Valence and/or Arousal score for sequence of texts',
    )
    parser.add_argument('-m', '--mode',
                        default='both',
                        type=str,
                        choices=['both', 'valence', 'arousal'],
                        help='train for valence, arousal, or both')
    parser.add_argument('-e', '--epoch',
                        default=1,
                        type=int,
                        help="Number of epochs for training")
    args = parser.parse_args()
    return args

def group_cross_validate(df: pl.DataFrame, y_name='valence'):
    y = df[y_name]
    X = df.drop(['valence', 'arousal'])
    groups = df['user_id']
    # Split to 6/4 for train / (dev + test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_idx, dev_test_idx = next(gss1.split(X, y, groups))
    train_users = df.with_row_index().filter(pl.col('index').is_in(train_idx)).drop('index')
    print(train_users['text_id'].unique())
    dev_test_df = df.with_row_index().filter(pl.col('index').is_in(dev_test_idx)).drop('index')
    y = dev_test_df[y_name]
    X = dev_test_df.drop(['valence', 'arousal'])
    groups = dev_test_df['user_id']
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    dev_idx, test_idx = next(gss2.split(X, y, groups))
    dev_users = dev_test_df.with_row_index().filter(pl.col('index').is_in(dev_idx)).drop('index')
    test_users = dev_test_df.with_row_index().filter(pl.col('index').is_in(test_idx)).drop('index')
    print(dev_users['text_id'].unique())
    print(test_users['text_id'].unique())


def main():
    args = parse_args()
    df = prepare_df(DATASET_PATH)
    if args.mode in ['both', 'valence']:
        group_cross_validate(df, y_name='valence')

if __name__ == '__main__':
    main()
