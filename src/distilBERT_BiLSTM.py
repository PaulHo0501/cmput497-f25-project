import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, ShuffleSplit
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import DistilBertModel, DistilBertTokenizer

from prepare_datasets import DATASET_PATH, prepare_df

MODEL_NAME = "distilbert-base-uncased"
MAX_SEQUENCE_LENGTH = 256
MAX_TOKEN_LENGTH = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_SIZE = 0.2
RANDOM_STATE = 51
DISTILBERT_HIDDEN_DIM = 768
BATCH_SIZE = 8
MODEL_PATH = 'models/distilbert-bilstm_{}.pt'

class CustomDataset(Dataset):
    def __init__(self, original_df: pl.DataFrame, y_name: str):
        self.processed_df: pl.DataFrame = original_df.with_columns(((pl.col(y_name) + 2) / 4).alias(f'{y_name}_normalized')).group_by('user_id').agg(
            pl.col('text_id').implode().alias('text_ids'),
            pl.col('text').implode().alias('texts'),
            pl.col(f'{y_name}').implode().alias(f"{y_name}s"),
            pl.col(f'{y_name}_normalized').implode().alias(f"scores"),
        )
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        embedder_instance = DistilBertModel.from_pretrained(MODEL_NAME, device_map=DEVICE)
        self.embedder = embedder_instance
        self.embedder.eval()

    def __len__(self):
        return self.processed_df.height

    def __getitem__(self, index):
        row: dict = self.processed_df.row(index, named=True)
        text_embeddings = self.encode_texts(row['texts'])
        pad_len = MAX_SEQUENCE_LENGTH - text_embeddings.size(0)
        padding_tensor = torch.zeros(pad_len, text_embeddings.size(1), dtype=torch.float32)
        padded_embeddings = torch.cat([text_embeddings, padding_tensor], dim=0)
        score_tensor = torch.tensor(row['scores'], dtype=torch.float32)
        score_padding = torch.full((pad_len,), -1.0, dtype=torch.float32)
        padded_scores = torch.cat([score_tensor, score_padding], dim = 0)
        mask = torch.cat([torch.ones(len(row['scores']), dtype=torch.bool), torch.zeros(pad_len, dtype=torch.bool)])
        return {
            "embeddings": padded_embeddings,
            "targets": padded_scores,
            "mask": mask,
            'user_id': row['user_id']
        }

    def get_unique_user_ids(self):
        return self.processed_df['user_id'].unique()

    @torch.no_grad()
    def encode_texts(self, texts):
        embeddings_list = []
        for text in texts:
            encoded_input = self.tokenizer(text, 
                                          padding='max_length', 
                                          truncation=True, 
                                          max_length=MAX_TOKEN_LENGTH,
                                          return_tensors='pt')
            input_ids = encoded_input['input_ids'].to(DEVICE)
            attention_mask = encoded_input['attention_mask'].to(DEVICE)
            output = self.embedder(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = output.last_hidden_state[:, 0, :].squeeze(0).cpu()
            embeddings_list.append(cls_embedding)
        return torch.stack(embeddings_list)

class BiLSTMRegressor(nn.Module):
    def __init__(self, lstm_hidden_dim=256, num_lstm_layers=1, dropout_rate=0.3):
        super(BiLSTMRegressor, self).__init__()
        self.lstm_hidden_dim = lstm_hidden_dim
        self.bilstm = nn.LSTM(
            input_size=DISTILBERT_HIDDEN_DIM,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(lstm_hidden_dim * 2, 1)

    def forward(self, batched_text_embeddings):
        lstm_output, _ = self.bilstm(batched_text_embeddings)
        x = self.dropout(lstm_output)
        scores = self.regressor(x)
        return scores.squeeze(-1)

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
    parser.add_argument('-e', '--epochs',
                        default=1,
                        type=int,
                        help="Number of epochs for training")
    args = parser.parse_args()
    return args

def get_data_loaders(dataset, indices, batch_size, shuffle=True):
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
    return loader
# Assuming all imports and class definitions (CustomDataset, BiLSTMRegressor) are as provided above.
# Define constants needed for denormalization:
VALENCE_MIN = -2.0
VALENCE_MAX = 2.0
VALENCE_RANGE = VALENCE_MAX - VALENCE_MIN # 4.0

# Define a function to calculate metrics
def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """Calculates RMSE and Accuracy after denormalization and rounding."""
    
    # 1. Denormalize
    pred_denormalized = predictions * VALENCE_RANGE + VALENCE_MIN
    target_denormalized = targets * VALENCE_RANGE + VALENCE_MIN
    
    # 2. Final discrete prediction (Rounding and Clipping)
    final_pred_scores = np.round(pred_denormalized).clip(VALENCE_MIN, VALENCE_MAX)
    
    # 3. Calculate Metrics
    # RMSE
    mse = np.mean((target_denormalized - final_pred_scores) ** 2, dtype=float)
    rmse = np.sqrt(mse)
    
    return mse, rmse

def evaluate_model(model, dataloader, criterion, y_name, fold_num=None):
    model.eval()
    total_loss = 0.0
    all_true_scores = []
    all_pred_scores = []
    
    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch['embeddings'].to(DEVICE)
            normalized_targets = batch['targets'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            
            # Forward pass
            normalized_predictions = model(embeddings)
            
            # Masked Loss Calculation
            flat_predictions = normalized_predictions.flatten()[mask.flatten()]
            flat_targets = normalized_targets.flatten()[mask.flatten()]
            
            loss = criterion(flat_predictions, flat_targets)
            total_loss += loss.item()
            
            # Collect data for metrics
            all_pred_scores.extend(flat_predictions.cpu().numpy())
            all_true_scores.extend(flat_targets.cpu().numpy())
            
    # Convert lists to NumPy arrays
    all_pred_scores_np = np.array(all_pred_scores)
    all_true_scores_np = np.array(all_true_scores)

    # Calculate final metrics
    mse, rmse = calculate_metrics(all_pred_scores_np, all_true_scores_np)
    avg_loss = total_loss / len(dataloader)
    
    print(f"{y_name.capitalize()} Fold: {fold_num} - Dev Loss: {avg_loss:.4f} | RMSE: {rmse:.4f} | MSE: {mse:.2f}")
        
    return {'val_loss': avg_loss, 'rmse': rmse, 'mse': mse}

def group_cross_validate(full_dataset: CustomDataset, y_name='valence', num_epochs=1):
    all_indices = np.arange(len(full_dataset))
    ss = ShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_dev_indices, test_indices = next(ss.split(all_indices))
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    all_fold_metrics = []
    criterion = nn.MSELoss()
    for fold, (train_subset_indices, dev_subset_indices) in enumerate(kf.split(train_dev_indices)):
        train_indices = train_dev_indices[train_subset_indices]
        dev_indices = train_dev_indices[dev_subset_indices]
        print(f"{'*'*20} Fold {fold + 1}/5 {'*'*20}")
        model = BiLSTMRegressor().to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        train_loader = get_data_loaders(full_dataset, train_indices, BATCH_SIZE, shuffle=True)
        dev_loader = get_data_loaders(full_dataset, dev_indices, BATCH_SIZE, shuffle=True)
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                embeddings = batch['embeddings'].to(DEVICE)
                normalized_targets = batch['targets'].to(DEVICE)
                mask = batch['mask'].to(DEVICE)
                optimizer.zero_grad()
                normalized_predictions = model(embeddings)
                flat_predictions = normalized_predictions.flatten()[mask.flatten()]
                flat_targets = normalized_targets.flatten()[mask.flatten()]
                loss = criterion(flat_predictions, flat_targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_train_loss = epoch_loss / len(train_loader)
            print(f"Fold {fold+1} - Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")
        fold_metrics = evaluate_model(model, dev_loader, criterion, y_name=y_name, fold_num=fold+1)
        all_fold_metrics.append(fold_metrics)
    print("\n" + "="*50)
    print("CV Mean Performance Report")
    print(f"Mean RMSE across 5 folds: {np.mean([m['rmse'] for m in all_fold_metrics]):.4f}")
    print(f"Mean MSE across 5 folds: {np.mean([m['mse'] for m in all_fold_metrics]):.2f}")
    print("="*50)
    
    # 3. --- Final Evaluation on the Test Set ---
    
    print("\n" + "#"*50)
    print("Final Model Training & Test Set Evaluation")
    print("#"*50)
    
    final_model = BiLSTMRegressor().to(DEVICE)
    final_optimizer = AdamW(final_model.parameters(), lr=2e-5)
    
    # Create DataLoader for the entire training pool
    train_loader = get_data_loaders(full_dataset, train_dev_indices, BATCH_SIZE, shuffle=True)
    test_loader = get_data_loaders(full_dataset, test_indices, BATCH_SIZE, shuffle=False)
    
    # Train the final model on the entire training pool (train_val_indices)
    for epoch in range(num_epochs): # Use same number of epochs or tune it
        final_model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            embeddings = batch['embeddings'].to(DEVICE)
            normalized_targets = batch['targets'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            final_optimizer.zero_grad()
            normalized_predictions = final_model(embeddings)
            flat_predictions = normalized_predictions.flatten()[mask.flatten()]
            flat_targets = normalized_targets.flatten()[mask.flatten()]
            loss = criterion(flat_predictions, flat_targets)
            loss.backward()
            final_optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Final - Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}") 
        
    # Evaluate on the FINAL, completely unseen test set
    final_test_metrics = evaluate_model(final_model, test_loader, criterion, y_name=y_name, fold_num="FINAL TEST")
    
    print("\n" + "*"*50)
    print("Final Performance (Unseen Data)")
    print(f"RMSE: {final_test_metrics['rmse']:.4f}")
    print(f"MSE: {final_test_metrics['mse']:.2f}")
    print("*"*50)
    model_path = Path(MODEL_PATH.format(y_name))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_model.state_dict(), model_path)
    print("Model saved successfully")
    return all_fold_metrics, final_test_metrics

def make_dataset(df: pl.DataFrame, y_name='valence'):
    df = df.drop(['timestamp', 'collection_phase', 'is_words'])
    dataset = CustomDataset(df, y_name=y_name)
    return dataset

def main():
    args = parse_args()
    df = prepare_df(DATASET_PATH)
    if args.mode in ['both', 'valence']:
        dataset = make_dataset(df, y_name='valence')
        group_cross_validate(dataset, y_name='valence', num_epochs=args.epochs)
    if args.mode in ['both', 'arousal']:
        dataset = make_dataset(df, y_name='arousal')
        group_cross_validate(dataset, y_name='arousal', num_epochs=args.epochs)

if __name__ == '__main__':
    main()
