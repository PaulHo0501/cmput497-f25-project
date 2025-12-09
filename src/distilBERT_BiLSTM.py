import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, ShuffleSplit
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import DistilBertModel, DistilBertTokenizer

MODEL_NAME = "distilbert-base-uncased"
MAX_SEQUENCE_LENGTH = 128
MAX_TOKEN_LENGTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_SIZE = 0.2
RANDOM_STATE = 51
DISTILBERT_HIDDEN_DIM = 768
BATCH_SIZE = 8

PATH_TRAIN_1 = 'data/train_subtask1.csv'
PATH_TRAIN_2A = 'data/train_subtask2a.csv'
PATH_TRAIN_2B = 'data/train_subtask2b.csv'


class DatasetSubtask1(Dataset):
    def __init__(self, df: pl.DataFrame, y_name: str):
        # Sort by user and time
        df = df.sort(["user_id", "timestamp"])
        
        target_col = y_name
        
        if target_col not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(target_col))

        self.processed_df = df.group_by('user_id').agg(
            pl.col('text').implode().alias('texts'),
            pl.col(target_col).implode().alias('targets')
        )
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        self.embedder = DistilBertModel.from_pretrained(MODEL_NAME, device_map=DEVICE)
        self.embedder.eval()

    def __len__(self):
        return self.processed_df.height

    def __getitem__(self, index):
        row = self.processed_df.row(index, named=True)
        
        text_embeddings = self.encode_texts(row['texts'])
        current_seq_len = text_embeddings.size(0)

        raw_targets = row['targets'][:MAX_SEQUENCE_LENGTH]
        
        clean_targets = [t if t is not None else 0.0 for t in raw_targets]
        target_tensor = torch.tensor(clean_targets, dtype=torch.float32)

        pad_len = MAX_SEQUENCE_LENGTH - current_seq_len
        
        padding_emb = torch.zeros(pad_len, text_embeddings.size(1), dtype=torch.float32)
        padded_embeddings = torch.cat([text_embeddings, padding_emb], dim=0)
        
        target_padding = torch.zeros(pad_len, dtype=torch.float32)
        padded_targets = torch.cat([target_tensor, target_padding], dim=0)
        
        mask = torch.ones(current_seq_len, dtype=torch.bool)
        mask_padding = torch.zeros(pad_len, dtype=torch.bool)
        full_mask = torch.cat([mask, mask_padding])

        return {
            "embeddings": padded_embeddings,
            "targets": padded_targets,
            "mask": full_mask,
            "user_id": row['user_id']
        }

    @torch.no_grad()
    def encode_texts(self, texts):
        texts = texts[:MAX_SEQUENCE_LENGTH]
        embeddings_list = []
        for i in range(0, len(texts), 32):
            batch_texts = texts[i:i+32]
            encoded = self.tokenizer(batch_texts, padding='max_length', truncation=True, 
                                     max_length=MAX_TOKEN_LENGTH, return_tensors='pt')
            input_ids = encoded['input_ids'].to(DEVICE)
            mask = encoded['attention_mask'].to(DEVICE)
            output = self.embedder(input_ids=input_ids, attention_mask=mask)
            embeddings_list.append(output.last_hidden_state[:, 0, :].cpu())
        
        if len(embeddings_list) > 0:
            return torch.cat(embeddings_list, dim=0)
        else:
            return torch.empty(0, 768)


class DatasetSubtask2A(Dataset):
    def __init__(self, df: pl.DataFrame, y_name: str):
        df = df.sort(["user_id", "timestamp"])
        target_col = f"state_change_{y_name}"
        if target_col not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(target_col))

        self.processed_df = df.group_by('user_id').agg(
            pl.col('text').implode().alias('texts'),
            pl.col(target_col).implode().alias('targets')
        )
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        self.embedder = DistilBertModel.from_pretrained(MODEL_NAME, device_map=DEVICE)
        self.embedder.eval()

    def __len__(self):
        return self.processed_df.height

    def __getitem__(self, index):
        row = self.processed_df.row(index, named=True)
        text_embeddings = self.encode_texts(row['texts'])
        current_seq_len = text_embeddings.size(0)

        raw_targets = row['targets'][:MAX_SEQUENCE_LENGTH] 
        clean_targets = [t if t is not None else 0.0 for t in raw_targets]
        target_tensor = torch.tensor(clean_targets, dtype=torch.float32)

        pad_len = MAX_SEQUENCE_LENGTH - current_seq_len
        padding_emb = torch.zeros(pad_len, text_embeddings.size(1), dtype=torch.float32)
        padded_embeddings = torch.cat([text_embeddings, padding_emb], dim=0)
        
        target_padding = torch.zeros(pad_len, dtype=torch.float32)
        padded_targets = torch.cat([target_tensor, target_padding], dim=0)
        
        mask = torch.ones(current_seq_len, dtype=torch.bool)
        if current_seq_len > 0:
            mask[-1] = False 
        
        mask_padding = torch.zeros(pad_len, dtype=torch.bool)
        full_mask = torch.cat([mask, mask_padding])

        return {
            "embeddings": padded_embeddings,
            "targets": padded_targets,
            "mask": full_mask,
            "user_id": row['user_id']
        }

    @torch.no_grad()
    def encode_texts(self, texts):
        texts = texts[:MAX_SEQUENCE_LENGTH]
        embeddings_list = []
        for i in range(0, len(texts), 32):
            batch_texts = texts[i:i+32]
            encoded = self.tokenizer(batch_texts, padding='max_length', truncation=True, 
                                     max_length=MAX_TOKEN_LENGTH, return_tensors='pt')
            input_ids = encoded['input_ids'].to(DEVICE)
            mask = encoded['attention_mask'].to(DEVICE)
            output = self.embedder(input_ids=input_ids, attention_mask=mask)
            embeddings_list.append(output.last_hidden_state[:, 0, :].cpu())
        if len(embeddings_list) > 0: return torch.cat(embeddings_list, dim=0)
        else: return torch.empty(0, 768)


class DatasetSubtask2B(Dataset):
    def __init__(self, df: pl.DataFrame, y_name: str):
        df = df.sort(["user_id", "timestamp"])
        target_col = f"disposition_change_{y_name}"
        if target_col not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(target_col))

        df = df.filter(pl.col("group") == 2)
        self.processed_df = df.group_by('user_id').agg(
            pl.col('text').implode().alias('texts'),
            pl.col(target_col).first().alias('target') 
        )
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        self.embedder = DistilBertModel.from_pretrained(MODEL_NAME, device_map=DEVICE)
        self.embedder.eval()

    def __len__(self):
        return self.processed_df.height

    def __getitem__(self, index):
        row = self.processed_df.row(index, named=True)
        text_embeddings = self.encode_texts(row['texts'])
        target = torch.tensor(row['target'], dtype=torch.float32)
        
        pad_len = MAX_SEQUENCE_LENGTH - text_embeddings.size(0)
        padding_tensor = torch.zeros(pad_len, text_embeddings.size(1), dtype=torch.float32)
        padded_embeddings = torch.cat([text_embeddings, padding_tensor], dim=0)
        
        return {
            "embeddings": padded_embeddings,
            "target": target, 
            "actual_len": text_embeddings.size(0) 
        }

    @torch.no_grad()
    def encode_texts(self, texts):
        texts = texts[:MAX_SEQUENCE_LENGTH]
        embeddings_list = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            encoded = self.tokenizer(batch, padding='max_length', truncation=True, max_length=MAX_TOKEN_LENGTH, return_tensors='pt')
            embeddings_list.append(self.embedder(encoded['input_ids'].to(DEVICE), encoded['attention_mask'].to(DEVICE)).last_hidden_state[:, 0, :].cpu())
        if len(embeddings_list) > 0: return torch.cat(embeddings_list, dim=0)
        else: return torch.empty(0, 768)


class ModelSubtask1(nn.Module):
    def __init__(self, lstm_hidden_dim=256, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=DISTILBERT_HIDDEN_DIM,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            bidirectional=True, 
            batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.regressor = nn.Linear(lstm_hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return self.regressor(out).squeeze(-1) 


class ModelSubtask2A(nn.Module):
    def __init__(self, lstm_hidden_dim=256, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=DISTILBERT_HIDDEN_DIM,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        self.regressor = nn.Linear(lstm_hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.regressor(out).squeeze(-1)

class ModelSubtask2B(nn.Module):
    def __init__(self, lstm_hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=DISTILBERT_HIDDEN_DIM,
            hidden_size=lstm_hidden_dim,
            bidirectional=True, 
            batch_first=True
        )
        self.dropout = nn.Dropout(0.3)
        self.regressor = nn.Linear(lstm_hidden_dim * 2, 1)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        final_h = torch.cat((hn[-2], hn[-1]), dim=1) 
        x = self.dropout(final_h)
        return self.regressor(x).squeeze(-1)


def train_epoch(model, loader, optimizer, criterion, mode='1'):
    model.train()
    total_loss = 0
    for batch in loader:
        emb = batch['embeddings'].to(DEVICE)
        optimizer.zero_grad()
        
        preds = model(emb)
        
        if mode in ['1', '2a']:
            targets = batch['targets'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            preds = preds.flatten()[mask.flatten()]
            targets = targets.flatten()[mask.flatten()]
        else: 
            targets = batch['target'].to(DEVICE)
            
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, criterion, mode='1'):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            emb = batch['embeddings'].to(DEVICE)
            preds = model(emb)
            
            if mode in ['1', '2a']:
                targets = batch['targets'].to(DEVICE)
                mask = batch['mask'].to(DEVICE)
                flat_p = preds.flatten()[mask.flatten()]
                flat_t = targets.flatten()[mask.flatten()]
                all_preds.extend(flat_p.cpu().numpy())
                all_targets.extend(flat_t.cpu().numpy())
                loss = criterion(flat_p, flat_t)
            else:
                targets = batch['target'].to(DEVICE)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                loss = criterion(preds, targets)
                
            total_loss += loss.item()

    if len(all_targets) > 0:
        mse = np.mean((np.array(all_targets) - np.array(all_preds))**2)
    else:
        mse = 0.0
    return mse, total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['1', '2a', '2b'])
    parser.add_argument('--target', type=str, default='valence', choices=['valence', 'arousal'])
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()

    print(f"Running Task {args.task} for target {args.target}...")

    if args.task == '1':
        df = pl.read_csv(PATH_TRAIN_1)
        if "timestamp" in df.columns: df = df.with_columns(pl.col("timestamp").str.to_datetime())
        dataset = DatasetSubtask1(df, args.target)
        model = ModelSubtask1().to(DEVICE)

    elif args.task == '2a':
        df = pl.read_csv(PATH_TRAIN_2A)
        if "timestamp" in df.columns: df = df.with_columns(pl.col("timestamp").str.to_datetime())
        dataset = DatasetSubtask2A(df, args.target)
        model = ModelSubtask2A().to(DEVICE)
    
    else: 
        df = pl.read_csv(PATH_TRAIN_2B)
        if "timestamp" in df.columns: df = df.with_columns(pl.col("timestamp").str.to_datetime())
        dataset = DatasetSubtask2B(df, args.target)
        model = ModelSubtask2B().to(DEVICE)

    indices = np.arange(len(dataset))
    train_idx, val_idx = next(ShuffleSplit(test_size=0.2, random_state=RANDOM_STATE).split(indices))
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=1e-4) 
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, mode=args.task)
        val_mse, val_loss = eval_epoch(model, val_loader, criterion, mode=args.task)
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val MSE {val_mse:.4f} | Val RMSE {np.sqrt(val_mse):.4f}")

    save_path = f"models/model_task{args.task}_{args.target}.pt"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()
