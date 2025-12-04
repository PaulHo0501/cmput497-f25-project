"""
Method 5: CNN (Convolutional Neural Network) for English V/A Prediction
Fast and effective for text classification/regression
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm


class TextCNN_VA_Model(nn.Module):
    """CNN model for V/A prediction."""
    
    def __init__(self, vocab_size, embedding_dim=300, num_filters=100,
                 filter_sizes=[3, 4, 5], dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Multiple convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=fs
            ) for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        total_filters = num_filters * len(filter_sizes)
        
        self.valence_head = nn.Sequential(
            nn.Linear(total_filters, total_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_filters // 2, 1)
        )
        
        self.arousal_head = nn.Sequential(
            nn.Linear(total_filters, total_filters // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_filters // 2, 1)
        )
    
    def forward(self, input_ids):
        # Embedding: (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(input_ids)
        
        # Transpose for Conv1d: (batch, embed_dim, seq_len)
        embedded = embedded.permute(0, 2, 1)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))  # (batch, num_filters, seq_len - kernel_size + 1)
            pooled = torch.max_pool1d(conv_out, conv_out.size(2))  # (batch, num_filters, 1)
            pooled = pooled.squeeze(2)  # (batch, num_filters)
            conv_outputs.append(pooled)
        
        # Concatenate all filter outputs
        combined = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * len(filter_sizes))
        combined = self.dropout(combined)
        
        # Predictions
        valence = self.valence_head(combined).squeeze()
        arousal = self.arousal_head(combined).squeeze()
        
        return valence, arousal


class CNNVAPredictor:
    """CNN-based predictor for V/A."""
    
    def __init__(self, vocab_size=10000, embedding_dim=300, num_filters=100,
                 filter_sizes=[3, 4, 5], device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.vocab = None
        self.model = TextCNN_VA_Model(
            vocab_size, embedding_dim, num_filters, filter_sizes
        )
        self.model.to(self.device)
        
        print(f"CNN Model: {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def build_vocabulary(self, texts, vocab_size=10000):
        """Build vocabulary from texts."""
        from collections import Counter
        
        print("Building vocabulary...")
        word_counts = Counter()
        
        for text in tqdm(texts, desc="Counting words"):
            words = str(text).lower().split()
            word_counts.update(words)
        
        most_common = word_counts.most_common(vocab_size - 1)
        self.vocab = {word: idx + 1 for idx, (word, _) in enumerate(most_common)}
        
        print(f"Vocabulary size: {len(self.vocab)}")
        return self.vocab
    
    def train(self, train_df, val_df, epochs=15, batch_size=32, lr=0.001,
              max_length=150):
        """Train CNN model."""
        from torch.utils.data import Dataset, DataLoader
        
        # Build vocabulary
        if self.vocab is None:
            self.build_vocabulary(train_df['text'].tolist())
        
        # Dataset class
        class SimpleDataset(Dataset):
            def __init__(self, texts, valences, arousals, vocab, max_len):
                self.texts = texts
                self.valences = valences
                self.arousals = arousals
                self.vocab = vocab
                self.max_len = max_len
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                words = str(self.texts[idx]).lower().split()[:self.max_len]
                indices = [self.vocab.get(w, 0) for w in words]
                
                # Pad
                if len(indices) < self.max_len:
                    indices += [0] * (self.max_len - len(indices))
                
                return {
                    'input_ids': torch.tensor(indices, dtype=torch.long),
                    'valence': torch.tensor(self.valences[idx], dtype=torch.float),
                    'arousal': torch.tensor(self.arousals[idx], dtype=torch.float)
                }
        
        train_dataset = SimpleDataset(
            train_df['text'].tolist(),
            train_df['valence'].tolist(),
            train_df['arousal'].tolist(),
            self.vocab,
            max_length
        )
        
        val_dataset = SimpleDataset(
            val_df['text'].tolist(),
            val_df['valence'].tolist(),
            val_df['arousal'].tolist(),
            self.vocab,
            max_length
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        print(f"\nTraining CNN for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                valence_true = batch['valence'].to(self.device)
                arousal_true = batch['arousal'].to(self.device)
                
                optimizer.zero_grad()
                
                valence_pred, arousal_pred = self.model(input_ids)
                
                loss_v = criterion(valence_pred, valence_true)
                loss_a = criterion(arousal_pred, arousal_true)
                loss = loss_v + loss_a
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    valence_true = batch['valence'].to(self.device)
                    arousal_true = batch['arousal'].to(self.device)
                    
                    valence_pred, arousal_pred = self.model(input_ids)
                    
                    loss_v = criterion(valence_pred, valence_true)
                    loss_a = criterion(arousal_pred, arousal_true)
                    loss = loss_v + loss_a
                    
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    'model_state': self.model.state_dict(),
                    'vocab': self.vocab
                }, 'models/cnn_va_model.pt')
                print(f"  ✓ Saved (val loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    
    def predict(self, texts, batch_size=32, max_length=150):
        """Predict V/A."""
        if self.vocab is None:
            raise ValueError("Model not trained or loaded")
        
        # Simple tokenization
        encoded = []
        for text in texts:
            words = str(text).lower().split()[:max_length]
            indices = [self.vocab.get(w, 0) for w in words]
            if len(indices) < max_length:
                indices += [0] * (max_length - len(indices))
            encoded.append(indices)
        
        encoded = torch.tensor(encoded, dtype=torch.long)
        
        self.model.eval()
        all_valences = []
        all_arousals = []
        
        with torch.no_grad():
            for i in range(0, len(encoded), batch_size):
                batch = encoded[i:i+batch_size].to(self.device)
                valence, arousal = self.model(batch)
                
                all_valences.extend(valence.cpu().numpy())
                all_arousals.extend(arousal.cpu().numpy())
        
        return all_valences, all_arousals
    
    def load_model(self, path='models/cnn_va_model.pt'):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.vocab = checkpoint['vocab']
        print(f"✓ Loaded CNN model from {path}")


def main():
    parser = argparse.ArgumentParser(description='CNN V/A prediction')
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-filters', type=int, default=100)
    
    args = parser.parse_args()
    
    # Load and split data
    df = pd.read_csv(args.data)
    df = df.dropna(subset=['text', 'valence', 'arousal'])
    
    users = df['user_id'].unique()
    train_users, test_users = train_test_split(users, test_size=0.2, random_state=42)
    train_users, val_users = train_test_split(train_users, test_size=0.1, random_state=42)
    
    train_df = df[df['user_id'].isin(train_users)]
    val_df = df[df['user_id'].isin(val_users)]
    test_df = df[df['user_id'].isin(test_users)]
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Train
    predictor = CNNVAPredictor(num_filters=args.num_filters)
    predictor.train(train_df, val_df, epochs=args.epochs, batch_size=args.batch_size)
    
    # Test
    valences, arousals = predictor.predict(test_df['text'].tolist())
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats import pearsonr
    
    val_mse = mean_squared_error(test_df['valence'], valences)
    ar_mse = mean_squared_error(test_df['arousal'], arousals)
    val_r, _ = pearsonr(test_df['valence'], valences)
    ar_r, _ = pearsonr(test_df['arousal'], arousals)
    
    print(f"\nTest Results:")
    print(f"  Valence - MSE: {val_mse:.4f}, Pearson: {val_r:.3f}")
    print(f"  Arousal - MSE: {ar_mse:.4f}, Pearson: {ar_r:.3f}")


if __name__ == '__main__':
    main()


"""
USAGE:

python 5_cnn_model.py --data semeval_data.csv --epochs 15 --batch-size 32
"""
