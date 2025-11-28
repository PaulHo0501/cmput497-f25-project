import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# -----------------------------------------------------------------------------
# Feature Engineering: BERT & DistilBERT Embeddings
# -----------------------------------------------------------------------------
def generate_embeddings(text_series, model_name, n_components=4, prefix='emb'):
    """Generates embeddings and reduces dimension via PCA."""
    print(f"Generating embeddings for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    embeddings = []
    # Process in batches (batch_size=32) for speed
    batch_size = 32
    text_list = text_series.astype(str).tolist()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size)):
            batch = text_list[i : i + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors='pt'
            ).to(device)
            output = model(**encoded)
            # Use CLS token (index 0) as sentence embedding
            cls_emb = output.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
            
    full_embeddings = np.vstack(embeddings)
    
    # Reduce dimensionality using PCA to keep TFT efficient
    print(f"Reducing {model_name} dimensions from {full_embeddings.shape[1]} to {n_components}...")
    pca = PCA(n_components=n_components)
    reduced_emb = pca.fit_transform(full_embeddings)
    
    # Create DataFrame with new feature columns
    cols = [f"{prefix}_{i}" for i in range(n_components)]
    return pd.DataFrame(reduced_emb, columns=cols, index=text_series.index)

def main():
    # 1. Load and Preprocess Data
    # -----------------------------------------------------------------------------
    data_path = 'train_subtask1.csv'
    # Assuming CSV structure matches the prompt.
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)

    # TFT requires a strictly increasing integer time index per group (user_id)
    data = data.sort_values(['user_id', 'timestamp'])
    data['time_idx'] = data.groupby('user_id').cumcount()

    # Convert Boolean 'is_words' to integer (0 or 1)
    data['is_words_int'] = data['is_words'].astype(int)

    # Ensure target columns are floats
    data['valence'] = data['valence'].astype(float)
    data['arousal'] = data['arousal'].astype(float)

    # Cast static categorical ID to string
    data['user_id'] = data['user_id'].astype(str)

    # Generate DistilBERT embeddings (lighter/faster)
    distilbert_feats = generate_embeddings(
        data['text'], "distilbert-base-uncased", n_components=4, prefix="distil"
    )
    data = pd.concat([data, distilbert_feats], axis=1)

    # Generate BERT embeddings
    bert_feats = generate_embeddings(
        data['text'], "bert-base-uncased", n_components=4, prefix="bert"
    )
    data = pd.concat([data, bert_feats], axis=1)

    # Collect new feature names for TFT
    embedding_features = list(distilbert_feats.columns) + list(bert_feats.columns)

    # 2. Define Dataset
    # -----------------------------------------------------------------------------
    print("Defining TimeSeriesDataSet...")
    max_prediction_length = 5
    max_encoder_length = 20
    training_cutoff = data["time_idx"].max() - max_prediction_length

    training_dataset = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="valence",
        group_ids=["user_id"],
        min_encoder_length=1,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["user_id"],
        time_varying_known_reals=[
            "time_idx", 
            "collection_phase", 
            "is_words_int",
        ] + embedding_features, # Add the embedding PCA columns here
        time_varying_unknown_reals=[
            "valence",
            "arousal"
        ],
        target_normalizer=GroupNormalizer(
            groups=["user_id"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training_dataset, data, predict=True, stop_randomization=True
    )

    batch_size = 64
    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size * 10, num_workers=0
    )

    # 3. Configure and Initialize Model
    # -----------------------------------------------------------------------------
    print("Configuring Model and Trainer...")
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=32,          # Increased hidden size to accommodate richer features
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=16, # Increased for embedding inputs
        output_size=1,
        loss=RMSE(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # 4. Train
    # -----------------------------------------------------------------------------
    print(f"Starting training for target: {training_dataset.target}")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # 5. Predict
    # -----------------------------------------------------------------------------
    best_model_path = trainer.checkpoint_callback.best_model_path
    print("Training complete. Best model saved at:", best_model_path)
    
    # Optional: Load best model to verify
    # best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

if __name__ == "__main__":
    main()
