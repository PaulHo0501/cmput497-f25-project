import argparse
import warnings
from typing import List, Union

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import RMSE, MultiLoss
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def generate_embeddings(text_series, model_name, n_components=32, prefix='emb'):
    """Generates embeddings and reduces dimension via PCA."""
    print(f"Generating embeddings for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except OSError:
        print(f"Error loading {model_name}. Please ensure internet connection or valid path.")
        return pd.DataFrame()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    embeddings = []
    batch_size = 32
    text_list = text_series.astype(str).tolist()
    
    with torch.no_grad():
        for i in tqdm(range(0, len(text_list), batch_size)):
            batch = text_list[i : i + batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors='pt'
            ).to(device)
            output = model(**encoded)
            cls_emb = output.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
            
    full_embeddings = np.vstack(embeddings)
    
    print(f"Reducing {model_name} dimensions from {full_embeddings.shape[1]} to {n_components}...")
    pca = PCA(n_components=n_components)
    reduced_emb = pca.fit_transform(full_embeddings)
    
    cols = [f"{prefix}_{i}" for i in range(n_components)]
    return pd.DataFrame(reduced_emb, columns=cols, index=text_series.index)

def parse_args():
    parser = argparse.ArgumentParser(description="TFT Training for SemEval Subtask 2")
    parser.add_argument(
        "--target", 
        type=str, 
        default="both", 
        choices=["valence", "arousal", "both"],
        help="Train on 'valence', 'arousal', or 'both' simultaneously."
    )
    parser.add_argument(
        "--approach", 
        type=str, 
        default="indirect", 
        choices=["direct", "indirect"],
        help="direct: train on state_change_X | indirect: train on X, calculate change later"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.target == "both":
        raw_cols_to_use = ["valence", "arousal"]
    else:
        raw_cols_to_use = [args.target]

    training_targets = []
    for col in raw_cols_to_use:
        if args.approach == "direct":
            training_targets.append(f"state_change_{col}")
        else:
            training_targets.append(col)
            
    target_arg = training_targets[0] if len(training_targets) == 1 else training_targets
    
    print(f"=== MODE: {args.target.upper()} TARGETS | {args.approach.upper()} APPROACH ===")
    print(f"Training on: {target_arg}")

    DATA_PATH = 'data/train_subtask2a.csv'
    try:
        data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: {DATA_PATH} not found.")
        return

    cols_to_numeric = ['valence', 'arousal', 'state_change_valence', 'state_change_arousal']
    for col in cols_to_numeric:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    change_cols = [f"state_change_{c}" for c in raw_cols_to_use]
    basic_reqs = ['user_id', 'timestamp'] + raw_cols_to_use
    initial_len = len(data)
    data.dropna(subset=basic_reqs, inplace=True)

    if args.approach == "direct":
        data.dropna(subset=change_cols, inplace=True)
        print(f"Direct Mode")
        
    else:
        for col in change_cols:
            if col in data.columns:
                data[col] = data[col].fillna(0.0)
        print(f"Indirect Mode")

    data = data.sort_values(['user_id', 'timestamp'])
    data['time_idx'] = data.groupby('user_id').cumcount()
    data['is_words_int'] = data['is_words'].astype(int)
    data['user_id'] = data['user_id'].astype(str)

    distilbert_feats = generate_embeddings(
        data['text'], "distilbert-base-uncased", n_components=32, prefix="distil"
    )
    data = pd.concat([data, distilbert_feats], axis=1)

    bert_feats = generate_embeddings(
        data['text'], "bert-base-uncased", n_components=32, prefix="bert"
    )
    data = pd.concat([data, bert_feats], axis=1)

    embedding_features = list(distilbert_feats.columns) + list(bert_feats.columns)

    max_prediction_length = 1
    max_encoder_length = 20
    training_cutoff = data["time_idx"].max() - max_prediction_length

    if isinstance(target_arg, list):
        target_normalizer = MultiNormalizer(
            [GroupNormalizer(groups=["user_id"], transformation=None) for _ in target_arg]
        )
    else:
        target_normalizer = GroupNormalizer(groups=["user_id"], transformation=None)

    training_dataset = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=target_arg, 
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
        ] + embedding_features,
        
        time_varying_unknown_reals=list(set(raw_cols_to_use + change_cols)),
        
        target_normalizer=target_normalizer,
        
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
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

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    if isinstance(target_arg, list):
        model_loss = MultiLoss([RMSE() for _ in target_arg])
        output_size = [1 for _ in target_arg] 
    else:
        model_loss = RMSE()
        output_size = 1

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=output_size,  
        loss=model_loss, 
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    print(f"Starting training...")
    trainer.fit(
        model=tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"\nTraining complete. Loading best model: {best_model_path}")
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path, weights_only=False)
    
    print("Evaluating...")
    
    out = best_tft.predict(val_dataloader, return_x=True, mode="prediction")
    predictions = out[0]
    x = out[1] 
    if isinstance(predictions, list):
        predictions = torch.stack(predictions, dim=-1)
    eval_groups = []
    all_actuals_batches = [y[0] for _, y in iter(val_dataloader)]
    if isinstance(target_arg, list):
        if predictions.ndim == 3: 
            predictions = predictions.squeeze(1)
        n_targets = len(target_arg)
        actuals_per_target = []
        for t_i in range(n_targets):
            target_t_actuals = torch.cat([batch[t_i] for batch in all_actuals_batches]).cpu().numpy().flatten()
            actuals_per_target.append(target_t_actuals)
            target_t_preds = predictions[:, t_i].cpu().numpy().flatten()
            eval_groups.append({
                "preds": target_t_preds,
                "actuals": target_t_actuals,
                "target_name": training_targets[t_i],
                "raw_name": raw_cols_to_use[t_i],
                "target_idx": t_i
            })
            
    else:
        if predictions.ndim == 2:
             predictions = predictions.squeeze(1)
        preds_np = predictions.cpu().numpy().flatten()
        actuals_np = torch.cat(all_actuals_batches).cpu().numpy().flatten()
        eval_groups.append({
            "preds": preds_np,
            "actuals": actuals_np,
            "target_name": target_arg,
            "raw_name": raw_cols_to_use[0],
            "target_idx": 0
        })
    
    encoder_lengths = x['encoder_lengths']
    
    for group in eval_groups:
        t_name = group['target_name']
        r_name = group['raw_name']
        preds = group['preds']
        actuals = group['actuals']
        t_idx = group['target_idx']
        
        print(f"\n--- EVALUATION: {t_name.upper()} ---")

        if args.approach == "direct":
            mse = np.mean((preds - actuals) ** 2)
            rmse = np.sqrt(mse)
            print(f"Method: Direct Prediction")
            print(f"Validation RMSE: {rmse:.4f}")
            
        else:
            if isinstance(x['encoder_target'], list):
                enc_target_tensor = x['encoder_target'][t_idx]
            else:
                enc_target_tensor = x['encoder_target']
            
            val_t = []
            for i in range(len(enc_target_tensor)):
                length = encoder_lengths[i]
                val_t.append(enc_target_tensor[i, length - 1].item())
            val_t = np.array(val_t)
            
            pred_change = preds - val_t
            actual_change = actuals - val_t
            
            mse = np.mean((pred_change - actual_change) ** 2)
            rmse = np.sqrt(mse)
            
            print(f"Method: Indirect (Pred[{r_name}] - Hist[{r_name}])")
            print(f"Validation RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()
