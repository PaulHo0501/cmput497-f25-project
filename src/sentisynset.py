import argparse
import os
import xml.etree.ElementTree as ET

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    nltk.download("averaged_perceptron_tagger_eng")

MANUAL_VA_MAP = {
    'joy':          {'v': 0.90,  'a': 0.70},
    'sadness':      {'v': -0.80, 'a': -0.60},
    'anger':        {'v': -0.70, 'a': 0.90},
    'fear':         {'v': -0.80, 'a': 0.85},
    'trust':        {'v': 0.60,  'a': 0.30},
    'disgust':      {'v': -0.70, 'a': 0.40},
    'surprise':     {'v': 0.10,  'a': 0.90},
    'anticipation': {'v': 0.30,  'a': 0.60},
    'positive':     {'v': 0.80,  'a': 0.00}, 
    'negative':     {'v': -0.80, 'a': 0.00}
}

class SentiSynsetPipeline:
    def __init__(self, xml_path, model_name='all-MiniLM-L6-v2'):
        self.synset_db = self._load_sentisynset(xml_path)
        print(f"Loaded {len(self.synset_db)} labeled synsets from SentiSynset.")
        
        print(f"Loading Neural WSD Model ({model_name})...")
        self.wsd_model = SentenceTransformer(model_name)
        self.lemmatizer = WordNetLemmatizer()

    def _load_sentisynset(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        db = {}
        
        for synset in root.findall('.//Synset'):
            s_id = synset.get('id') 
            label = synset.get('emotion')
            if not label:
                label = synset.get('polarity')
                
            if s_id and label:
                db[s_id] = label.lower()
        return db

    def _get_nltk_id(self, synset):
        offset = synset.offset()
        pos = synset.pos()
        return f"wn:{offset:08d}{pos}"

    def _get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return None

    def neural_disambiguate(self, text, word, pos_tag):
        wn_pos = self._get_wordnet_pos(pos_tag)
        if not wn_pos:
            return None 

        candidates = wn.synsets(word, pos=wn_pos)
        
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0] 

        glosses = [f"{word}: {s.definition()}" for s in candidates]
        
        embeddings = self.wsd_model.encode([text] + glosses, convert_to_tensor=True, show_progress_bar=False)
        
        context_emb = embeddings[0]
        gloss_embs = embeddings[1:]
        
        scores = util.cos_sim(context_emb, gloss_embs)[0]
        best_idx = scores.argmax().item()
        
        return candidates[best_idx]

    def predict_text(self, text):
        if not isinstance(text, str):
            return 0.0, 0.0, []

        tokens = nltk.word_tokenize(text)
        tagged_tokens = nltk.pos_tag(tokens)
        
        valence_scores = []
        arousal_scores = []
        debug_trace = []

        for word, tag in tagged_tokens:
            try:
                if len(word) <= 2: continue 
                disambiguated_synset = self.neural_disambiguate(text, word, tag)
            except Exception:
                continue
            
            if disambiguated_synset:
                ss_id = self._get_nltk_id(disambiguated_synset)
                category = self.synset_db.get(ss_id)
                
                if category and category in MANUAL_VA_MAP:
                    coords = MANUAL_VA_MAP[category]
                    valence_scores.append(coords['v'])
                    arousal_scores.append(coords['a'])
                    debug_trace.append(f"{word}({category})")

        if not valence_scores:
            return 0.0, 0.0, debug_trace 

        final_v = np.mean(valence_scores)
        final_a = np.mean(arousal_scores)
        
        return final_v, final_a, debug_trace


def scale_scores(raw_v, raw_a):
    scaled_v = raw_v * 2.0
    scaled_a = raw_a + 1.0
    return scaled_v, scaled_a

def process_subtask1(df, pipeline, output_file):
    pred_valences = []
    pred_arousals = []
    traces = []

    for text in tqdm(df['text'], desc="Predicting"):
        raw_v, raw_a, trace = pipeline.predict_text(text)
        scaled_v, scaled_a = scale_scores(raw_v, raw_a)
        
        pred_valences.append(scaled_v)
        pred_arousals.append(scaled_a)
        traces.append(",".join(trace) if trace else "")

    df['pred_valence'] = pred_valences
    df['pred_arousal'] = pred_arousals
    df['senti_concepts'] = traces

    if 'valence' in df.columns:
        valid_v = df.dropna(subset=['valence', 'pred_valence'])
        mse_v = np.mean((valid_v['valence'] - valid_v['pred_valence']) ** 2)
        print(f"Valence RMSE: {np.sqrt(mse_v):.4f} (MSE: {mse_v:.4f})")

    if 'arousal' in df.columns:
        valid_a = df.dropna(subset=['arousal', 'pred_arousal'])
        mse_a = np.mean((valid_a['arousal'] - valid_a['pred_arousal']) ** 2)
        print(f"Arousal RMSE: {np.sqrt(mse_a):.4f} (MSE: {mse_a:.4f})")

    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

def process_subtask2a(df, pipeline, output_file):
    
    tqdm.pandas(desc="Predicting Raw Scores")
    
    def predict_row(text):
        raw_v, raw_a, _ = pipeline.predict_text(text)
        return scale_scores(raw_v, raw_a)

    predictions = df['text'].progress_apply(predict_row)
    df['pred_val_t'], df['pred_ar_t'] = zip(*predictions)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['user_id', 'timestamp'])
    
    df['pred_val_next'] = df.groupby('user_id')['pred_val_t'].shift(-1)
    df['pred_ar_next'] = df.groupby('user_id')['pred_ar_t'].shift(-1)
    
    df['pred_state_change_valence'] = df['pred_val_next'] - df['pred_val_t']
    df['pred_state_change_arousal'] = df['pred_ar_next'] - df['pred_ar_t']
    
    
    if 'state_change_valence' in df.columns:
        valid = df.dropna(subset=['state_change_valence', 'pred_state_change_valence'])
        mse = np.mean((valid['state_change_valence'] - valid['pred_state_change_valence'])**2)
        print(f"Valence Change RMSE: {np.sqrt(mse):.4f}")
        
    if 'state_change_arousal' in df.columns:
        valid = df.dropna(subset=['state_change_arousal', 'pred_state_change_arousal'])
        mse = np.mean((valid['state_change_arousal'] - valid['pred_state_change_arousal'])**2)
        print(f"Arousal Change RMSE: {np.sqrt(mse):.4f}")

    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

def process_subtask2b(df, pipeline, output_file):
    tqdm.pandas(desc="Predicting Raw Scores")
    def predict_row(text):
        raw_v, raw_a, _ = pipeline.predict_text(text)
        return scale_scores(raw_v, raw_a)

    predictions = df['text'].progress_apply(predict_row)
    df['pred_val_t'], df['pred_ar_t'] = zip(*predictions)
    
    if 'group' not in df.columns and 'timestamp' in df.columns:
        print("Generating 'group' column based on timestamp rank...")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['user_id', 'timestamp'])
        
        df['rank'] = df.groupby('user_id')['timestamp'].rank(method='first')
        df['count'] = df.groupby('user_id')['timestamp'].transform('count')
        
        df['group'] = np.where(df['rank'] <= (df['count'] / 2), 1, 2)
    
    agg = df.groupby(['user_id', 'group'])[['pred_val_t', 'pred_ar_t']].mean().reset_index()
    
    pivot = agg.pivot(index='user_id', columns='group', values=['pred_val_t', 'pred_ar_t'])
    
    pivot.columns = [f'{col[0]}_{col[1]}' for col in pivot.columns]
    pivot = pivot.reset_index()
    
    pivot['pred_disp_change_valence'] = pivot['pred_val_t_2'] - pivot['pred_val_t_1']
    pivot['pred_disp_change_arousal'] = pivot['pred_ar_t_2'] - pivot['pred_ar_t_1']
    
    if 'disposition_change_valence' in df.columns:
        truth = df.groupby('user_id')[['disposition_change_valence', 'disposition_change_arousal']].first().reset_index()
        final = pd.merge(pivot, truth, on='user_id')
        
        
        valid_v = final.dropna(subset=['disposition_change_valence', 'pred_disp_change_valence'])
        if len(valid_v) > 0:
            mse_v = np.mean((valid_v['disposition_change_valence'] - valid_v['pred_disp_change_valence'])**2)
            print(f"Disposition Valence RMSE: {np.sqrt(mse_v):.4f}")
            
        valid_a = final.dropna(subset=['disposition_change_arousal', 'pred_disp_change_arousal'])
        if len(valid_a) > 0:
            mse_a = np.mean((valid_a['disposition_change_arousal'] - valid_a['pred_disp_change_arousal'])**2)
            print(f"Disposition Arousal RMSE: {np.sqrt(mse_a):.4f}")
            
        final.to_csv(output_file, index=False)
    else:
        pivot.to_csv(output_file, index=False)
        
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentiSynset Pipeline for SemEval Subtasks")
    parser.add_argument("--task", type=str, required=True, choices=['1', '2a', '2b'], 
                        help="Task to run: '1' (Text Classification), '2a' (Forecasting), '2b' (Disposition)")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--xml", type=str, default="data/sentisynset_lexicon.xml", help="Path to SentiSynset XML")
    parser.add_argument("--output", type=str, default="output/sentisynset_preds.csv", help="Path to save output")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    try:
        pipeline = SentiSynsetPipeline(args.xml, model_name=args.model)
        df = pd.read_csv(args.input)
        if df.empty:
            print("CSV is empty.")
            exit()
        if args.task == '1':
            process_subtask1(df, pipeline, args.output)
        elif args.task == '2a':
            process_subtask2a(df, pipeline, args.output)
        elif args.task == '2b':
            process_subtask2b(df, pipeline, args.output)

    except FileNotFoundError as e:
        print(f"Error: {e}")
