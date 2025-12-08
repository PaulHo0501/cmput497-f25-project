import argparse
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
import polars as pl
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = 'cuda'
MODEL_NAME = 'microsoft/Phi-4-mini-instruct'
NOW = datetime.now()
OUTPUTS_PATH = Path('outputs/')
LOGS_PATH = Path("logs/")
PATTERN = r"\*\*Result:\*\*\s*(.+)"

SUBTASK2A_PROMPT_TEMPLATE = '''You are predicting emotional scores for a person's next text based on their previous text.

Here is the person's previous text with its emotional scores:

Text (Valence: {valence_1}, Arousal: {arousal_1}):
{text_1}

Score Scales:
{valence_scale}
{arousal_scale}

Now, based on the emotional pattern, predict the emotional scores for the following text:

{text_2}

Predict:
1. Valence score (range: -2 to +2)
2. Arousal score (range: 0 to +2)

Your reply format:
**Reasoning:** <Explain the emotional pattern and your prediction>

**Result:** <two scores in comma-separated format>

Format: valence_score, arousal_score

IMPORTANT: Output exactly 2 numbers representing the predicted scores (range: -2 to +2 for Valence score and 0 to 2 for Arousal score).
'''

SUBTASK2A_RUBRICS = {
    "valence_scale": "Valence Scale: -2 (Very Negative), -1 (Negative), 0 (Neutral), +1 (Positive), +2 (Very Positive)",
    "arousal_scale": "Arousal Scale: 0 (Neutral), +1 (Excited), +2 (Very Excited)"
}

DATA_PATH_SUBTASK2A = './data/train_subtask2a.csv'

def read_data_subtask2a():
    return pd.read_csv(DATA_PATH_SUBTASK2A)

def prepare_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto",
        torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

def evaluate_subtask2a(args):
    print("Evaluate Subtask 2a - Next Text Score Prediction")
    df = read_data_subtask2a()
    OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    
    df = df.sort_values(['user_id', 'text_id']).reset_index(drop=True)
    
    model, tokenizer = prepare_model()
    pattern = re.compile(PATTERN)
    all_predictions = []
    all_changes = [] 
    
    print(f"Total rows in dataset: {len(df)}")
    
    with open(LOGS_PATH/"subtask2a_avg.txt", 'w', encoding='utf-8') as log_file:
        for idx in tqdm(range(len(df)), desc="Processing rows"):
            row = df.iloc[idx]
            
            text_2 = row['text']
            user_id = row['user_id']
            text_id_2 = row['text_id']
            
            if idx == 0:
                prediction_text = "0.0, 0.0"
                change_text = "0.0, 0.0"
                all_predictions.append(prediction_text)
                all_changes.append(change_text)
                
                log_file.write(f"{'='*50}\n")
                log_file.write(f"Row {idx}: First entry for user {user_id}, using 0.0, 0.0\n")
                log_file.write(f"{'='*50}\n\n")
                
                continue
            
            prev_row = df.iloc[idx - 1]
            
            if prev_row['user_id'] != user_id:
                prediction_text = "0.0, 0.0"
                change_text = "0.0, 0.0"
                all_predictions.append(prediction_text)
                all_changes.append(change_text)
                
                log_file.write(f"{'='*50}\n")
                log_file.write(f"Row {idx}: First entry for user {user_id}, using 0.0, 0.0\n")
                log_file.write(f"{'='*50}\n\n")
                
                if args.debug:
                    print(f"Debug Mode: New user, using default 0.0, 0.0")
                    break
                continue
            
            text_1 = prev_row['text']
            valence_1 = prev_row['valence']
            arousal_1 = prev_row['arousal']
            text_id_1 = prev_row['text_id']
            
            prompt = SUBTASK2A_PROMPT_TEMPLATE.format(
                text_1=text_1,
                text_2=text_2,
                valence_1=valence_1,
                arousal_1=arousal_1,
                **SUBTASK2A_RUBRICS
            )
            
            messages = [{"role": "user", "content": prompt}]
            
            chat_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            model_inputs = tokenizer([chat_text], return_tensors="pt").to(DEVICE)
            attention_mask = model_inputs.attention_mask
            
            generated_ids = model.generate(
                model_inputs.input_ids, 
                attention_mask=attention_mask, 
                max_new_tokens=512,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            log_file.write(f"{'='*50}\n")
            log_file.write(f"Row {idx} - User {user_id}: text_id {text_id_1} → {text_id_2}\n")
            log_file.write(f"Text 1 (V={valence_1}, A={arousal_1}): {text_1}\n")
            log_file.write(f"Text 2: {text_2}\n")
            log_file.write(f"Response:\n{response}\n")
            
            score_text_match = re.search(pattern, response)
            if score_text_match == None:
                prediction_text = "|None|"
                change_text = "|None|"
                print(f"Warning: Could not extract predictions for row {idx}")
            else:
                prediction_text = score_text_match.group(1).strip()
                
                prediction_text = re.sub(r'valence_score[_\d]*:\s*', '', prediction_text)
                prediction_text = re.sub(r'arousal_score[_\d]*:\s*', '', prediction_text)
                prediction_text = re.sub(r'valence[_\d]*:\s*', '', prediction_text)
                prediction_text = re.sub(r'arousal[_\d]*:\s*', '', prediction_text)
                prediction_text = prediction_text.strip()
                
                predictions_list = [s.strip() for s in prediction_text.split(',')]
                if len(predictions_list) != 2:
                    print(f"Warning: Expected 2 scores for row {idx}, got {len(predictions_list)}")
                    print(f"Predictions: {prediction_text}")
                    change_text = "|None|"
                else:
                    try:
                        valence_pred = float(predictions_list[0])
                        arousal_pred = float(predictions_list[1])
                        
                        if not (-2 <= valence_pred <= 2 and -2 <= arousal_pred <= 2):
                            print(f"Warning: Scores out of range [-2, +2] for row {idx}: {prediction_text}")
                            change_text = "|None|"
                        else:
                            delta_valence = valence_pred - valence_1
                            delta_arousal = arousal_pred - arousal_1
                            change_text = f"{delta_valence:.2f}, {delta_arousal:.2f}"
                            
                            log_file.write(f"Predicted scores: V={valence_pred}, A={arousal_pred}\n")
                            log_file.write(f"Calculated change: ΔV={delta_valence:.2f}, ΔA={delta_arousal:.2f}\n")
                    except ValueError:
                        print(f"Warning: Could not parse predictions for row {idx}: {prediction_text}")
                        change_text = "|None|"
            
            log_file.write(f"{'='*50}\n\n")
            log_file.flush()
            
            all_predictions.append(prediction_text)
            all_changes.append(change_text)
            
            if args.debug and idx >= 8:
                print(f"Debug Mode: stop after 4th comparison")
                print(f"Row {idx} - User {user_id}: text_id {text_id_1} → {text_id_2}")
                print(f"Text 1 (V={valence_1}, A={arousal_1}): {text_1}")
                print(f"Text 2: {text_2}")
                print(f"Extracted predictions: {prediction_text}")
                print(f"Calculated change: {change_text}")
                break
    
    with open(OUTPUTS_PATH/"subtask2a_predicted_scores_avg.txt", 'w', encoding='utf-8') as output_file:
        for prediction in all_predictions:
            output_file.write(f"{prediction}\n")
    
    with open(OUTPUTS_PATH/"subtask2a_avg.txt", 'w', encoding='utf-8') as output_file:
        for change in all_changes:
            output_file.write(f"{change}\n")
    
    print(f"Done. Total rows processed: {len(all_predictions)}")
    print(f"Saved predicted scores to: {OUTPUTS_PATH}/subtask2a_scores.txt")
    print(f"Saved calculated changes to: {OUTPUTS_PATH}/subtask2a_changes.txt")

def parse_args():
    arg_parser = argparse.ArgumentParser(prog='LLMPrompting', description='Prompt an LLM for next text score prediction')
    arg_parser.add_argument('-d', '--debug', action='store_true', help='Debugging mode')
    args = arg_parser.parse_args()
    return args

def main():
    args = parse_args()
    evaluate_subtask2a(args)

if __name__ == '__main__':
    main()