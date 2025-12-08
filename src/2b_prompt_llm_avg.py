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

SUBTASK2B_PROMPT_TEMPLATE = '''You are predicting average emotional scores for a person's future texts based on their previous texts.

Here are the person's previous {num_context_texts} texts (observed segment) with their emotional scores:

{context_texts}

Average scores from the observed segment:
- Average Valence: {avg_valence:.2f}
- Average Arousal: {avg_arousal:.2f}

Score Scales:
{valence_scale}
{arousal_scale}

Now, based on the emotional pattern from these {num_context_texts} texts, predict the AVERAGE emotional scores for the following {num_predict_texts} texts (future segment):

{predict_texts}

Predict:
1. Average Valence score for future texts (range: -2 to +2)
2. Average Arousal score for future texts (range: 0 to +2)

Your reply format:
**Reasoning:** <Explain the emotional patterns and predict the average scores for future texts>

**Result:** <two scores in comma-separated format>

Format: avg_valence_score, avg_arousal_score

IMPORTANT: Output exactly 2 numbers representing the predicted AVERAGE scores for the future segment (range: -2 to +2 each).
'''

SUBTASK2B_RUBRICS = {
    "valence_scale": "Valence Scale: -2 (Very Negative), -1 (Negative), 0 (Neutral), +1 (Positive), +2 (Very Positive)",
    "arousal_scale": "0 (Neutral), +1 (Excited), +2 (Very Excited)"
}

DATA_PATH_SUBTASK2B = './data/train_subtask2b.csv'

def read_data_subtask2b():
    return pd.read_csv(DATA_PATH_SUBTASK2B)

def prepare_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto",
        torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

def evaluate_subtask2b(args):
    print("Evaluate Subtask 2b - Average Future Score Prediction")
    df = read_data_subtask2b()
    OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    
    df = df.sort_values(['user_id', 'text_id']).reset_index(drop=True)
    
    model, tokenizer = prepare_model()
    pattern = re.compile(PATTERN)
    all_predictions = []
    all_changes = []
    
    grouped = df.groupby('user_id')
    
    print(f"Number of users: {len(grouped)}")
    
    with open(LOGS_PATH/"subtask2b.txt", 'w', encoding='utf-8') as log_file:
        count = 0
        for user_id, user_group in tqdm(grouped, desc="Processing users"):
            
            user_group = user_group.reset_index(drop=True)
            num_texts = len(user_group)
            
            if num_texts < 2:
                print(f"Warning: User {user_id} has only {num_texts} text(s), skipping")
                continue
            
            split_point = (num_texts + 1) // 2
            
            context_group = user_group.iloc[:split_point]
            predict_group = user_group.iloc[split_point:]
            
            num_context_texts = len(context_group)
            num_predict_texts = len(predict_group)
            expected_scores = 2 
            
            avg_valence = context_group['valence'].mean()
            avg_arousal = context_group['arousal'].mean()
            
            log_file.write(f"{'='*70}\n")
            log_file.write(f"User {user_id}: {num_texts} total texts\n")
            log_file.write(f"Context: {num_context_texts} texts, Predict: {num_predict_texts} texts\n")
            log_file.write(f"Context Average - Valence: {avg_valence:.2f}, Arousal: {avg_arousal:.2f}\n")
            log_file.write(f"{'='*70}\n\n")
            
            context_texts_formatted = []
            for idx, row in context_group.iterrows():
                text_num = idx + 1
                context_texts_formatted.append(
                    f"Text {text_num} (Valence: {row['valence']}, Arousal: {row['arousal']}):\n{row['text']}"
                )
            context_texts_str = "\n\n".join(context_texts_formatted)
            
            predict_texts_formatted = []
            for i, (idx, row) in enumerate(predict_group.iterrows(), start=1):
                predict_texts_formatted.append(
                    f"Text {num_context_texts + i}:\n{row['text']}"
                )
            predict_texts_str = "\n\n".join(predict_texts_formatted)
            
            prompt = SUBTASK2B_PROMPT_TEMPLATE.format(
                num_context_texts=num_context_texts,
                context_texts=context_texts_str,
                num_predict_texts=num_predict_texts,
                predict_texts=predict_texts_str,
                avg_valence=avg_valence,
                avg_arousal=avg_arousal,
                **SUBTASK2B_RUBRICS
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
                max_new_tokens=1024,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            log_file.write(f"Context Texts:\n{context_texts_str}\n\n")
            log_file.write(f"Predict Texts:\n{predict_texts_str}\n\n")
            log_file.write(f"Response:\n{response}\n")
            
            score_text_match = re.search(pattern, response)
            if score_text_match == None:
                prediction_text = "|None|"
                change_text = "|None|"
                print(f"Warning: Could not extract predictions for user {user_id}")
            else:
                prediction_text = score_text_match.group(1).strip()
                
                prediction_text = re.sub(r'avg_valence_score[_\d]*:\s*', '', prediction_text)
                prediction_text = re.sub(r'avg_arousal_score[_\d]*:\s*', '', prediction_text)
                prediction_text = re.sub(r'valence_score[_\d]*:\s*', '', prediction_text)
                prediction_text = re.sub(r'arousal_score[_\d]*:\s*', '', prediction_text)
                prediction_text = re.sub(r'valence[_\d]*:\s*', '', prediction_text)
                prediction_text = re.sub(r'arousal[_\d]*:\s*', '', prediction_text)
                prediction_text = prediction_text.strip()
                
                scores_list = [s.strip() for s in prediction_text.split(',')]
                if len(scores_list) != expected_scores:
                    print(f"Warning: Expected {expected_scores} scores for user {user_id}, got {len(scores_list)}")
                    print(f"Predictions: {prediction_text}")
                    change_text = "|None|"
                else:
                    try:
                        avg_valence_pred = float(scores_list[0])
                        avg_arousal_pred = float(scores_list[1])
                        
                        if not (-2 <= avg_valence_pred <= 2 and -2 <= avg_arousal_pred <= 2):
                            print(f"Warning: Score out of range [-2, +2] for user {user_id}: {prediction_text}")
                            change_text = "|None|"
                        else:
                            delta_valence = avg_valence_pred - avg_valence
                            delta_arousal = avg_arousal_pred - avg_arousal
                            change_text = f"{delta_valence:.2f}, {delta_arousal:.2f}"
                            
                            log_file.write(f"Predicted average scores: V={avg_valence_pred}, A={avg_arousal_pred}\n")
                            log_file.write(f"Calculated dispositional change: ΔV={delta_valence:.2f}, ΔA={delta_arousal:.2f}\n")
                    except ValueError:
                        print(f"Warning: Could not parse predictions for user {user_id}: {prediction_text}")
                        change_text = "|None|"
            
            log_file.write(f"{'='*70}\n\n")
            log_file.flush()
            
            all_predictions.append(prediction_text)
            all_changes.append(change_text)
            
            count += 1
            if args.debug and count == 4:
                print(f"Debug Mode: stop after 4th user")
                print(f"User {user_id}: {num_context_texts} context texts, {num_predict_texts} predict texts")
                print(f"Expected {expected_scores} scores")
                print(f"Extracted predictions: {prediction_text}")
                print(f"Calculated change: {change_text}")
                break
    
    with open(OUTPUTS_PATH/"subtask2b_predicted_scores_avg.txt", 'w', encoding='utf-8') as output_file:
        for prediction in all_predictions:
            output_file.write(f"{prediction}\n")
    
    with open(OUTPUTS_PATH/"subtask2b_avg.txt", 'w', encoding='utf-8') as output_file:
        for change in all_changes:
            output_file.write(f"{change}\n")
    
    print(f"Done. Total users processed: {len(all_predictions)}")
    print(f"Saved predicted scores to: {OUTPUTS_PATH}/subtask2b_scores.txt")
    print(f"Saved calculated changes to: {OUTPUTS_PATH}/subtask2b_changes.txt")

def parse_args():
    arg_parser = argparse.ArgumentParser(prog='LLMPrompting', description='Prompt an LLM for average future score prediction')
    arg_parser.add_argument('-d', '--debug', action='store_true', help='Debugging mode')
    args = arg_parser.parse_args()
    return args

def main():
    args = parse_args()
    evaluate_subtask2b(args)

if __name__ == '__main__':
    main()