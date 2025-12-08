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

SUBTASK2A_PROMPT_TEMPLATE = '''You are evaluating emotional changes between two texts from the same person.

Rules:
(1) Give two scores - one for each rubric.
(2) Evaluate each rubric independently.

TEXT 1 has these scores:
- Valence (emotional positivity): {valence_1}
- Arousal (excitement level): {arousal_1}

Now read TEXT 2 and determine:
1. Did valence increase or decrease? By how much?
2. Did arousal increase or decrease? By how much?

TEXT 1:
```
{text_1}
```

TEXT 2:
```
{text_2}
```

Score Scales:
{valence_scale}
{arousal_scale}

Your reply format:
**Reasoning:** <Explain the emotional change between the two texts>

**Result:** <score for valence_change>, <score for arousal_change>

Where:
- valence_change: change in valence from TEXT 1 to TEXT 2 (range: -4 to +4)
  - Positive number if valence increased (e.g., +1, +2, +3)
  - Negative number if valence decreased (e.g., -1, -2, -3)
  - 0 if no change
- arousal_change: change in arousal from TEXT 1 to TEXT 2 (range: -2 to +2)
  - Positive number if arousal increased (e.g., +1, +2)
  - Negative number if arousal decreased (e.g., -1, -2)
  - 0 if no change

IMPORTANT: Output exactly 2 numbers representing the changes (range: -4 to +4 for Valence change and -2 tp 2 for Arousal change).
'''
SUBTASK2A_RUBRICS = {
    "valence_scale": "Valence Scale: -2 (Very Negative), -1 (Negative), 0 (Neutral), +1 (Positive), +2 (Very Positive)",
    "arousal_scale": "0 (Neutral), +1 (Excited), +2 (Very Excited)"
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
    print("Evaluate Subtask 2a - Emotional Change Detection")
    df = read_data_subtask2a()
    OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    
    df = df.sort_values(['user_id', 'text_id']).reset_index(drop=True)
    
    model, tokenizer = prepare_model()
    pattern = re.compile(PATTERN)
    all_changes = []
    
    print(f"Total rows in dataset: {len(df)}")
    
    with open(LOGS_PATH/"subtask2a.txt", 'w', encoding='utf-8') as log_file:
        for idx in tqdm(range(len(df)), desc="Processing rows"):
            row = df.iloc[idx]
            
            text_2 = row['text']
            user_id = row['user_id']
            text_id_2 = row['text_id']
            
            if idx == 0:
                change_text = "0.0, 0.0"
                all_changes.append(change_text)
                
                log_file.write(f"{'='*50}\n")
                log_file.write(f"Row {idx}: First entry for user {user_id}, using 0.0, 0.0\n")
                log_file.write(f"{'='*50}\n\n")
                
                continue
            
            prev_row = df.iloc[idx - 1]
            
            if prev_row['user_id'] != user_id:
                change_text = "0.0, 0.0"
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
            log_file.write(f"{'='*50}\n\n")
            log_file.flush()
            
            score_text_match = re.search(pattern, response)
            if score_text_match == None:
                change_text = "|None|, |None|"
                print(f"Warning: Could not extract changes for row {idx}")
            else:
                change_text = score_text_match.group(1).strip()
                
                change_text = re.sub(r'valence_change:\s*', '', change_text)
                change_text = re.sub(r'arousal_change:\s*', '', change_text)
                change_text = change_text.strip()
                
                changes_list = [s.strip() for s in change_text.split(',')]
                if len(changes_list) != 2:
                    print(f"Warning: Expected 2 changes for row {idx}, got {len(changes_list)}")
                    print(f"Changes: {change_text}")
                else:
                    try:
                        valence_change = float(changes_list[0])
                        arousal_change = float(changes_list[1])
                        if not (-4 <= valence_change <= 4 and -4 <= arousal_change <= 4):
                            print(f"Warning: Changes out of range [-4, +4] for row {idx}: {change_text}")
                    except ValueError:
                        print(f"Warning: Could not parse changes for row {idx}: {change_text}")
            
            all_changes.append(change_text)
            
            if args.debug and idx >= 8:
                print(f"Debug Mode: stop after 4th comparison")
                print(f"Row {idx} - User {user_id}: text_id {text_id_1} → {text_id_2}")
                print(f"Text 1 (V={valence_1}, A={arousal_1}): {text_1}")
                print(f"Text 2: {text_2}")
                print(f"Extracted changes: {change_text}")
                break
    
    with open(OUTPUTS_PATH/"subtask2a_change.txt", 'w', encoding='utf-8') as output_file:
        for change in all_changes:
            output_file.write(f"{change}\n")
    
    print(f"Done. Total rows processed: {len(all_changes)}")

def parse_args():
    arg_parser = argparse.ArgumentParser(prog='LLMPrompting', description='Prompt an LLM for emotional change detection')
    arg_parser.add_argument('-d', '--debug', action='store_true', help='Debugging mode')
    args = arg_parser.parse_args()
    return args

def main():
    args = parse_args()
    evaluate_subtask2a(args)

if __name__ == '__main__':
    main()