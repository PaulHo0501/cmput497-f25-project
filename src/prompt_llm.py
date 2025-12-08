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
PATTERN = r"\*\*Result:\*\*\s(.+)"

PER_SENTENCE_PROMPT_TEMPLATE = '''You are tasked with evaluating a response based on two scoring rubrics. Provide comprehensive feedback strictly adhering to the scoring rubrics. Follow this with two scores, first score is between -2 and 2, and second score is between 0 and 2.  Do not generate any additional opening, closing, or explanations.

Rules:
(1) Give two scores - one for each rubric.
(2) Evaluate each rubric independently.

Format:
**Reasoning:** <Your feedback>

**Result:** <score for rubric 1>, <score for rubric 2>

Response:
```
{response}
```

Score Rubric 1: {rubric_objective_1}
Score -2: {rubric_score_1_description}
Score -1: {rubric_score_2_description}
Score 0: {rubric_score_3_description}
Score 1: {rubric_score_4_description}
Score 2: {rubric_score_5_description}

Score Rubric 2: {rubric_objective_2}
Score 0: {rubric_score_8_description}
Score 1: {rubric_score_9_description}
Score 2: {rubric_score_10_description}
'''

PER_SENTENCE_RUBRICS = {
    "rubric_objective_1": "Is this text emotionally positive or negative?",
    "rubric_score_1_description": "Very Negative",
    "rubric_score_2_description": "Negative",
    "rubric_score_3_description": "Neutral",
    "rubric_score_4_description": "Positive",
    "rubric_score_5_description": "Very Positive",
    
    "rubric_objective_2": "Does this text show calmness or excitement?",
    "rubric_score_8_description": "Neutral",
    "rubric_score_9_description": "Excited",
    "rubric_score_10_description": "Very Excited",
}

DATA_PATH = './data/train_subtask1.csv'

def read_data():
    df = pl.read_csv(DATA_PATH, try_parse_dates=True)
    return df

def prepare_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

def read_data_pd():
    return pd.read_csv(DATA_PATH)

def evaluate_per_sentence(args):
    print("Evaluate per sentence")
    df = read_data()
    OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    model, tokenizer = prepare_model()
    texts = df['text']
    pattern = re.compile(PATTERN)
    scores = []
    
    with open(LOGS_PATH/"per_sentence.txt", 'w', encoding='utf-8') as log_file:
        for text in tqdm(texts, desc="Evaluating sentences"):
            prompt = PER_SENTENCE_PROMPT_TEMPLATE.format(**PER_SENTENCE_RUBRICS, response=text)
            
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant that evaluates text."},
                {"role": "user", "content": prompt}
            ]
            
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
            log_file.write(f"{'='*20}\n{response}\n{'='*20}\n")
            log_file.flush()
            
            score_text_match = re.search(pattern, response)
            if score_text_match == None:
                score_text = "|None|"
            else:
                score_text = score_text_match.group(1)
            scores.append(score_text)
            
            if args.debug:
                print("Debug Mode: stop after 1st iteration")
                break
    
    with open(OUTPUTS_PATH/"per_sentence.txt", 'w', encoding='utf-8') as output_file:
        for score in scores:
            output_file.write(f"{score}\n")
    print("Done")


PER_USER_PROMPT_TEMPLATE = '''You are evaluating {text_list_size} different texts. For EACH text, provide exactly TWO scores (one for each rubric).

CRITICAL: Output exactly {total_scores} numbers total ({text_list_size} texts × 2 scores each).

    Format your result as ONE line of comma-separated scores:
    score1_text1, score2_text1, score1_text2, score2_text2, ...

    Example for 3 texts:
    If Text 1 scores (1, 0), Text 2 scores (-1, 1), Text 3 scores (2, -1)
    Then output: 1, 0, -1, 1, 2, -1

    Your reply format:
    **Reasoning:** <Brief feedback for each text>

    **Result:** <all {total_scores} scores in one comma-separated line>

    Texts to evaluate:
    {numbered_responses}

    Rubric 1: {rubric_objective_1}
    -2: {rubric_score_1_description}
    -1: {rubric_score_2_description}
    0: {rubric_score_3_description}
    1: {rubric_score_4_description}
    2: {rubric_score_5_description}

    Rubric 2: {rubric_objective_2}
    0: {rubric_score_8_description}
    1: {rubric_score_9_description}
    2: {rubric_score_10_description}

    IMPORTANT: Output exactly {total_scores} numbers (not {text_list_size} numbers).
'''

def evaluate_per_user(args):
    print("Evaluate per user")
    df = read_data_pd()
    OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    model, tokenizer = prepare_model()
    pattern = re.compile(PATTERN)
    all_scores = []
    grouped = df.groupby("user_id")["text"].apply(list)
    print(grouped)
    print(f"Number of users: {len(grouped)}")

    with open(LOGS_PATH/"per_user.txt", 'w', encoding='utf-8') as log_file:
        for user_id in tqdm(grouped.index, desc="Processing users"):
            text_list = grouped[user_id]
            len_text = len(text_list)
            total_scores = len_text * 2
            
            numbered_responses = "\n\n".join([f"{i+1}. {text}" for i, text in enumerate(text_list)])
            
            prompt = PER_USER_PROMPT_TEMPLATE.format(
                **PER_SENTENCE_RUBRICS,
                text_list_size=len_text,
                total_scores=total_scores,
                numbered_responses=numbered_responses
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
            log_file.write(f"{'='*20}\nUser ID: {user_id}\n{response}\n{'='*20}\n")
            log_file.flush()
            
            score_text_match = re.search(pattern, response)
            if score_text_match == None:
                score_text = "|None|"
            else:
                score_text = score_text_match.group(1).strip()
                
                scores_list = [s.strip() for s in score_text.split(',')]
                
                if len(scores_list) != total_scores:
                    print(f"Warning: Expected {total_scores} scores for user {user_id}, got {len(scores_list)}")
                    print(f"Scores: {score_text}")
            
            all_scores.append(score_text)
            
            if args.debug:
                print(f"Debug Mode: stop after 1st user")
                print(f"User {user_id} has {len_text} texts")
                print(f"Expected {total_scores} individual scores")
                print(f"Extracted: {score_text}")
                break
    
    with open(OUTPUTS_PATH/"per_user.txt", 'w', encoding='utf-8') as output_file:
        for score in all_scores:
            output_file.write(f"{score}\n")
    print(f"Done. Total users processed: {len(all_scores)}")




def parse_args():
    arg_parser = argparse.ArgumentParser(prog='LLMPrompting', description='Prompt an LLM for V-A score pair')
    arg_parser.add_argument('-d', '--debug', action='store_true', help='Debugging mode')
    arg_parser.add_argument('-u', '--user', action='store_true', help='Whether or not LLM should evaluate V-A per users or per sentences')
    args = arg_parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.user:
        evaluate_per_user(args)
    else:
        evaluate_per_sentence(args)

if __name__ == '__main__':
    main()