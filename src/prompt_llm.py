import argparse
import re
from datetime import datetime
from pathlib import Path

import polars as pl
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = 'cuda'
MODEL_NAME = 'AtlaAI/Selene-1-Mini-Llama-3.1-8B'
NOW = datetime.now()
OUTPUTS_PATH = Path('outputs/')
LOGS_PATH = Path("logs/")
PATTERN = r"\*\*Result:\*\*\s(.+)"
PER_SENTENCE_PROMPT_TEMPLATE = '''
You are tasked with evaluating a response based on two scoring rubrics that serve as the evaluation standards. Provide a comprehensive feedback strictly adhering to the scoring rubrics, without any general evaluation. Follow this with two scores, each between -2 and 2, referring to the scoring rubric. Avoid generating any additional opening, closing, or explanations.

  Here are some rules of the evaluation:
  (1) You will be asked score the response strictly based on the rubric. You have to give two scores, each score corresponds with a rubric.
  (2) You do not need to evaluate the second rubric based on the first rubric. Treat the score as individual. Do not let your judgement of the first score affect the second score.

  Your reply should strictly follow this format:
  **Reasoning:** <Your feedback>

  **Result:** <a pair of scores, each between -2 and 2, separated by a comma>

  Here is the data:

  Response:
  ```
  {response}
  ```

  Score Rubric 1:
  {rubric_objective_1}
  Score -2: {rubric_score_1_description}
  Score -1 {rubric_score_2_description}
  Score 0: {rubric_score_3_description}
  Score 1: {rubric_score_4_description}
  Score 2: {rubric_score_5_description}

  Score Rubric 2:
  {rubric_objective_2}
  Score -2: {rubric_score_6_description}
  Score -1 {rubric_score_7_description}
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
    "rubric_objective_2": "Is this text shows calmness or excitement?",
    "rubric_score_6_description": "Very Calm",
    "rubric_score_7_description": "Calm",
    "rubric_score_8_description": "Neutral",
    "rubric_score_9_description": "Excited",
    "rubric_score_10_description": "Very Excited",
}

DATA_PATH = './data/train_subtask1.csv'

def read_data():
    df = pl.read_csv(DATA_PATH, try_parse_dates=True)
    return df

def prepare_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer


def evaluate_per_sentence(args):
    print("Evaluate per sentence")
    df = read_data()
    OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)
    LOGS_PATH.mkdir(parents=True, exist_ok=True)
    model, tokenizer = prepare_model()
    texts = df['text']
    pattern = re.compile(PATTERN)
    scores = []
    with open(LOGS_PATH/f"per_sentece_{NOW}.txt", 'w', encoding='utf-8') as log_file:
        for text in tqdm(texts):
            prompt = PER_SENTENCE_PROMPT_TEMPLATE.format(**PER_SENTENCE_RUBRICS, response=text)
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
            attention_mask = model_inputs.attention_mask
            generated_ids = model.generate(model_inputs.input_ids, attention_mask=attention_mask, max_new_tokens=512, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            log_file.write(f"{'='*20}\n{response}\n{'='*20}\n")
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
        output_file.writelines(scores)
    print("Done")


def evaluate_per_user(args):
    print(args)

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
