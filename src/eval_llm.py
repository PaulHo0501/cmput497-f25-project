import pandas as pd
import numpy as np
import re

SUBTASK1_DATAPATH = './data/train_subtask1.csv'
SUBTASK2B_DATAPATH = './data/train_subtask2b.csv'
SUBTASK2A_DATAPATH = './data/train_subtask2a.csv'

SUBTASK1A_PREDICTION_DATA = './outputs/subtask1_per_sentence.txt'
SUBTASK1B_PREDICTION_DATA = './outputs/subtask1_per_user.txt'
SUBTASK2A_PREDICTION_DATA = './outputs/subtask2a.txt'
SUBTASK2B_PREDICTION_DATA = './outputs/subtask2b.txt'

def read_data_pd(data_path, include_user_id=False):
    # Read valence, arousal, and optionally user_id
    if include_user_id:
        return pd.read_csv(data_path, usecols=['user_id', 'valence', 'arousal'])
    else:
        return pd.read_csv(data_path, usecols=['valence', 'arousal'])


# MSE Eval
def mse_eval(task, checklist, predict_list):
    if task == "1a":
        result = mse_eval_1a(checklist, predict_list)
    elif task == "2a":
        result = mse_eval_2a(checklist, predict_list)
    elif task == "1b":
        result = mse_eval_1b(checklist, predict_list)
    elif task == "2b":
        result = mse_eval_2b(checklist, predict_list)    
    return result
    
def mse_eval_1a(checklist, predict_list):
    # Convert predictions to numpy arrays
    pred_valence = np.array(predict_list['valence'], dtype=float)
    pred_arousal = np.array(predict_list['arousal'], dtype=float)
    
    # Get ground truth values
    true_valence = checklist['valence'].values
    true_arousal = checklist['arousal'].values
    
    # Check if lengths match
    if len(pred_valence) != len(true_valence):
        raise ValueError(f"Length mismatch: predictions ({len(pred_valence)}) vs ground truth ({len(true_valence)})")
    
    # MSE
    mse_valence = np.mean((pred_valence - true_valence) ** 2)
    mse_arousal = np.mean((pred_arousal - true_arousal) ** 2)
    
    # RMSE
    rmse_valence = np.sqrt(mse_valence)
    rmse_arousal = np.sqrt(mse_arousal)
    
    # Calculate average MSE and RMSE
    avg_mse = (mse_valence + mse_arousal) / 2
    avg_rmse = (rmse_valence + rmse_arousal) / 2
    
    results = {
        'mse_valence': mse_valence,
        'mse_arousal': mse_arousal,
        'avg_mse': avg_mse,
        'rmse_valence': rmse_valence,
        'rmse_arousal': rmse_arousal,
        'avg_rmse': avg_rmse
    }
    
    return results

def mse_eval_1b(checklist, predict_list):

    all_pred_valence = []
    all_true_valence = []
    all_pred_arousal = []
    all_true_arousal = []
    
    skipped_users = []
    truncated_users = []
    
    # Iterate through all users
    for user_id in checklist.keys():
        if user_id not in predict_list:
            # print(f"Warning: User {user_id} not found in predictions, skipping...")
            skipped_users.append(user_id)
            continue
            
        true_valence = checklist[user_id]['valence']
        true_arousal = checklist[user_id]['arousal']
        pred_valence = predict_list[user_id]['valence']
        pred_arousal = predict_list[user_id]['arousal']
        
        true_length = len(true_valence)
        pred_length = len(pred_valence)
        
        if pred_length < true_length:
            # print(f"Warning: User {user_id} - prediction too short ({pred_length} < {true_length}), skipping...")
            skipped_users.append(user_id)
            continue
        
        if pred_length > true_length:
            # print(f"Info: User {user_id} - truncating predictions from {pred_length} to {true_length}")
            pred_valence = pred_valence[:true_length]
            pred_arousal = pred_arousal[:true_length]
            truncated_users.append(user_id)
        
        all_pred_valence.extend(pred_valence)
        all_true_valence.extend(true_valence)
        all_pred_arousal.extend(pred_arousal)
        all_true_arousal.extend(true_arousal)
    
    all_pred_valence = np.array(all_pred_valence, dtype=float)
    all_true_valence = np.array(all_true_valence, dtype=float)
    all_pred_arousal = np.array(all_pred_arousal, dtype=float)
    all_true_arousal = np.array(all_true_arousal, dtype=float)
    
    # Calculate MSE
    mse_valence = np.mean((all_pred_valence - all_true_valence) ** 2)
    mse_arousal = np.mean((all_pred_arousal - all_true_arousal) ** 2)
    
    # Calculate RMSE
    rmse_valence = np.sqrt(mse_valence)
    rmse_arousal = np.sqrt(mse_arousal)
    
    # Calculate average MSE and RMSE
    avg_mse = (mse_valence + mse_arousal) / 2
    avg_rmse = (rmse_valence + rmse_arousal) / 2
    
    results = {
        'mse_valence': mse_valence,
        'mse_arousal': mse_arousal,
        'avg_mse': avg_mse,
        'rmse_valence': rmse_valence,
        'rmse_arousal': rmse_arousal,
        'avg_rmse': avg_rmse,
        'total_samples': len(all_pred_valence),
        'num_users': len([u for u in checklist.keys() if u in predict_list and u not in skipped_users]),
        'num_skipped': len(skipped_users),
        'num_truncated': len(truncated_users)
    }
    
    return results

def mse_eval_2a(checklist, predict_list):
    # Convert predictions to numpy arrays
    pred_valence = np.array(predict_list['valence'], dtype=float)
    pred_arousal = np.array(predict_list['arousal'], dtype=float)
    
    # Get ground truth values
    true_valence = checklist['valence'].values
    true_arousal = checklist['arousal'].values
    
    # Check if lengths match
    if len(pred_valence) != len(true_valence):
        raise ValueError(f"Length mismatch: predictions ({len(pred_valence)}) vs ground truth ({len(true_valence)})")
    
    # Calculate MSE
    mse_valence = np.mean((pred_valence - true_valence) ** 2)
    mse_arousal = np.mean((pred_arousal - true_arousal) ** 2)
    
    # Calculate RMSE
    rmse_valence = np.sqrt(mse_valence)
    rmse_arousal = np.sqrt(mse_arousal)
    
    # Calculate average MSE and RMSE
    avg_mse = (mse_valence + mse_arousal) / 2
    avg_rmse = (rmse_valence + rmse_arousal) / 2
    
    results = {
        'mse_valence': mse_valence,
        'mse_arousal': mse_arousal,
        'avg_mse': avg_mse,
        'rmse_valence': rmse_valence,
        'rmse_arousal': rmse_arousal,
        'avg_rmse': avg_rmse
    }
    
    return results

def mse_eval_2b(checklist, predict_list):

    all_pred_valence = []
    all_true_valence = []
    all_pred_arousal = []
    all_true_arousal = []
    
    skipped_users = []
    truncated_users = []
    
    # Iterate through all users
    for user_id in checklist.keys():
        if user_id not in predict_list:
            # print(f"Warning: User {user_id} not found in predictions, skipping...")
            skipped_users.append(user_id)
            continue
            
        true_valence = checklist[user_id]['valence']
        true_arousal = checklist[user_id]['arousal']
        pred_valence = predict_list[user_id]['valence']
        pred_arousal = predict_list[user_id]['arousal']
        
        true_length = len(true_valence)
        pred_length = len(pred_valence)
        
        # If prediction is shorter than ground truth, skip this user
        if pred_length < true_length:
            # print(f"Warning: User {user_id} - prediction too short ({pred_length} < {true_length}), skipping...")
            skipped_users.append(user_id)
            continue
        
        # If prediction is longer than ground truth, truncate to match
        if pred_length > true_length:
            # print(f"Info: User {user_id} - truncating predictions from {pred_length} to {true_length}")
            pred_valence = pred_valence[:true_length]
            pred_arousal = pred_arousal[:true_length]
            truncated_users.append(user_id)
        
        # Accumulate all predictions and ground truth
        all_pred_valence.extend(pred_valence)
        all_true_valence.extend(true_valence)
        all_pred_arousal.extend(pred_arousal)
        all_true_arousal.extend(true_arousal)
    
    # Convert to numpy arrays
    all_pred_valence = np.array(all_pred_valence, dtype=float)
    all_true_valence = np.array(all_true_valence, dtype=float)
    all_pred_arousal = np.array(all_pred_arousal, dtype=float)
    all_true_arousal = np.array(all_true_arousal, dtype=float)
    
    # Calculate MSE
    mse_valence = np.mean((all_pred_valence - all_true_valence) ** 2)
    mse_arousal = np.mean((all_pred_arousal - all_true_arousal) ** 2)
    
    # Calculate RMSE
    rmse_valence = np.sqrt(mse_valence)
    rmse_arousal = np.sqrt(mse_arousal)
    
    # Calculate average MSE and RMSE
    avg_mse = (mse_valence + mse_arousal) / 2
    avg_rmse = (rmse_valence + rmse_arousal) / 2
    
    results = {
        'mse_valence': mse_valence,
        'mse_arousal': mse_arousal,
        'avg_mse': avg_mse,
        'rmse_valence': rmse_valence,
        'rmse_arousal': rmse_arousal,
        'avg_rmse': avg_rmse,
        'total_samples': len(all_pred_valence),
        'num_users': len([u for u in checklist.keys() if u in predict_list and u not in skipped_users]),
        'num_skipped': len(skipped_users),
        'num_truncated': len(truncated_users)
    }
    
    return results

def main():
    # Get validation data
    subtask1_data = read_data_pd(SUBTASK1_DATAPATH)
    subtask1_data_with_user = read_data_pd(SUBTASK1_DATAPATH, include_user_id=True)
    subtask2a_data = read_data_pd(SUBTASK2A_DATAPATH)
    subtask2b_data = read_data_pd(SUBTASK2B_DATAPATH)
    subtask2b_data_with_user = read_data_pd(SUBTASK2B_DATAPATH, include_user_id=True)

    with open(SUBTASK1A_PREDICTION_DATA, 'r', encoding='utf-8') as file:
        valence_subtask1a_score = []
        arousal_subtask1a_score = []
        for line in file:
            predicted = line.strip().split(',')
            valence_subtask1a_score.append(float(predicted[0]))
            arousal_subtask1a_score.append(float(predicted[1]))

    checklist_1b_dict = {}
    for user_id in subtask1_data_with_user['user_id'].unique():
        user_data = subtask1_data_with_user[subtask1_data_with_user['user_id'] == user_id]
        checklist_1b_dict[str(user_id)] = {
            'valence': user_data['valence'].tolist(),
            'arousal': user_data['arousal'].tolist()
        }
    
    # Prompt_llm --user already sorted via id trhough groupby--> This one needed too
    ordered_user_ids = sorted(subtask1_data_with_user['user_id'].unique())
    
    predict_1b_dict = {}
    with open(SUBTASK1B_PREDICTION_DATA, 'r', encoding='utf-8') as file:
        line_count = 0
        user_index = 0
        
        for line in file:
            line = line.strip()
            line_count += 1
            
            if user_index >= len(ordered_user_ids):
                # print(f"Warning: More prediction lines than users in dataset")
                break
                
            user_id = str(ordered_user_ids[user_index])
            user_index += 1
            
            # Skip lines with |None|
            if '|None|' in line:
                # print(f"Line {line_count} (User {user_id}): Skipping - Contains |None|")
                continue
            
            
            line = line.replace('+', '')
            values = [x.strip() for x in line.replace(',', ' ').split() if x.strip()]
            
            if len(values) < 2:
                # print(f"Line {line_count} (User {user_id}): Skipping - Too few values ({len(values)})")
                continue
            
            valence_scores = []
            arousal_scores = []
            
            for i in range(0, len(values), 2):
                if i + 1 < len(values):  
                    try:
                        valence_scores.append(float(values[i]))
                        arousal_scores.append(float(values[i + 1]))
                    except ValueError as e:
                        print(f"Line {line_count} (User {user_id}): Skipping pair at index {i}: {e}")
                        continue
            
            if valence_scores and arousal_scores:
                predict_1b_dict[user_id] = {
                    'valence': valence_scores,
                    'arousal': arousal_scores
                }
            #     print(f"Line {line_count} (User {user_id}): Added {len(valence_scores)} valence/arousal pairs")
            # else:
            #     print(f"Line {line_count} (User {user_id}): Skipping Add")
    
    print(f"\nTotal lines processed: {line_count}")
    print(f"Total users with predictions: {len(predict_1b_dict)}")
    print(f"Total users in checklist: {len(checklist_1b_dict)}\n")
    
    # 2a preprocessing
    with open(SUBTASK2A_PREDICTION_DATA, 'r', encoding='utf-8') as file:
        valence_subtask2a_score = []
        arousal_subtask2a_score = []
        skipped_2a = 0
        
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            
            # Skip lines with |None|
            if '|None|' in line:
                valence_subtask2a_score.append(0.0)
                arousal_subtask2a_score.append(0.0)
                skipped_2a += 1
                continue

            cleaned_line = re.sub(r'\([^)]*\)', '', line)
            cleaned_line = cleaned_line.strip()
            
            parts = re.split(r'[,\s]+', cleaned_line)
            
            # Filter to get only number-like parts
            numbers = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                try:
                    part = part.replace(' ', '')
                    # Handle "- +1" -> "-1" or "-+1" -> "-1"
                    if part.startswith('-+') or part.startswith('+-'):
                        part = '-' + part.lstrip('-+').lstrip('+-')
                    numbers.append(float(part))
                except ValueError:
                    continue
            
            # Should have exactly 2 numbers
            if len(numbers) >= 2:
                valence_subtask2a_score.append(numbers[0])
                arousal_subtask2a_score.append(numbers[1])
            else:
                # print(f"Warning: Line {line_num} has {len(numbers)} numbers, expected 2: '{line}'")
                # Use 0.0 as default for malformed lines
                valence_subtask2a_score.append(0.0)
                arousal_subtask2a_score.append(0.0)
                skipped_2a += 1
    
    print(f"Subtask 2a: Loaded {len(valence_subtask2a_score)} predictions ({skipped_2a} skipped/malformed)")


    # Preprocess 2B
    checklist_2b_dict = {}
    user_prediction_counts = {}  
    
    for user_id in subtask2b_data_with_user['user_id'].unique():
        user_data = subtask2b_data_with_user[subtask2b_data_with_user['user_id'] == user_id]
        total_texts = len(user_data)
        
        # floor(N/2) texts for input, remaining for prediction
        input_count = total_texts // 2
        predict_count = total_texts - input_count
        
        # Get the changes for the prediction portion (last predict_count changes)
        # Note: Changes are from text[i-1] to text[i], so we need the last predict_count changes
        all_valence = user_data['valence'].tolist()
        all_arousal = user_data['arousal'].tolist()
        
        # Take the last predict_count changes
        pred_valence = all_valence[-predict_count:] if predict_count > 0 else []
        pred_arousal = all_arousal[-predict_count:] if predict_count > 0 else []
        
        checklist_2b_dict[str(user_id)] = {
            'valence': pred_valence,
            'arousal': pred_arousal
        }
        user_prediction_counts[str(user_id)] = predict_count
    
    # Get the ordered list of user_ids from the checklist
    ordered_user_ids_2b = sorted(subtask2b_data_with_user['user_id'].unique())
    
    # Get predicted data for subtask 2b (per user format)
    # Format: Each line contains alternating valence_change, arousal_change for one user
    # Lines correspond to users in order (no user_id in file)
    predict_2b_dict = {}
    with open(SUBTASK2B_PREDICTION_DATA, 'r', encoding='utf-8') as file:
        line_count = 0
        user_index = 0
        
        for line in file:
            line = line.strip()
            line_count += 1
            
            # Get the corresponding user_id for this line
            if user_index >= len(ordered_user_ids_2b):
                # print(f"Warning: More prediction lines than users in dataset")
                break
                
            user_id = str(ordered_user_ids_2b[user_index])
            expected_count = user_prediction_counts[user_id]
            user_index += 1
            
            # Skip lines with |None|
            if '|None|' in line:
                # print(f"Line {line_count} (User {user_id}): Skipping - Contains |None|")
                continue
            
            # Remove + signs and split by comma or space
            line = line.replace('+', '')
            values = [x.strip() for x in line.replace(',', ' ').split() if x.strip()]
            
            if len(values) < 2:  # Need at least one valence_change, arousal_change pair
                print(f"Line {line_count} (User {user_id}): Skipping - Too few values ({len(values)})")
                continue
            
            # Parse alternating valence and arousal changes
            valence_changes = []
            arousal_changes = []
            
            for i in range(0, len(values), 2):
                if i + 1 < len(values):
                    try:
                        valence_changes.append(float(values[i]))
                        arousal_changes.append(float(values[i + 1]))
                    except ValueError as e:
                        # print(f"Line {line_count} (User {user_id}): Skipping pair at index {i}: {e}")
                        continue
            
            if valence_changes and arousal_changes:
                predict_2b_dict[user_id] = {
                    'valence': valence_changes,
                    'arousal': arousal_changes
                }
            #     print(f"Line {line_count} (User {user_id}): Added {len(valence_changes)} change pairs (expected {expected_count})")
            # else:
            #     print(f"Line {line_count} (User {user_id}): Skipping - No valid changes")
    
    print(f"\nTotal lines processed: {line_count}")
    print(f"Total users with predictions: {len(predict_2b_dict)}")
    print(f"Total users in checklist: {len(checklist_2b_dict)}\n")
        
    # Prepare prediction dict for subtask 1a
    predict_1a = {
        'valence': valence_subtask1a_score,
        'arousal': arousal_subtask1a_score
    }
    print("=" * 50)
    print("Subtask 1a")
    print("=" * 50)
    results_1a = mse_eval('1a', subtask1_data, predict_1a)
    
    print(f"Valence MSE:  {results_1a['mse_valence']:.6f}")
    print(f"Arousal MSE:  {results_1a['mse_arousal']:.6f}")
    print(f"Average MSE:  {results_1a['avg_mse']:.6f}")
    print()
    print(f"Valence RMSE: {results_1a['rmse_valence']:.6f}")
    print(f"Arousal RMSE: {results_1a['rmse_arousal']:.6f}")
    print(f"Average RMSE: {results_1a['avg_rmse']:.6f}")
    print("=" * 50)
    print()
    
    print("=" * 50)
    print("Subtask 1b")
    print("=" * 50)
    results_1b = mse_eval('1b', checklist_1b_dict, predict_1b_dict)
    
    print(f"Valence MSE:  {results_1b['mse_valence']:.6f}")
    print(f"Arousal MSE:  {results_1b['mse_arousal']:.6f}")
    print(f"Average MSE:  {results_1b['avg_mse']:.6f}")
    print(f"Valence RMSE: {results_1b['rmse_valence']:.6f}")
    print(f"Arousal RMSE: {results_1b['rmse_arousal']:.6f}")
    print(f"Average RMSE: {results_1b['avg_rmse']:.6f}")

    print(f"Total Samples: {results_1b['total_samples']}")
    print(f"Number of Users Evaluated: {results_1b['num_users']}")
    print(f"Number of Users Skipped: {results_1b['num_skipped']}")
    print(f"Number of Users Truncated: {results_1b['num_truncated']}")
    print("=" * 50)

    results_1b = mse_eval('1b', checklist_1b_dict, predict_1b_dict)
    
    print(f"Valence MSE:  {results_1b['mse_valence']:.6f}")
    print(f"Arousal MSE:  {results_1b['mse_arousal']:.6f}")
    print(f"Average MSE:  {results_1b['avg_mse']:.6f}")
    print(f"Valence RMSE: {results_1b['rmse_valence']:.6f}")
    print(f"Arousal RMSE: {results_1b['rmse_arousal']:.6f}")
    print(f"Average RMSE: {results_1b['avg_rmse']:.6f}")

    print(f"Total Samples: {results_1b['total_samples']}")
    print(f"Number of Users Evaluated: {results_1b['num_users']}")
    print(f"Number of Users Skipped: {results_1b['num_skipped']}")
    print(f"Number of Users Truncated: {results_1b['num_truncated']}")
    print("=" * 50)

    # Prepare prediction dict for subtask 2a
    predict_2a = {
        'valence': valence_subtask2a_score,
        'arousal': arousal_subtask2a_score
    }
    
    # Evaluate subtask 2a
    print("=" * 50)
    print("Subtask 2a Evaluation Results")
    print("=" * 50)
    results_2a = mse_eval('2a', subtask2a_data, predict_2a)
    
    print(f"Valence MSE:  {results_2a['mse_valence']:.6f}")
    print(f"Arousal MSE:  {results_2a['mse_arousal']:.6f}")
    print(f"Average MSE:  {results_2a['avg_mse']:.6f}")
    print()
    print(f"Valence RMSE: {results_2a['rmse_valence']:.6f}")
    print(f"Arousal RMSE: {results_2a['rmse_arousal']:.6f}")
    print(f"Average RMSE: {results_2a['avg_rmse']:.6f}")
    print("=" * 50)
    print()


    # Evaluate subtask 2b
    print("=" * 50)
    print("Subtask 2b Evaluation Results")
    print("=" * 50)
    results_2b = mse_eval('2b', checklist_2b_dict, predict_2b_dict)
    
    print(f"Valence MSE:  {results_2b['mse_valence']:.6f}")
    print(f"Arousal MSE:  {results_2b['mse_arousal']:.6f}")
    print(f"Average MSE:  {results_2b['avg_mse']:.6f}")
    print()
    print(f"Valence RMSE: {results_2b['rmse_valence']:.6f}")
    print(f"Arousal RMSE: {results_2b['rmse_arousal']:.6f}")
    print(f"Average RMSE: {results_2b['avg_rmse']:.6f}")
    print()
    print(f"Total Samples: {results_2b['total_samples']}")
    print(f"Number of Users Evaluated: {results_2b['num_users']}")
    print(f"Number of Users Skipped: {results_2b['num_skipped']}")
    print(f"Number of Users Truncated: {results_2b['num_truncated']}")
    print("=" * 50)    
    return results_1a, results_1b


if __name__ == "__main__":
    main()