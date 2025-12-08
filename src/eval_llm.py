import pandas as pd
import numpy as np
import re

SUBTASK1_DATAPATH = './data/train_subtask1.csv'
SUBTASK2B_DATAPATH = './data/train_subtask2b.csv'
SUBTASK2A_DATAPATH = './data/train_subtask2a.csv'

# Change the files here for direct and indirect llm
SUBTASK1A_PREDICTION_DATA = './outputs/subtask1_per_sentence.txt'
SUBTASK1B_PREDICTION_DATA = './outputs/subtask1_per_user.txt'
SUBTASK2A_PREDICTION_DATA = './outputs/subtask2a_change.txt'
SUBTASK2B_PREDICTION_DATA = './outputs/subtask2b_change.txt'


def read_data_pd(data_path, include_user_id=False):
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
    pred_valence = np.array(predict_list['valence'], dtype=float)
    pred_arousal = np.array(predict_list['arousal'], dtype=float)
    
    true_valence = checklist['valence'].values
    true_arousal = checklist['arousal'].values
    
    if len(pred_valence) != len(true_valence):
        raise ValueError(f"Length mismatch: predictions ({len(pred_valence)}) vs ground truth ({len(true_valence)})")
    
    # MSE
    mse_valence = np.mean((pred_valence - true_valence) ** 2)
    mse_arousal = np.mean((pred_arousal - true_arousal) ** 2)
    
    # RMSE
    rmse_valence = np.sqrt(mse_valence)
    rmse_arousal = np.sqrt(mse_arousal)
    
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
    
    # MSE
    mse_valence = np.mean((all_pred_valence - all_true_valence) ** 2)
    mse_arousal = np.mean((all_pred_arousal - all_true_arousal) ** 2)
    
    # RMSE
    rmse_valence = np.sqrt(mse_valence)
    rmse_arousal = np.sqrt(mse_arousal)
    
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

def mse_eval_2a(checklist_changes, predict_changes):
    pred_delta_valence = np.array(predict_changes['delta_valence'], dtype=float)
    pred_delta_arousal = np.array(predict_changes['delta_arousal'], dtype=float)
    
    true_delta_valence = np.array(checklist_changes['delta_valence'], dtype=float)
    true_delta_arousal = np.array(checklist_changes['delta_arousal'], dtype=float)
    
    valid_mask = ~(np.isnan(pred_delta_valence) | np.isnan(pred_delta_arousal) | 
                   np.isnan(true_delta_valence) | np.isnan(true_delta_arousal))
    
    pred_delta_valence = pred_delta_valence[valid_mask]
    pred_delta_arousal = pred_delta_arousal[valid_mask]
    true_delta_valence = true_delta_valence[valid_mask]
    true_delta_arousal = true_delta_arousal[valid_mask]
    
    if len(pred_delta_valence) != len(true_delta_valence):
        raise ValueError(f"Length mismatch: predictions ({len(pred_delta_valence)}) vs ground truth ({len(true_delta_valence)})")
    
    if len(pred_delta_valence) == 0:
        raise ValueError("No valid predictions after removing NaN values")
    
    mse_valence = np.mean((pred_delta_valence - true_delta_valence) ** 2)
    mse_arousal = np.mean((pred_delta_arousal - true_delta_arousal) ** 2)
    
    rmse_valence = np.sqrt(mse_valence)
    rmse_arousal = np.sqrt(mse_arousal)
    
    avg_mse = (mse_valence + mse_arousal) / 2
    avg_rmse = (rmse_valence + rmse_arousal) / 2
    
    results = {
        'mse_valence': mse_valence,
        'mse_arousal': mse_arousal,
        'avg_mse': avg_mse,
        'rmse_valence': rmse_valence,
        'rmse_arousal': rmse_arousal,
        'avg_rmse': avg_rmse,
        'total_samples': len(pred_delta_valence)
    }
    
    return results


def mse_eval_2b(checklist_changes, predict_changes):
    all_pred_delta_valence = []
    all_true_delta_valence = []
    all_pred_delta_arousal = []
    all_true_delta_arousal = []
    
    skipped_users = []
    
    for user_id in checklist_changes.keys():
        if user_id not in predict_changes:
            skipped_users.append(user_id)
            continue
        
        true_delta_valence = checklist_changes[user_id]['delta_valence']
        true_delta_arousal = checklist_changes[user_id]['delta_arousal']
        pred_delta_valence = predict_changes[user_id]['delta_valence']
        pred_delta_arousal = predict_changes[user_id]['delta_arousal']
        
        # For Subtask 2B
        if isinstance(true_delta_valence, list):
            if len(true_delta_valence) > 0:
                true_delta_valence = true_delta_valence[0]
                true_delta_arousal = true_delta_arousal[0]
            else:
                skipped_users.append(user_id)
                continue
        
        if isinstance(pred_delta_valence, list):
            if len(pred_delta_valence) > 0:
                pred_delta_valence = pred_delta_valence[0]
                pred_delta_arousal = pred_delta_arousal[0]
            else:
                skipped_users.append(user_id)
                continue
        
        if np.isnan(pred_delta_valence) or np.isnan(pred_delta_arousal) or \
           np.isnan(true_delta_valence) or np.isnan(true_delta_arousal):
            skipped_users.append(user_id)
            continue
        
        all_pred_delta_valence.append(pred_delta_valence)
        all_true_delta_valence.append(true_delta_valence)
        all_pred_delta_arousal.append(pred_delta_arousal)
        all_true_delta_arousal.append(true_delta_arousal)
    
    all_pred_delta_valence = np.array(all_pred_delta_valence, dtype=float)
    all_true_delta_valence = np.array(all_true_delta_valence, dtype=float)
    all_pred_delta_arousal = np.array(all_pred_delta_arousal, dtype=float)
    all_true_delta_arousal = np.array(all_true_delta_arousal, dtype=float)
    
    if len(all_pred_delta_valence) == 0:
        raise ValueError("No valid predictions after filtering")
    
    mse_valence = np.mean((all_pred_delta_valence - all_true_delta_valence) ** 2)
    mse_arousal = np.mean((all_pred_delta_arousal - all_true_delta_arousal) ** 2)
    
    rmse_valence = np.sqrt(mse_valence)
    rmse_arousal = np.sqrt(mse_arousal)
    
    avg_mse = (mse_valence + mse_arousal) / 2
    avg_rmse = (rmse_valence + rmse_arousal) / 2
    
    results = {
        'mse_valence': mse_valence,
        'mse_arousal': mse_arousal,
        'avg_mse': avg_mse,
        'rmse_valence': rmse_valence,
        'rmse_arousal': rmse_arousal,
        'avg_rmse': avg_rmse,
        'total_samples': len(all_pred_delta_valence),
        'num_users': len(all_pred_delta_valence),
        'num_skipped': len(skipped_users)
    }
    return results


def main():
    subtask1_data = read_data_pd(SUBTASK1_DATAPATH)
    subtask1_data_with_user = read_data_pd(SUBTASK1_DATAPATH, include_user_id=True)
    subtask2a_data = read_data_pd(SUBTASK2A_DATAPATH)
    subtask2a_data_sorted = pd.read_csv(SUBTASK2A_DATAPATH).sort_values(['user_id', 'text_id']).reset_index(drop=True)
    subtask2b_data = read_data_pd(SUBTASK2B_DATAPATH)
    subtask2b_data_with_user = pd.read_csv(SUBTASK2B_DATAPATH, usecols=['user_id', 'text_id', 'valence', 'arousal'])
    subtask2b_data_with_user = subtask2b_data_with_user.sort_values(['user_id', 'text_id']).reset_index(drop=True)

    # ========== Subtask 1A ==========
    with open(SUBTASK1A_PREDICTION_DATA, 'r', encoding='utf-8') as file:
        valence_subtask1a_score = []
        arousal_subtask1a_score = []
        for line in file:
            predicted = line.strip().split(',')
            valence_subtask1a_score.append(float(predicted[0]))
            arousal_subtask1a_score.append(float(predicted[1]))

    # ========== Subtask 1B ==========
    checklist_1b_dict = {}
    for user_id in subtask1_data_with_user['user_id'].unique():
        user_data = subtask1_data_with_user[subtask1_data_with_user['user_id'] == user_id]
        checklist_1b_dict[str(user_id)] = {
            'valence': user_data['valence'].tolist(),
            'arousal': user_data['arousal'].tolist()
        }
    
    ordered_user_ids = sorted(subtask1_data_with_user['user_id'].unique())
    
    predict_1b_dict = {}
    with open(SUBTASK1B_PREDICTION_DATA, 'r', encoding='utf-8') as file:
        line_count = 0
        user_index = 0
        
        for line in file:
            line = line.strip()
            line_count += 1
            
            if user_index >= len(ordered_user_ids):
                break
                
            user_id = str(ordered_user_ids[user_index])
            user_index += 1
            
            if '|None|' in line:
                continue
            
            line = line.replace('+', '')
            values = [x.strip() for x in line.replace(',', ' ').split() if x.strip()]
            
            if len(values) < 2:
                continue
            
            valence_scores = []
            arousal_scores = []
            
            for i in range(0, len(values), 2):
                if i + 1 < len(values):  
                    try:
                        valence_scores.append(float(values[i]))
                        arousal_scores.append(float(values[i + 1]))
                    except ValueError:
                        continue
            
            if valence_scores and arousal_scores:
                predict_1b_dict[user_id] = {
                    'valence': valence_scores,
                    'arousal': arousal_scores
                }
    
    print(f"Subtask 1B: Total users with predictions: {len(predict_1b_dict)}")
    print(f"Subtask 1B: Total users in checklist: {len(checklist_1b_dict)}\n")
    
    # ========== Subtask 2A ==========
    gt_delta_valence_2a = []
    gt_delta_arousal_2a = []
    
    for idx in range(len(subtask2a_data_sorted)):
        row = subtask2a_data_sorted.iloc[idx]
        
        if idx == 0 or subtask2a_data_sorted.iloc[idx - 1]['user_id'] != row['user_id']:
            continue
        
        prev_row = subtask2a_data_sorted.iloc[idx - 1]
        
        # Δ1 = v_{t+1} - v_t
        gt_delta_valence = row['valence'] - prev_row['valence']
        gt_delta_arousal = row['arousal'] - prev_row['arousal']
        
        gt_delta_valence_2a.append(gt_delta_valence)
        gt_delta_arousal_2a.append(gt_delta_arousal)
    
    pred_delta_valence_2a = []
    pred_delta_arousal_2a = []
    skipped_2a = 0
    
    with open(SUBTASK2A_PREDICTION_DATA, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            
            if '|None|' in line:
                skipped_2a += 1
                continue
            
            cleaned_line = re.sub(r'\([^)]*\)', '', line)
            cleaned_line = cleaned_line.strip()
            
            parts = re.split(r'[,\s]+', cleaned_line)
            
            numbers = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                try:
                    part = part.replace(' ', '')
                    if part.startswith('-+') or part.startswith('+-'):
                        part = '-' + part.lstrip('-+').lstrip('+-')
                    numbers.append(float(part))
                except ValueError:
                    continue
            
            if len(numbers) >= 2:
                pred_delta_valence_2a.append(numbers[0])
                pred_delta_arousal_2a.append(numbers[1])
            else:
                skipped_2a += 1

    if len(pred_delta_valence_2a) > len(gt_delta_valence_2a):
        pred_delta_valence_2a = pred_delta_valence_2a[:len(gt_delta_valence_2a)]
        pred_delta_arousal_2a = pred_delta_arousal_2a[:len(gt_delta_arousal_2a)]
    
    print(f"Subtask 2A: Loaded {len(pred_delta_valence_2a)} predictions (GT: {len(gt_delta_valence_2a)}, skipped: {skipped_2a})")

    # ========== Subtask 2B ==========
    checklist_2b_changes = {}
    
    grouped = subtask2b_data_with_user.groupby('user_id')
    for user_id, user_group in grouped:
        user_group = user_group.reset_index(drop=True)
        num_texts = len(user_group)
        
        if num_texts < 2:
            continue
        
        split_point = (num_texts + 1) // 2
        context_group = user_group.iloc[:split_point]
        predict_group = user_group.iloc[split_point:]
        
        # Δavg = avg(future) - avg(context)
        avg_context_valence = context_group['valence'].mean()
        avg_context_arousal = context_group['arousal'].mean()
        avg_future_valence = predict_group['valence'].mean()
        avg_future_arousal = predict_group['arousal'].mean()
        
        gt_delta_valence = avg_future_valence - avg_context_valence
        gt_delta_arousal = avg_future_arousal - avg_context_arousal
        
        checklist_2b_changes[str(user_id)] = {
            'delta_valence': [gt_delta_valence],
            'delta_arousal': [gt_delta_arousal]
        }
    
    ordered_user_ids_2b = sorted(subtask2b_data_with_user['user_id'].unique())
    
    predict_2b_changes = {}
    with open(SUBTASK2B_PREDICTION_DATA, 'r', encoding='utf-8') as file:
        user_index = 0
        
        for line in file:
            line = line.strip()
            
            if user_index >= len(ordered_user_ids_2b):
                break
                
            user_id = str(ordered_user_ids_2b[user_index])
            user_index += 1
            
            if '|None|' in line:
                continue
            
            line = line.replace('+', '')
            values = [x.strip() for x in line.replace(',', ' ').split() if x.strip()]
            
            if len(values) < 2:
                continue
            
            try:
                delta_valence = float(values[0])
                delta_arousal = float(values[1])
                
                predict_2b_changes[user_id] = {
                    'delta_valence': [delta_valence],
                    'delta_arousal': [delta_arousal]
                }
            except ValueError:
                continue
    
    print(f"Subtask 2B: Total users with predictions: {len(predict_2b_changes)}")
    print(f"Subtask 2B: Total users in checklist: {len(checklist_2b_changes)}\n")
        
    # ========== EVALUATION ==========
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
    print()

    checklist_2a_changes = {
        'delta_valence': gt_delta_valence_2a,
        'delta_arousal': gt_delta_arousal_2a
    }
    
    predict_2a_changes = {
        'delta_valence': pred_delta_valence_2a,
        'delta_arousal': pred_delta_arousal_2a
    }
    
    print("=" * 50)
    print("Subtask 2a Evaluation Results (CHANGES)")
    print("=" * 50)
    try:
        results_2a = mse_eval('2a', checklist_2a_changes, predict_2a_changes)
        
        print(f"Valence MSE:  {results_2a['mse_valence']:.6f}")
        print(f"Arousal MSE:  {results_2a['mse_arousal']:.6f}")
        print(f"Average MSE:  {results_2a['avg_mse']:.6f}")
        print()
        print(f"Valence RMSE: {results_2a['rmse_valence']:.6f}")
        print(f"Arousal RMSE: {results_2a['rmse_arousal']:.6f}")
        print(f"Average RMSE: {results_2a['avg_rmse']:.6f}")
        print()
        print(f"Total Samples: {results_2a['total_samples']}")
    except Exception as e:
        print(f"Error: {e}")
    print("=" * 50)
    print()

    print("=" * 50)
    print("Subtask 2b Evaluation Results (CHANGES)")
    print("=" * 50)
    try:
        results_2b = mse_eval('2b', checklist_2b_changes, predict_2b_changes)
        
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
    except Exception as e:
        print(f"Error: {e}")
    print("=" * 50)
    return results_1a, results_1b


if __name__ == "__main__":
    main()