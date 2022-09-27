import json
import pandas as pd
import argparse
from typing import List
import os


# 소수 두째자리까지 round
def rounder(num):
    return round(num, 2)

def post_process_metric(x, not_hundred_metrics):
    keys = x.keys()
    for k in keys:
        if max([hm in k for hm in not_hundred_metrics]):
            x[k] = x[k] * 100
        x[k] = rounder(x[k])
    return xs


def parse_logfile(log_folder_name):
    if not log_folder_name.endswith('.json'):
        latest_checkpoint = sorted([d for d in os.listdir(path) if 'checkpoint' in d and not 'ipynb' in d],
                                    key=lambda x : int(x.split('-')[1]))[-1]
        file = os.path.join(log_folder_name, latest_checkpoint, 'trainer_state.json')
    else:
        file = log_folder_name
    logs = pd.read_json(file).log_history
    
    # aggregate logs
    eval_seen = [l for l in logs if 'eval_seen_ppl' in l.keys()]
    eval_unseen = [l for l in logs if 'eval_unseen_ppl' in l.keys()]
    test_seen = [l for l in logs if 'test_seen_ppl' in l.keys()]
    test_unseen = [l for l in logs if 'test_unseen_ppl' in l.keys()]
    
    assert len(eval_seen) == len(eval_unseen) == len(test_seen) == len(test_unseen)
    for i in range(len(eval_seen)):
        eval_seen[i].update(eval_unseen[i])
        eval_seen[i].update(test_seen[i])
        eval_seen[i].update(test_unseen[i])
    
    log_pd = pd.DataFrame(eval_seen)
    keys = log_pd.keys()
    
    metrics = sorted([s for s in set([k.split('seen')[-1].strip('_') for k in keys])
                      if not max([i in s for i in ['num', 'runtime', 'second', 'epoch', 'step']])])
    not_hundred_metrics = [m for m in metrics if not max([i in m for i in ['know_acc', 'loss', 'ppl']])]
    
    new_keys = ['epoch', 'step']
    for m in metrics:
        new_keys.append('eval_seen_'+m)
        new_keys.append('eval_unseen_'+m)
        new_keys.append('test_seen_'+m)
        new_keys.append('test_unseen_'+m)
        
    log_pd_processed = log_pd.apply(lambda x : post_process_metric(x, not_hundred_metrics), axis=1)
    return log_pd_processed[new_keys]   


def write_folder_tocsv(write_file_name,
                       log_file_names : List,
                       ):
    for log_file_name in log_file_names:
        parsed_log = parse_logfile(log_file_name)
        
        # write file as csv
        write_csv_file_name = write_file_name+'.csv'
        parsed_log.to_csv(write_csv_file_name, mode='a')
        with open(write_csv_file_name, 'a') as f:
            f.write(log_file_name)
            f.write('\n\n')

            
def get_log_file_names(log_folder):
    log_file_names = [os.path.join(log_folder,log_file)
                      for log_file in os.listdir(log_folder)]
    return log_file_names



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_folder', type=str)
    parser.add_argument('--write_file_name', type=str)
    args = parser.parse_args()
    
    log_file_names = get_log_file_names(args.log_folder)
    write_folder_tocsv(args.write_file_name, log_file_names)
    print('Done!')