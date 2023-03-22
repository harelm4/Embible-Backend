import json
import sys
import time

import config
from src.classes.model_result import ModelResult
from src.model.ensemble import Ensemble
from huggingface_hub import login

from src.model.model import Model
from src.model.standard_model import StandardModel


def calculate_hit_at_k(k: int, model: Model,max_loops=sys.maxsize) -> float:
    """
    calculating hit at k for a model based on test set
    :param k: param of hit at k
    :param model: model to evaluate
    :return: hit@k score
    """
    with open('../../data/test_v2.json', 'r') as r:
        test_data = json.load(r)
    hit_at_ks = []
    to_print=''
    for entry_idx,entry in enumerate(test_data):
        real_values = entry['missing'].values()
        predictions = model.predict(entry['text']).get_only_k_predictions(k).lst  # list of text parts
        predictions = [x.predictions for x in predictions]  # list of lists of predicion objects
        list_of_preds = []
        for l in predictions:
            preds = []
            for pred in l:
                preds.append(pred.value)
            list_of_preds.append(preds)

        # === print loading state ===
        for c in to_print:
            sys.stdout.write('\b')
        to_print=f'{entry_idx+1}/{len(test_data)}'
        sys.stdout.write(to_print)
        sys.stdout.flush()
        #====
        hit_at_ks.append(_hit_at_k(list_of_preds, [x for x in real_values if x != '']))
        if entry_idx==max_loops:
            break
    sys.stdout.write('\n')
    return (sum(hit_at_ks) / len(hit_at_ks))



def _hit_at_k(predictions, real_values):
    """
    private function to calculate hit@k
    real_values-list of words/characters
    predictions-list of lists while each list contains k words/characters
    example:
    k=2
    real_values=[שלום,ישראל]
    predictions=[[ישראל,יעקב],[חלום,ביטחון]]
    return-> 0.5
    """
    count_mone, count_mechane = 0, 0
    for i, word in enumerate(real_values):
        if word in predictions[i]:
            count_mone += 1
        count_mechane += 1
    if count_mechane == 0:
        return 0
    return count_mone / count_mechane


login(config.configs['hf_token'])
model_name=config.configs['char_model_path']
# print(f'==={model_name}===')
t1=time.time()
res=calculate_hit_at_k(5, Ensemble())
print(f'hit@5 result: {res}')
t2=time.time()
print(f'time: {(t2-t1)/60} minutes')