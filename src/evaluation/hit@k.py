import json

import config
from src.model.ensamble import Ensamble
from huggingface_hub import login

from src.model.model import Model


def calculate_hit_at_k(k: int, model: Model) -> float:
    """
    calculating hit at k for a model based on test set
    :param k: param of hit at k
    :param model: model to evaluate
    :return: hit@k score
    """
    with open('../../data/test_v2.json', 'r') as r:
        test_data = json.load(r)
    hit_at_ks = []
    for entry in test_data:
        print(entry)
        real_values = entry['missing'].values()
        predictions = model.predict(entry['text']).get_only_k_predictions(k).lst  # list of text parts
        predictions = [x.predictions for x in predictions]  # list of lists of predicion objects
        list_of_preds = []
        for l in predictions:
            preds = []
            for pred in l:
                preds.append(pred.value)
            list_of_preds.append(preds)
        print(list_of_preds)
        print(real_values)
        hit_at_ks.append(_hit_at_k(list_of_preds, [x for x in real_values if x != '']))
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
calculate_hit_at_k(5, Ensamble())