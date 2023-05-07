import json
import time

from huggingface_hub import login

import config
from src.evaluation.hit_at_k import calculate_hit_at_k, get_data_at_hit_at_k_test_format
from src.model.ensemble import Ensemble
from src.model.model import Model
from src.model.standard_model import StandardModel
import pandas as pd
import pywhatkit

login(config.configs['hf_token'])
results = []
epochs = [10, 20, 50]
data_masked1 = [15, 25, 30]
data_masked2 = [5, 10, 15]
ensemble = Ensemble()


def hit_at_k_eval(model: Model, file: str, k: int):
    input_data = get_data_at_hit_at_k_test_format(file)
    print(f'==={model.model_path}===')
    t1 = time.time()
    res = calculate_hit_at_k(k, model, input_data)
    print(f'hit@5 result: {res}')
    t2 = time.time()
    t = (t2 - t1) / 60
    print(f'time: {t} minutes')
    results.append({'k': k, 'model': model.model_path, 'hit@k': res, 'file': file, 'time': t})


def log_error(model: Model, file: str, k: int, e: str):
    results.append(
        {'k': k, 'model': model.model_path, 'hit@k': None, 'file': file, 'error': e})


def hit_at(k):
    for i, mask_precent in enumerate(data_masked1):
        word_file = f'../../data/Hit@K/masked words char tokens/word_chars_{mask_precent}.json'
        mix_file = f'../../data/Hit@K/masked MIX char tokens/mix_{data_masked2[i]}.json'
        char_file = \
            f'../../data/Hit@K/masked chars and subwords with masked spaces char tokens/chars_{mask_precent}.json'
        files = [word_file, mix_file, char_file]
        for file in files:
            try:
                model = ensemble
                hit_at_k_eval(model, file, k)
            except Exception as e:
                log_error(model, file, k,e)
                print(e)

            for epoch in epochs:
                try:
                    model = StandardModel(f'Embible/mBert-{epoch}-epochs')
                    hit_at_k_eval(model, file, k)
                except Exception as e:
                    log_error(model, file, k,e)
                    print(e)

                try:
                    model = StandardModel(f'Embible/tavbert-{epoch}-epochs')
                    hit_at_k_eval(model, file, k)
                except Exception as e:
                    log_error(model, file, k,e)
                    print(e)

                try:
                    model = StandardModel(f'Embible/distilBert-{epoch}-epochs')
                    hit_at_k_eval(model, file, k)
                except Exception as e:
                    log_error(model, file, k,e)
                    print(e)

    res_df = pd.DataFrame(results)
    print(res_df)
    csv_location = '../../data/results/'
    print(f'writing to {csv_location}')
    res_df.to_csv(csv_location + f'hit_at_{k}_compare_result.csv')


hit_at(1)
hit_at(5)
