import time
from typing import List

import pandas as pd
from huggingface_hub import login

import config
from src.evaluation.Classes.CharHitAtK import CharHitAtK
from src.evaluation.Classes.HitAtK import HitAtK
from src.evaluation.Classes.WordHitAtK import WordHitAtK
from src.model.SameLengthWordModel import SameLengthWordModel
from src.model.ensemble_v2 import EnsembleV2
from src.model.model import Model
from src.model.word_model import WordModel

word_hit_at_k = WordHitAtK()
char_hit_at_k = CharHitAtK()
login(config.configs['hf_token'])

epochs = [10, 20, 50]
data_masked1 = [15, 25, 30]
data_masked2 = [5, 10, 15]
ensemble = EnsembleV2()
word_models = ['AlephBertGimmel', 'mBert', 'distilBert']




def hit_at_k_eval(model: Model, file: str, k: int, hit_at_k_strategy: HitAtK, results: List):
    data=hit_at_k_strategy.get_data_at_hit_at_k_test_format(file)
    t1 = time.time()
    res = hit_at_k_strategy.calculate(model, data, k)
    print(f'hit@{k} result: {res}')
    t2 = time.time()
    t = (t2 - t1) / 60
    print(f'time: {t} minutes')
    results.append({'k': k, 'model': model.model_path, 'hit@k': res, 'file': file, 'time': t})


def log_error(model: Model, file: str, k: int, e: str, results: List):
    results.append(
        {'k': k, 'model': model.model_path, 'hit@k': None, 'file': file, 'error': e})


def getModel(baseline: int or str, model: str) -> Model:
    if baseline == 'baseline1':
        return WordModel(model)
    elif baseline == 'baseline2':
        return SameLengthWordModel(model)
    elif baseline == 'ensemble':
        return ensemble


def hit_at_k(baseline: str, k: int, hit_at_k_strategy: HitAtK, models: List[str]):
    results = []
    for i, mask_precent in enumerate(data_masked1):
        mix_file = f'../../data/Hit@K/masked MIX char tokens/mix_{data_masked2[i]}.json'
        files = [mix_file]
        for file in files:
            for model_name in models:
                if baseline == 'ensemble':
                    # try:
                        model = ensemble
                        hit_at_k_eval(model, file, k, hit_at_k_strategy,results)
                    # except Exception as e:
                    #     print(e)
                    #     log_error(model, file, k, e, results)

                else:
                    for epoch in epochs:
                        # try:
                            model = getModel(baseline, f'Embible/{model_name}-{epoch}-epochs')
                            hit_at_k_eval(model, file, k, hit_at_k_strategy,results)
                        # except Exception as e:
                        #     print(e)
                        #     log_error(model, file, k, e, results)

        res_df = pd.DataFrame(results)
        csv_location = '../../data/results/' + f'{baseline}-{hit_at_k_strategy.__class__.__name__[:-1]}{k}.csv'
        print(f'writing to {csv_location}')
        res_df.to_csv(csv_location)


for k in [1,5]:
    # hit_at_k('baseline1', k, word_hit_at_k, word_models, )
    for strategy in [char_hit_at_k]:
        # hit_at_k('baseline2', k, strategy, word_models)
        hit_at_k('ensemble', k, strategy, ['ensemble'])
