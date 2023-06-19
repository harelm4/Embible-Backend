import time
from typing import List

import pandas as pd
from huggingface_hub import login

import config
from src.evaluation.Classes.CharHitAtK import CharHitAtK
from src.evaluation.Classes.HitAtK import HitAtK
from src.evaluation.Classes.WordHitAtK import WordHitAtK
from src.model.SameLengthAndCharsWordModel import SameLengthAndCharsWordModel
from src.model.SameLengthWordModel import SameLengthWordModel
from src.model.ensemble_v2 import EnsembleV2
from src.model.model import Model
from src.model.standard_model import StandardModel
from src.model.word_model import WordModel
from src.space_predictor.Iterative_space_predictor import  Iterative_space_predictor
word_hit_at_k = WordHitAtK()
char_hit_at_k = CharHitAtK()
login(config.configs['hf_token'])

def hit_at_k_eval(model: Model, file: str, k: int, hit_at_k_strategy: HitAtK, results: List, mask:int):
    data=hit_at_k_strategy.get_data_at_hit_at_k_test_format(file)
    t1 = time.time()
    res = hit_at_k_strategy.calculate(model, data, k)
    print(f'hit@{k} result: {res}')
    t2 = time.time()
    t = (t2 - t1) / 60
    print(f'time: {t} minutes')
    results.append({'k': k, 'model': model.model_path, 'hit@k': res, 'file': file, 'time': t})



def hit_at_k(k: int, hit_at_k_strategy: HitAtK):
    results = []
    for i, mask_precent in enumerate([5, 10, 15]):
        file=f'../../data/Hit@K/mixed test dfs known spaces new P/MIX_test_df_known_spaces_{mask_precent}_percent.json'
        model = EnsembleV2()
        hit_at_k_eval(model, file, k, hit_at_k_strategy,results,mask_precent)
        
    res_df = pd.DataFrame(results)
    csv_location = '../../data/results/test results/known spaces/' + f'Hero-{hit_at_k_strategy.__class__.__name__[:-1]}{k}.csv'
    print(f'writing to {csv_location}')
    res_df.to_csv(csv_location)

for k in [1, 5]:
    for strategy in [char_hit_at_k, word_hit_at_k]:
        hit_at_k(k, strategy)

# model = StandardModel('HeNLP/HeRo')
# print(model.predict('מה ? לך'))

