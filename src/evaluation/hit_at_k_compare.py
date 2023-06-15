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
from src.space_predictor.Iterative_space_predictor import  Iterative_space_predictor
word_hit_at_k = WordHitAtK()
char_hit_at_k = CharHitAtK()
login(config.configs['hf_token'])

epochs = [10, 20, 50]
data_masked1 = [15, 25, 30]
data_masked2 = [5, 15, 25,30]
sp=Iterative_space_predictor()
ensemble = EnsembleV2(space_predictor=sp)
word_models = ['AlephBertGimmel', 'mBert', 'distilBert']


def hit_at_k_eval(model: Model, file: str, k: int, hit_at_k_strategy: HitAtK, results: List, mask:int):
    data=hit_at_k_strategy.get_data_at_hit_at_k_test_format(file)
    t1 = time.time()
    res = hit_at_k_strategy.calculate(model, data, k)
    print(f'hit@{k} result: {res}')
    t2 = time.time()
    t = (t2 - t1) / 60
    print(f'time: {t} minutes')
    results.append({'k': k, 'model': model.model_path, 'hit@k': res, 'file': file, 'time': t})
    res_df = pd.DataFrame(results)
    model.model_path =  model.model_path.replace("Embible/", "")
    csv_location = '../../data/results/test results/' + f'{model.model_path}_' \
                                                      f'{hit_at_k_strategy.__class__.__name__[:-1]}{k}_mix_{mask}.csv'
    print(f'writing to {csv_location}')
    res_df.to_csv(csv_location)


def getModel(baseline: int or str, model: str) -> Model:
    if baseline == 'baseline1':
        return WordModel(model)
    elif baseline == 'baseline2':
        return SameLengthWordModel(model)
    elif baseline == 'ensemble':
        return ensemble


def hit_at_k(baseline: str, k: int, hit_at_k_strategy: HitAtK, models: List[str]):
    results = []
    for i, mask_precent in enumerate([5, 10, 15]):
        # "C:\Users\Niv Fono\PycharmProjects\Embible-Backend\data\Hit@K\mixed real test dfs No spaces"
        #mix_file = f'../../data/Hit@K/masked MIX char tokens/mix_{data_masked2[i]}.json'
        # "data/Hit@K/masked MIX char tokens/mix_5.json"
        # "C:\Users\Niv Fono\PycharmProjects\Embible-Backend\data\Hit@K\no masked spaces char tokens\5.json"
        mix_file_W_spaces = f'C:/Users/Niv Fono/PycharmProjects/Embible-Backend/data/Hit@K/mixed real test dfs W spaces/MIX_test_df_{mask_precent}.json'
        mix_file_No_spaces = f'C:/Users/Niv Fono/PycharmProjects/Embible-Backend/data/Hit@K/mixed real test dfs No spaces/MIX_test_df_no_spaces_{mask_precent}.json'
        #mix_file_W_spaces = f'C:/Users/Niv Fono/PycharmProjects/Embible-Backend/data/Hit@K/masked MIX char tokens/mix_{mask_precent}.json'
        #mix_file_No_spaces = f'C:/Users/Niv Fono/PycharmProjects/Embible-Backend/data/Hit@K/no masked spaces char tokens/{mask_precent}.json'

        files = [mix_file_W_spaces,mix_file_No_spaces]
        for file in files:
            for model_name in models:
                if baseline == 'ensemble':
                        model = ensemble
                        if(file==mix_file_No_spaces):
                            model.space_predictor=None
                        else:
                            model.space_predictor=sp

                        hit_at_k_eval(model, file, k, hit_at_k_strategy,results,mask_precent)
                else:
                    for epoch in epochs:
                        model = getModel(baseline, f'Embible/{model_name}-{epoch}-epochs')
                        hit_at_k_eval(model, file, k, hit_at_k_strategy,results,mask_precent)


        res_df = pd.DataFrame(results)
        csv_location = '../../data/results/test results/' + f'{baseline}-{hit_at_k_strategy.__class__.__name__[:-1]}{k} NoHIT+1.csv'
        print(f'writing to {csv_location}')
        res_df.to_csv(csv_location)


# for k in [1, 5]:
#     for strategy in [char_hit_at_k, word_hit_at_k]:
#     # for strategy in [word_hit_at_k]:
#         hit_at_k('ensemble',k, strategy,['ensemble'])

word_models = ['AlephBertGimmel', 'mBert', 'distilBert']

for k in [1,5]:
    for i,strategy in enumerate([char_hit_at_k]):
        hit_at_k('ensemble',k, strategy,['ensemble'])
