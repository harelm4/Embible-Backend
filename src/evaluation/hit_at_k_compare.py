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
data_masked2 = [5, 15, 25,30]
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
        #mix_file = f'../../data/Hit@K/masked MIX char tokens/mix_{data_masked2[i]}.json'
        mix_file =f'C:/Users/itaia/Desktop/שנה ד/סמסטר ח/פרויקט גמר/Embible-Backend/data/Hit@K/masked chars and subwords no masked spaces char tokens/masked_test_df_chars_and_subwords_no_masked_spaces_char_tokens_{data_masked2[i]}_no_niqqud_new.json'
        files = [mix_file]
        for file in files:
            for model_name in models:
                if baseline == 'ensemble':
                        model = ensemble
                        hit_at_k_eval(model, file, k, hit_at_k_strategy,results)
                else:
                    for epoch in epochs:
                            model = getModel(baseline, f'Embible/{model_name}-{epoch}-epochs')
                            hit_at_k_eval(model, file, k, hit_at_k_strategy,results)


        res_df = pd.DataFrame(results)
        csv_location = 'C:/Users/itaia/Desktop/שנה ד/סמסטר ח/פרויקט גמר/' + f'{baseline}-{hit_at_k_strategy.__class__.__name__[:-1]}{k}.csv'
        print(f'writing to {csv_location}')
        res_df.to_csv(csv_location)


for k in [1,5]:
    # hit_at_k('baseline1', k, word_hit_at_k, word_models, )
    for strategy in [word_hit_at_k]:
        hit_at_k('baseline2', k, word_hit_at_k   , word_models)
        hit_at_k('baseline1', k, word_hit_at_k, word_models)
        #hit_at_k('ensemble', k, strategy, ['ensemble'])
