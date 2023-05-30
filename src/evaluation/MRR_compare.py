import time
from typing import List

import pandas as pd
from huggingface_hub import login

import config
from src.evaluation.Classes.HitAtK import HitAtK
from src.model.SameLengthWordModel import SameLengthWordModel
from src.model.ensemble_v2 import EnsembleV2
from src.model.model import Model
from src.model.word_model import WordModel
from src.evaluation.Classes.MRR_Chars import MRR_CHARS
from src.evaluation.Classes.MRR_Words import MRR_WORDS

MRR_words_eval=MRR_WORDS()
MRR_chars_eval=MRR_CHARS()

login(config.configs['hf_token'])

epochs = [10, 20, 50]
data_masked1 = [15, 25, 30]
data_masked2 = [5,15,25,30]
ensemble = EnsembleV2()
word_models = ['AlephBertGimmel', 'mBert', 'distilBert']

def MRR_eval(model: Model, file: str, MRR_strategy: HitAtK, results: List):
    data=MRR_strategy.get_data_at_hit_at_k_test_format(file)
    t1 = time.time()
    res = MRR_strategy.calculate(model, data)
    print(f'MRR result: {res}')
    t2 = time.time()
    t = (t2 - t1) / 60
    print(f'time: {t} minutes')
    results.append({'model': model.model_path, 'MRR Res': res, 'file': file, 'time': t})


def getModel(baseline: int or str, model: str) -> Model:
    if baseline == 'baseline1':
        return WordModel(model)
    elif baseline == 'baseline2':
        return SameLengthWordModel(model)
    elif baseline == 'ensemble':
        return ensemble


def MRR_CALCULATE(baseline: str, MRR_strategy: HitAtK, models: List[str]):
    results = []
    for i, mask_precent in enumerate(data_masked2):
        #mix_file = f'../../data/Hit@K/masked MIX char tokens/mix_{data_masked2[i]}.json'
        mix_file =f'C:/Users/itaia/Desktop/שנה ד/סמסטר ח/פרויקט גמר/Embible-Backend/data/Hit@K/masked chars and subwords no masked spaces char tokens/masked_test_df_chars_and_subwords_no_masked_spaces_char_tokens_{data_masked2[i]}_no_niqqud_new.json'
        files = [mix_file]
        for file in files:
            for model_name in models:
                if baseline == 'ensemble':
                        model = ensemble
                        MRR_eval(model, file, MRR_strategy,results)
                else:
                    for epoch in epochs:
                            model = getModel(baseline, f'Embible/{model_name}-{epoch}-epochs')
                            MRR_eval(model, file, MRR_strategy,results)


        res_df = pd.DataFrame(results)
        csv_location = 'C:/Users/itaia/Desktop/שנה ד/סמסטר ח/פרויקט גמר/' + f'{baseline}-{MRR_strategy.__class__.__name__[:-1]}.csv'
        print(f'writing to {csv_location}')
        res_df.to_csv(csv_location)


for strategy in [MRR_chars_eval]:
    MRR_CALCULATE('baseline2', MRR_chars_eval ,word_models)
    #MRR_CALCULATE('baseline1',MRR_chars_eval, word_models)
    MRR_CALCULATE('ensemble', MRR_chars_eval, ['ensemble'])
