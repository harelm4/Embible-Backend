import time
from typing import List

import pandas as pd
from huggingface_hub import login

import config
from src.evaluation.Classes.CharHitAtK import CharHitAtK
from src.evaluation.Classes.HitAtK import HitAtK
from src.evaluation.Classes.WordHitAtK import WordHitAtK
from src.model.Char_model import CharModel
from src.model.SameLengthWordModel import SameLengthWordModel
from src.model.ensemble_v2 import EnsembleV2
from src.model.model import Model
from src.model.word_model import WordModel

word_hit_at_k = WordHitAtK()
char_hit_at_k = CharHitAtK()
login(config.configs['hf_token'])
data_masked=[5,10,15]

epochs = [10, 20, 50]
word_models = ['AlephBertGimmel', 'mBert', 'distilBert']

no_fine_tuned = [CharModel('tau/tavbert-he'),WordModel('distilbert-base-uncased'),WordModel('dicta-il/alephbertgimmel-base'),WordModel('bert-base-multilingual-cased')]
fine_tuned = [WordModel(f'Embible/{m}-20-epochs') for m in word_models]+[CharModel('Embible/tavbert-20-epochs')]

def hit_at_k_eval(model: Model, file: str, k: int, hit_at_k_strategy: HitAtK, results: List):
    data=hit_at_k_strategy.get_data_at_hit_at_k_test_format(file)
    t1 = time.time()
    res = hit_at_k_strategy.calculate(model, data, k)
    print(f'hit@{k} result: {res}')
    t2 = time.time()
    t = (t2 - t1) / 60
    print(f'time: {t} minutes')
    results.append({'k': k, 'model': model.model_path, 'hit@k': res, 'file': file, 'time': t})




def hit_at_k(k: int,hit_at_k_strategy: HitAtK, models: List[Model],file_name:str):
    results = []
    for i, mask_precent in enumerate(data_masked):
        'data/Hit@K/masked MIX char tokens/mix_5.json'
        mix_file =f'../../data/Hit@K/masked MIX char tokens/mix_{mask_precent}.json'
        file = mix_file
        for model in models:
            hit_at_k_eval(model, file, k, hit_at_k_strategy,results)


    res_df = pd.DataFrame(results)
    csv_location = f'../../data/results/base/{file_name}.csv'
    print(f'writing to {csv_location}')
    res_df.to_csv(csv_location)

hit_at_k(1, WordHitAtK(), fine_tuned,'fine_tuned')
hit_at_k(1, WordHitAtK(), no_fine_tuned,'no_fine_tuned')
