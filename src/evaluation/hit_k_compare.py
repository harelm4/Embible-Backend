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

results = []
epochs = [10,20, 50]
data_masked = [15, 25, 30]
k = 5


def hit_at_k_eval(model: Model, data: str, domain: str, mask_precent):
    input_data=get_data_at_hit_at_k_test_format(data)
    print(f'==={model.model_path}===')
    t1 = time.time()
    res = calculate_hit_at_k(k, model, input_data)
    print(f'hit@5 result: {res}')
    t2 = time.time()
    t = (t2 - t1)/60
    print(f'time: {t} minutes')
    results.append({'domain': domain, 'mask_precent': mask_precent, 'model': model.model_path, 'hit@5': res,'data':data,'time':t})


login(config.configs['hf_token'])
ensemble=Ensemble()
for mask_precent in data_masked:
    word_token_file = f'../../data/Hit@K/masked words words tokens/masked_test_df_words_words_tokens_{mask_precent}_no_niqqud_new.json'
    char_token_file = f'../../data/Hit@K/masked words char tokens/masked_test_df_words_char_tokens_{mask_precent}_no_niqqud_new.json'
    char_token_with_masked_spaces = \
        f'../../data/Hit@K/masked chars and subwords with masked spaces char tokens/masked_spaces_char_tokens_{mask_precent}_no_niqqud_new.json'

    # try:
    #     hit_at_k_eval(ensemble
    #                   , char_token_file
    #                   , 'char'
    #                   , mask_precent
    #                   )
    # except Exception as e:
    #     print(e)

    try:
        hit_at_k_eval(ensemble
                      , char_token_with_masked_spaces
                      , 'char'
                      , mask_precent
                      )
    except Exception as e:
        print(e)
    for epoch in epochs:
        try:
            hit_at_k_eval(StandardModel(f'Embible/mBert-{epoch}-epochs')
                          , word_token_file
                          , 'word'
                          , mask_precent
                          )
        except Exception as e:
            print(e)

        try:
            hit_at_k_eval(StandardModel(f'Embible/tavbert-{epoch}-epochs')
                          , char_token_with_masked_spaces
                          , 'char'
                          , mask_precent
                          )
        except Exception as e:
            print(e)

        try:
            hit_at_k_eval(StandardModel(f'Embible/tavbert-{epoch}-epochs')
                          , word_token_file
                          , 'word'
                          , mask_precent
                          )
        except Exception as e:
            print(e)

        try:
            hit_at_k_eval(StandardModel(f'Embible/distilBert-{epoch}-epochs')
                          , word_token_file
                          , 'word'
                          , mask_precent
                          )
        except Exception as e:
            print(e)

        try:
            hit_at_k_eval(StandardModel(f'Embible/distilBert-{epoch}-epochs')
                          , char_token_with_masked_spaces
                          , 'char'
                          , mask_precent
                          )
        except Exception as e:
            print(e)
        try:
            hit_at_k_eval(StandardModel(f'Embible/NormalBert-{epoch}-epochs')
                          , word_token_file
                          , 'word'
                          , mask_precent
                          )
        except Exception as e:
            print(e)
        try:
            hit_at_k_eval(StandardModel(f'Embible/NormalBert-{epoch}-epochs')
                          , char_token_with_masked_spaces
                          , 'char'
                          , mask_precent
                          )
        except Exception as e:
            print(e)


res_df = pd.DataFrame(results)
print(res_df)
csv_location = '../../data/'
print(f'writing to {csv_location}')
res_df.to_csv(csv_location + 'hit_at_k_compare_result.csv')
