import json
import time

from huggingface_hub import login

import config
from src.evaluation.hit_at_k import calculate_hit_at_k, get_data_at_hit_at_k_test_format
from src.model.ensemble import Ensemble
from src.model.standard_model import StandardModel
import pandas as pd

login(config.configs['hf_token'])
# word_token_file='../../data/Hit@K/masked words words tokens/masked_test_df_words_words_tokens_30_no_niqqud_new.json'
# char_token_file='../../data/Hit@K/masked words char tokens/masked_test_df_words_char_tokens_30_no_niqqud_new.json'
# results=[]
# data=get_data_at_hit_at_k_test_format(word_token_file)
# model_name=config.configs['word_model_path']
# print(f'==={model_name}===')
# t1 = time.time()
# res = calculate_hit_at_k(5, StandardModel(model_name),data)
# print(f'hit@5 result: {res}')
# t2 = time.time()
# print(f'time: {(t2 - t1) / 60} minutes')
# results.append({'domain':'word','model':model_name,'hit@5':res})
#
# data=get_data_at_hit_at_k_test_format(char_token_file)
# model_name='Ensemble'
# print(f'==={model_name}===')
# t1 = time.time()
# res = calculate_hit_at_k(5, Ensemble(),data)
# print(f'hit@5 result: {res}')
# t2 = time.time()
# print(f'time: {(t2 - t1) / 60} minutes')
# results.append({'domain':'word','model':model_name,'hit@5':res})

data=get_data_at_hit_at_k_test_format(
    '../../data/Hit@K/masked chars and subwords with masked spaces char tokens/masked_spaces_char_tokens_30_no_niqqud_new.json'
)
model_name='Ensemble'
print(f'==={model_name}===')
t1 = time.time()
res = calculate_hit_at_k(5, Ensemble(),data)
print(f'hit@5 result: {res}')
t2 = time.time()
print(f'time: {(t2 - t1) / 60} minutes')
results.append({'domain':'chars','model':model_name,'hit@5':res})


print(pd.DataFrame(results))