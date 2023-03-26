import time

from huggingface_hub import login

import config

from src.evaluation.hit_at_k import calculate_hit_at_k
from src.model.ensemble import Ensemble
from src.model.standard_model import StandardModel

login(config.configs['hf_token'])
# model_name=config.configs['char_model_path']

for k in range(10):
    print(f'==={k}===')
    t1=time.time()
    res=calculate_hit_at_k(5, Ensemble(k))
    print(f'hit@5 result: {res}')
    t2=time.time()
    print(f'time: {(t2-t1)/60} minutes')
