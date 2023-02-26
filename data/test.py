import time

import pandas as pd
import config
from src.model.ensamble import Ensamble
from huggingface_hub import login

login(config.configs['hf_token'])
ens=Ensamble()


data_list=list(pd.read_csv('test.csv', sep='\t', encoding='utf-8')['verse'])
time_list=[]
word_len_list=[]
for txt in data_list:
    print(f"========")
    print(txt)
    t0=time.time()
    res = ens.predict(txt)
    t1 = time.time()
    t=t1-t0
    word_len_list.append(txt.count("?"))
    time_list.append(t)
    print(f"--{t}--")
    print(res)
    print(f"========")
print(f'{len(time_list)} entries')
print(f'avg time: {sum(time_list)/len(time_list)}')
print(f'avg ? count: {sum(word_len_list)/len(word_len_list)}')
# print(pd.read_csv('test.csv', sep='\t', encoding='utf-8')['missing_dictionary'])