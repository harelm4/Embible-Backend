
import json
import time

import pandas as pd
import config
from src.ensamble import Ensamble
from huggingface_hub import login



"""
real_values-list of words/characters
predictions-list of lists while each list contains k words/characters
example:
k=2
real_values=[שלום,ישראל]
predictions=[[ישראל,יעקב],[חלום,ביטחון]]
return-> 0.5
"""
def hit_at_k(predictions,real_values):
    count_mone,count_mechane=0,0
    for i,word in enumerate(real_values):
        if word in predictions[i]:
            count_mone+=1
        count_mechane+=1
    return count_mone/count_mechane


login(config.configs['hf_token'])
ens=Ensamble()
with open('test.json','r') as r:
    test_data=json.load(r)

for entry in test_data:
    print(entry)
    real_values=entry['missing'].values()
    predictions=ens.predict(entry['text']).get_only_k_predictions(5).lst#list of text parts
    predictions=[x.predictions for x in predictions]#list of lists of predicion objects
    list_of_preds=[]
    for l in predictions:
        preds=[]
        for pred in l:
            preds.append(pred.value)
        list_of_preds.append(preds)
    print(list_of_preds)
    print(hit_at_k(list_of_preds,real_values))
    break
